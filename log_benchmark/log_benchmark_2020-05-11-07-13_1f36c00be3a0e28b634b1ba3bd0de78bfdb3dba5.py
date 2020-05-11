
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4ae10bef98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 07:14:04.166279
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 07:14:04.170112
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 07:14:04.173880
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 07:14:04.177421
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4aece82438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354502.9375
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 260784.6719
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 174913.7812
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 109807.9844
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 66799.6094
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 41102.4297
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 26726.4023
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 18484.9531
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 13469.7559
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 10205.3564

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.3689148e+00 -1.3985094e-01 -1.8332413e+00 -6.8546534e-01
  -8.5295272e-01  1.2205422e+00 -2.8780907e-01  1.0963564e+00
  -1.5413201e+00  2.9741287e-01  1.3878990e+00 -6.2277246e-01
  -6.5904558e-01 -5.2358389e-01  6.6946310e-01  1.2168818e+00
   7.5076032e-01 -5.0297275e-02  1.3247974e+00  1.4497451e+00
   1.4794542e+00  1.9449019e+00  1.6780412e+00  3.1596744e-01
  -1.5098268e+00 -9.2732954e-01 -1.9437963e+00  1.4344882e+00
   7.5053513e-01 -7.9927123e-01 -7.7672350e-01  6.9315773e-01
  -2.7065459e-01 -2.0102334e+00 -4.0649873e-01 -4.7751421e-01
   2.2334149e-01 -2.2147809e-01 -1.8057802e+00  4.8697862e-01
   9.6104813e-01  7.0178694e-01  3.2462293e-01  1.3209735e+00
   2.9320383e-01 -1.1218597e+00  5.3953189e-01 -4.2626056e-01
  -5.2989465e-01 -5.7003272e-01  3.5922515e-01 -5.6833321e-01
   6.1275989e-01  5.5324334e-01 -1.9761351e+00 -9.4076872e-02
  -1.0423094e+00 -6.6622186e-01 -4.8526996e-01 -6.7389384e-03
   5.9588379e-01  5.4241073e-01  1.9017475e+00  8.1106991e-01
  -4.9574077e-03  1.6742927e+00  6.5897071e-01  4.1990727e-02
  -6.9263965e-02 -2.0238476e+00 -2.8801143e-01 -8.3580446e-01
  -2.0565372e+00  4.2344606e-01 -2.1501915e-01 -2.9601932e-02
  -1.2279062e-01  1.4103171e+00 -8.4870648e-01  1.8019888e+00
   1.0741341e+00 -1.6195188e+00 -9.4634092e-01 -5.3630340e-01
   1.1647772e+00 -6.9221056e-01 -4.8413455e-01  3.2504010e-01
  -7.7487364e-02  5.5736250e-01  1.7240146e+00 -2.7928057e-01
  -1.1367137e+00 -5.0972319e-01 -2.1019968e-01  7.3344213e-01
   1.0662117e+00  3.2948196e-02 -7.7098560e-01  8.6581546e-01
   2.7979726e-01 -1.3089558e+00 -8.6832762e-01 -7.7844524e-01
   5.8723229e-01 -8.8865268e-01  9.4632441e-01  2.7367401e-01
   6.3253945e-01 -2.2182000e-01  4.5453101e-01  1.0738454e+00
   1.2295216e+00  1.5824826e-01  7.0752931e-01  6.1588413e-01
  -9.4586492e-01 -3.0014336e-01 -4.0685907e-01 -4.7402352e-01
   1.9863245e-01  6.1513300e+00  6.7995892e+00  5.2567120e+00
   6.5993137e+00  5.1921406e+00  5.1416159e+00  7.7783475e+00
   7.8403511e+00  4.9895678e+00  6.4901290e+00  6.0986671e+00
   5.2994337e+00  4.2121425e+00  5.0462489e+00  7.0400763e+00
   7.4844551e+00  5.7856541e+00  5.9073491e+00  5.2538767e+00
   6.1772146e+00  7.1358576e+00  6.3232455e+00  5.8078318e+00
   5.3665514e+00  5.2649293e+00  4.5159273e+00  5.7658958e+00
   5.8538532e+00  5.3760729e+00  6.2175407e+00  6.5817132e+00
   5.4243188e+00  6.8024993e+00  7.2787719e+00  5.7180715e+00
   6.8705344e+00  5.4986944e+00  6.6244626e+00  6.9352832e+00
   7.6406355e+00  5.8762331e+00  6.4062037e+00  6.1693969e+00
   6.0120778e+00  5.5144029e+00  5.7149901e+00  7.6856556e+00
   8.0471325e+00  7.0095272e+00  5.5481815e+00  5.1269479e+00
   7.6095977e+00  6.6326332e+00  7.9324675e+00  5.7235365e+00
   7.6272106e+00  5.3931956e+00  6.7695980e+00  6.3426466e+00
   1.2931449e+00  1.5549313e+00  8.1528920e-01  2.2725611e+00
   1.8034238e-01  1.5506113e-01  1.8719141e+00  2.0698509e+00
   3.4873176e-01  8.9312184e-01  2.4420655e-01  7.5179780e-01
   4.6563876e-01  1.4644480e+00  1.7134236e+00  1.1706023e+00
   8.2386571e-01  5.1424873e-01  2.7466130e-01  1.4799631e+00
   2.5181460e-01  6.7100543e-01  2.6742773e+00  1.2119470e+00
   4.3384755e-01  9.8243451e-01  4.7688890e-01  4.5867562e-01
   7.3007613e-01  3.8955009e-01  5.9708512e-01  1.4531877e+00
   2.9769897e+00  1.8961818e+00  1.1522423e+00  2.0828013e+00
   3.2320738e-01  8.7244964e-01  3.4337103e-01  6.8774641e-01
   3.3321840e-01  2.0193982e-01  1.6117842e+00  2.2985804e-01
   2.7912898e+00  1.2505015e+00  3.6251575e-01  2.1763334e+00
   4.6917015e-01  1.8859261e+00  4.4849908e-01  1.9693756e-01
   1.4344320e+00  1.1343164e+00  1.3679910e-01  2.0196826e+00
   1.1602225e+00  1.3679563e+00  1.3320069e+00  1.1107514e+00
   4.5639133e-01  6.1778545e-01  4.0183747e-01  5.4410559e-01
   1.1558186e+00  3.8169944e-01  1.1766113e+00  1.9630861e-01
   9.3839961e-01  4.6196485e-01  1.1806263e+00  2.2868770e-01
   2.6830304e-01  1.1367195e+00  8.3641464e-01  5.1043761e-01
   1.6453912e+00  3.7916589e-01  6.2963206e-01  5.3098464e-01
   8.9966124e-01  2.4366140e-01  1.5652655e+00  9.8592675e-01
   8.0901730e-01  9.8360997e-01  1.4365435e+00  3.6351585e-01
   7.3827052e-01  7.3612475e-01  3.8148940e-01  1.5014127e+00
   1.5737796e+00  9.1667175e-01  2.2567520e+00  2.0840794e-01
   1.9773858e+00  8.6158103e-01  2.7090222e-01  3.4011167e-01
   4.7688496e-01  1.4623697e+00  1.8646257e+00  5.2621007e-01
   7.7269226e-01  6.8537277e-01  2.9751711e+00  6.5912974e-01
   1.1944467e+00  1.9787502e+00  5.5391872e-01  3.8691545e-01
   6.5066093e-01  1.1334648e+00  4.2468148e-01  1.1496965e+00
   4.4328463e-01  1.8861439e+00  1.3132669e+00  1.1342909e+00
   8.5180283e-02  7.5881429e+00  6.7463369e+00  7.7035441e+00
   6.0002317e+00  6.6935639e+00  7.3747654e+00  7.3792052e+00
   6.7457056e+00  7.5609999e+00  6.4237843e+00  5.7483048e+00
   4.9814725e+00  7.5831876e+00  6.7139831e+00  5.5291924e+00
   7.0633078e+00  7.1978655e+00  8.3526325e+00  7.4234343e+00
   6.7684579e+00  6.5307598e+00  6.5907454e+00  8.1963596e+00
   7.5584707e+00  5.5111027e+00  7.0601439e+00  6.8427076e+00
   7.0775809e+00  7.1296077e+00  6.6947546e+00  6.8208385e+00
   6.5780549e+00  7.9654074e+00  6.4378920e+00  6.6430750e+00
   7.1218224e+00  6.1726260e+00  6.7589979e+00  6.6868291e+00
   5.5153422e+00  7.7438049e+00  6.0919785e+00  7.5858269e+00
   5.3375678e+00  6.4223056e+00  7.8949671e+00  8.1817818e+00
   6.3371615e+00  7.8367543e+00  7.2798676e+00  7.9791565e+00
   5.3352518e+00  7.2524662e+00  5.6669474e+00  5.2058206e+00
   6.4541645e+00  5.8060212e+00  7.2633543e+00  6.6203771e+00
  -2.0411208e+00 -6.7699857e+00  7.4022522e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 07:14:14.664484
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                     96.01
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 07:14:14.668511
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9236.28
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 07:14:14.673267
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.8099
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 07:14:14.676553
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -826.174
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139959237967264
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139958010756568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139958010757072
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139958010319368
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139958010319872
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139958010320376

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4ae0d7b160> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.630856
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.591963
grad_step = 000002, loss = 0.557798
grad_step = 000003, loss = 0.520693
grad_step = 000004, loss = 0.480205
grad_step = 000005, loss = 0.444796
grad_step = 000006, loss = 0.426136
grad_step = 000007, loss = 0.417275
grad_step = 000008, loss = 0.397003
grad_step = 000009, loss = 0.372168
grad_step = 000010, loss = 0.353815
grad_step = 000011, loss = 0.340426
grad_step = 000012, loss = 0.327304
grad_step = 000013, loss = 0.311857
grad_step = 000014, loss = 0.294995
grad_step = 000015, loss = 0.277974
grad_step = 000016, loss = 0.261436
grad_step = 000017, loss = 0.245692
grad_step = 000018, loss = 0.230974
grad_step = 000019, loss = 0.217930
grad_step = 000020, loss = 0.207172
grad_step = 000021, loss = 0.197744
grad_step = 000022, loss = 0.188064
grad_step = 000023, loss = 0.177969
grad_step = 000024, loss = 0.167848
grad_step = 000025, loss = 0.158546
grad_step = 000026, loss = 0.150001
grad_step = 000027, loss = 0.141621
grad_step = 000028, loss = 0.132822
grad_step = 000029, loss = 0.123833
grad_step = 000030, loss = 0.115557
grad_step = 000031, loss = 0.108447
grad_step = 000032, loss = 0.102126
grad_step = 000033, loss = 0.095883
grad_step = 000034, loss = 0.089641
grad_step = 000035, loss = 0.083801
grad_step = 000036, loss = 0.078484
grad_step = 000037, loss = 0.073465
grad_step = 000038, loss = 0.068568
grad_step = 000039, loss = 0.063847
grad_step = 000040, loss = 0.059364
grad_step = 000041, loss = 0.055162
grad_step = 000042, loss = 0.051231
grad_step = 000043, loss = 0.047638
grad_step = 000044, loss = 0.044270
grad_step = 000045, loss = 0.041022
grad_step = 000046, loss = 0.037977
grad_step = 000047, loss = 0.035216
grad_step = 000048, loss = 0.032624
grad_step = 000049, loss = 0.030078
grad_step = 000050, loss = 0.027608
grad_step = 000051, loss = 0.025394
grad_step = 000052, loss = 0.023419
grad_step = 000053, loss = 0.021552
grad_step = 000054, loss = 0.019764
grad_step = 000055, loss = 0.018105
grad_step = 000056, loss = 0.016609
grad_step = 000057, loss = 0.015229
grad_step = 000058, loss = 0.013942
grad_step = 000059, loss = 0.012725
grad_step = 000060, loss = 0.011578
grad_step = 000061, loss = 0.010555
grad_step = 000062, loss = 0.009656
grad_step = 000063, loss = 0.008833
grad_step = 000064, loss = 0.008052
grad_step = 000065, loss = 0.007354
grad_step = 000066, loss = 0.006743
grad_step = 000067, loss = 0.006184
grad_step = 000068, loss = 0.005665
grad_step = 000069, loss = 0.005204
grad_step = 000070, loss = 0.004798
grad_step = 000071, loss = 0.004438
grad_step = 000072, loss = 0.004117
grad_step = 000073, loss = 0.003832
grad_step = 000074, loss = 0.003584
grad_step = 000075, loss = 0.003373
grad_step = 000076, loss = 0.003185
grad_step = 000077, loss = 0.003016
grad_step = 000078, loss = 0.002875
grad_step = 000079, loss = 0.002758
grad_step = 000080, loss = 0.002650
grad_step = 000081, loss = 0.002558
grad_step = 000082, loss = 0.002484
grad_step = 000083, loss = 0.002423
grad_step = 000084, loss = 0.002368
grad_step = 000085, loss = 0.002322
grad_step = 000086, loss = 0.002285
grad_step = 000087, loss = 0.002259
grad_step = 000088, loss = 0.002242
grad_step = 000089, loss = 0.002237
grad_step = 000090, loss = 0.002232
grad_step = 000091, loss = 0.002218
grad_step = 000092, loss = 0.002167
grad_step = 000093, loss = 0.002126
grad_step = 000094, loss = 0.002120
grad_step = 000095, loss = 0.002129
grad_step = 000096, loss = 0.002124
grad_step = 000097, loss = 0.002091
grad_step = 000098, loss = 0.002063
grad_step = 000099, loss = 0.002056
grad_step = 000100, loss = 0.002060
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002052
grad_step = 000102, loss = 0.002028
grad_step = 000103, loss = 0.002007
grad_step = 000104, loss = 0.001999
grad_step = 000105, loss = 0.001998
grad_step = 000106, loss = 0.001991
grad_step = 000107, loss = 0.001973
grad_step = 000108, loss = 0.001956
grad_step = 000109, loss = 0.001946
grad_step = 000110, loss = 0.001942
grad_step = 000111, loss = 0.001938
grad_step = 000112, loss = 0.001929
grad_step = 000113, loss = 0.001916
grad_step = 000114, loss = 0.001903
grad_step = 000115, loss = 0.001891
grad_step = 000116, loss = 0.001884
grad_step = 000117, loss = 0.001878
grad_step = 000118, loss = 0.001874
grad_step = 000119, loss = 0.001870
grad_step = 000120, loss = 0.001866
grad_step = 000121, loss = 0.001862
grad_step = 000122, loss = 0.001858
grad_step = 000123, loss = 0.001852
grad_step = 000124, loss = 0.001847
grad_step = 000125, loss = 0.001841
grad_step = 000126, loss = 0.001835
grad_step = 000127, loss = 0.001827
grad_step = 000128, loss = 0.001816
grad_step = 000129, loss = 0.001802
grad_step = 000130, loss = 0.001789
grad_step = 000131, loss = 0.001776
grad_step = 000132, loss = 0.001765
grad_step = 000133, loss = 0.001755
grad_step = 000134, loss = 0.001747
grad_step = 000135, loss = 0.001740
grad_step = 000136, loss = 0.001732
grad_step = 000137, loss = 0.001725
grad_step = 000138, loss = 0.001719
grad_step = 000139, loss = 0.001714
grad_step = 000140, loss = 0.001717
grad_step = 000141, loss = 0.001744
grad_step = 000142, loss = 0.001841
grad_step = 000143, loss = 0.002060
grad_step = 000144, loss = 0.002229
grad_step = 000145, loss = 0.002033
grad_step = 000146, loss = 0.001727
grad_step = 000147, loss = 0.001750
grad_step = 000148, loss = 0.001951
grad_step = 000149, loss = 0.001826
grad_step = 000150, loss = 0.001654
grad_step = 000151, loss = 0.001811
grad_step = 000152, loss = 0.001827
grad_step = 000153, loss = 0.001729
grad_step = 000154, loss = 0.001675
grad_step = 000155, loss = 0.001727
grad_step = 000156, loss = 0.001765
grad_step = 000157, loss = 0.001613
grad_step = 000158, loss = 0.001666
grad_step = 000159, loss = 0.001710
grad_step = 000160, loss = 0.001615
grad_step = 000161, loss = 0.001621
grad_step = 000162, loss = 0.001633
grad_step = 000163, loss = 0.001632
grad_step = 000164, loss = 0.001579
grad_step = 000165, loss = 0.001575
grad_step = 000166, loss = 0.001614
grad_step = 000167, loss = 0.001557
grad_step = 000168, loss = 0.001545
grad_step = 000169, loss = 0.001559
grad_step = 000170, loss = 0.001549
grad_step = 000171, loss = 0.001541
grad_step = 000172, loss = 0.001508
grad_step = 000173, loss = 0.001511
grad_step = 000174, loss = 0.001523
grad_step = 000175, loss = 0.001505
grad_step = 000176, loss = 0.001493
grad_step = 000177, loss = 0.001476
grad_step = 000178, loss = 0.001473
grad_step = 000179, loss = 0.001482
grad_step = 000180, loss = 0.001475
grad_step = 000181, loss = 0.001471
grad_step = 000182, loss = 0.001464
grad_step = 000183, loss = 0.001451
grad_step = 000184, loss = 0.001444
grad_step = 000185, loss = 0.001435
grad_step = 000186, loss = 0.001428
grad_step = 000187, loss = 0.001425
grad_step = 000188, loss = 0.001424
grad_step = 000189, loss = 0.001432
grad_step = 000190, loss = 0.001465
grad_step = 000191, loss = 0.001533
grad_step = 000192, loss = 0.001689
grad_step = 000193, loss = 0.001768
grad_step = 000194, loss = 0.001734
grad_step = 000195, loss = 0.001504
grad_step = 000196, loss = 0.001405
grad_step = 000197, loss = 0.001515
grad_step = 000198, loss = 0.001599
grad_step = 000199, loss = 0.001527
grad_step = 000200, loss = 0.001400
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001402
grad_step = 000202, loss = 0.001506
grad_step = 000203, loss = 0.001532
grad_step = 000204, loss = 0.001453
grad_step = 000205, loss = 0.001380
grad_step = 000206, loss = 0.001397
grad_step = 000207, loss = 0.001444
grad_step = 000208, loss = 0.001448
grad_step = 000209, loss = 0.001404
grad_step = 000210, loss = 0.001366
grad_step = 000211, loss = 0.001379
grad_step = 000212, loss = 0.001409
grad_step = 000213, loss = 0.001408
grad_step = 000214, loss = 0.001377
grad_step = 000215, loss = 0.001351
grad_step = 000216, loss = 0.001360
grad_step = 000217, loss = 0.001385
grad_step = 000218, loss = 0.001387
grad_step = 000219, loss = 0.001368
grad_step = 000220, loss = 0.001348
grad_step = 000221, loss = 0.001341
grad_step = 000222, loss = 0.001349
grad_step = 000223, loss = 0.001359
grad_step = 000224, loss = 0.001361
grad_step = 000225, loss = 0.001354
grad_step = 000226, loss = 0.001342
grad_step = 000227, loss = 0.001332
grad_step = 000228, loss = 0.001329
grad_step = 000229, loss = 0.001332
grad_step = 000230, loss = 0.001337
grad_step = 000231, loss = 0.001341
grad_step = 000232, loss = 0.001342
grad_step = 000233, loss = 0.001343
grad_step = 000234, loss = 0.001342
grad_step = 000235, loss = 0.001341
grad_step = 000236, loss = 0.001337
grad_step = 000237, loss = 0.001333
grad_step = 000238, loss = 0.001327
grad_step = 000239, loss = 0.001323
grad_step = 000240, loss = 0.001319
grad_step = 000241, loss = 0.001316
grad_step = 000242, loss = 0.001313
grad_step = 000243, loss = 0.001309
grad_step = 000244, loss = 0.001306
grad_step = 000245, loss = 0.001304
grad_step = 000246, loss = 0.001302
grad_step = 000247, loss = 0.001301
grad_step = 000248, loss = 0.001300
grad_step = 000249, loss = 0.001299
grad_step = 000250, loss = 0.001297
grad_step = 000251, loss = 0.001295
grad_step = 000252, loss = 0.001293
grad_step = 000253, loss = 0.001292
grad_step = 000254, loss = 0.001290
grad_step = 000255, loss = 0.001290
grad_step = 000256, loss = 0.001290
grad_step = 000257, loss = 0.001295
grad_step = 000258, loss = 0.001312
grad_step = 000259, loss = 0.001365
grad_step = 000260, loss = 0.001497
grad_step = 000261, loss = 0.001828
grad_step = 000262, loss = 0.002079
grad_step = 000263, loss = 0.002250
grad_step = 000264, loss = 0.001819
grad_step = 000265, loss = 0.001485
grad_step = 000266, loss = 0.001517
grad_step = 000267, loss = 0.001607
grad_step = 000268, loss = 0.001661
grad_step = 000269, loss = 0.001440
grad_step = 000270, loss = 0.001367
grad_step = 000271, loss = 0.001630
grad_step = 000272, loss = 0.001430
grad_step = 000273, loss = 0.001367
grad_step = 000274, loss = 0.001426
grad_step = 000275, loss = 0.001395
grad_step = 000276, loss = 0.001442
grad_step = 000277, loss = 0.001317
grad_step = 000278, loss = 0.001334
grad_step = 000279, loss = 0.001382
grad_step = 000280, loss = 0.001371
grad_step = 000281, loss = 0.001291
grad_step = 000282, loss = 0.001308
grad_step = 000283, loss = 0.001319
grad_step = 000284, loss = 0.001311
grad_step = 000285, loss = 0.001307
grad_step = 000286, loss = 0.001261
grad_step = 000287, loss = 0.001296
grad_step = 000288, loss = 0.001294
grad_step = 000289, loss = 0.001276
grad_step = 000290, loss = 0.001260
grad_step = 000291, loss = 0.001266
grad_step = 000292, loss = 0.001271
grad_step = 000293, loss = 0.001261
grad_step = 000294, loss = 0.001258
grad_step = 000295, loss = 0.001239
grad_step = 000296, loss = 0.001253
grad_step = 000297, loss = 0.001254
grad_step = 000298, loss = 0.001243
grad_step = 000299, loss = 0.001244
grad_step = 000300, loss = 0.001230
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001233
grad_step = 000302, loss = 0.001236
grad_step = 000303, loss = 0.001232
grad_step = 000304, loss = 0.001230
grad_step = 000305, loss = 0.001221
grad_step = 000306, loss = 0.001219
grad_step = 000307, loss = 0.001217
grad_step = 000308, loss = 0.001218
grad_step = 000309, loss = 0.001218
grad_step = 000310, loss = 0.001213
grad_step = 000311, loss = 0.001211
grad_step = 000312, loss = 0.001206
grad_step = 000313, loss = 0.001203
grad_step = 000314, loss = 0.001201
grad_step = 000315, loss = 0.001200
grad_step = 000316, loss = 0.001199
grad_step = 000317, loss = 0.001197
grad_step = 000318, loss = 0.001196
grad_step = 000319, loss = 0.001194
grad_step = 000320, loss = 0.001191
grad_step = 000321, loss = 0.001190
grad_step = 000322, loss = 0.001187
grad_step = 000323, loss = 0.001185
grad_step = 000324, loss = 0.001184
grad_step = 000325, loss = 0.001182
grad_step = 000326, loss = 0.001180
grad_step = 000327, loss = 0.001180
grad_step = 000328, loss = 0.001181
grad_step = 000329, loss = 0.001185
grad_step = 000330, loss = 0.001195
grad_step = 000331, loss = 0.001214
grad_step = 000332, loss = 0.001255
grad_step = 000333, loss = 0.001315
grad_step = 000334, loss = 0.001431
grad_step = 000335, loss = 0.001541
grad_step = 000336, loss = 0.001672
grad_step = 000337, loss = 0.001596
grad_step = 000338, loss = 0.001413
grad_step = 000339, loss = 0.001204
grad_step = 000340, loss = 0.001175
grad_step = 000341, loss = 0.001302
grad_step = 000342, loss = 0.001378
grad_step = 000343, loss = 0.001308
grad_step = 000344, loss = 0.001178
grad_step = 000345, loss = 0.001156
grad_step = 000346, loss = 0.001235
grad_step = 000347, loss = 0.001287
grad_step = 000348, loss = 0.001264
grad_step = 000349, loss = 0.001182
grad_step = 000350, loss = 0.001140
grad_step = 000351, loss = 0.001161
grad_step = 000352, loss = 0.001207
grad_step = 000353, loss = 0.001228
grad_step = 000354, loss = 0.001189
grad_step = 000355, loss = 0.001146
grad_step = 000356, loss = 0.001131
grad_step = 000357, loss = 0.001149
grad_step = 000358, loss = 0.001174
grad_step = 000359, loss = 0.001177
grad_step = 000360, loss = 0.001155
grad_step = 000361, loss = 0.001129
grad_step = 000362, loss = 0.001122
grad_step = 000363, loss = 0.001132
grad_step = 000364, loss = 0.001144
grad_step = 000365, loss = 0.001144
grad_step = 000366, loss = 0.001131
grad_step = 000367, loss = 0.001117
grad_step = 000368, loss = 0.001113
grad_step = 000369, loss = 0.001119
grad_step = 000370, loss = 0.001126
grad_step = 000371, loss = 0.001127
grad_step = 000372, loss = 0.001122
grad_step = 000373, loss = 0.001113
grad_step = 000374, loss = 0.001106
grad_step = 000375, loss = 0.001103
grad_step = 000376, loss = 0.001104
grad_step = 000377, loss = 0.001106
grad_step = 000378, loss = 0.001109
grad_step = 000379, loss = 0.001111
grad_step = 000380, loss = 0.001112
grad_step = 000381, loss = 0.001111
grad_step = 000382, loss = 0.001109
grad_step = 000383, loss = 0.001107
grad_step = 000384, loss = 0.001103
grad_step = 000385, loss = 0.001100
grad_step = 000386, loss = 0.001096
grad_step = 000387, loss = 0.001093
grad_step = 000388, loss = 0.001090
grad_step = 000389, loss = 0.001088
grad_step = 000390, loss = 0.001086
grad_step = 000391, loss = 0.001084
grad_step = 000392, loss = 0.001083
grad_step = 000393, loss = 0.001081
grad_step = 000394, loss = 0.001080
grad_step = 000395, loss = 0.001079
grad_step = 000396, loss = 0.001078
grad_step = 000397, loss = 0.001077
grad_step = 000398, loss = 0.001076
grad_step = 000399, loss = 0.001075
grad_step = 000400, loss = 0.001075
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001077
grad_step = 000402, loss = 0.001081
grad_step = 000403, loss = 0.001090
grad_step = 000404, loss = 0.001111
grad_step = 000405, loss = 0.001148
grad_step = 000406, loss = 0.001227
grad_step = 000407, loss = 0.001328
grad_step = 000408, loss = 0.001495
grad_step = 000409, loss = 0.001554
grad_step = 000410, loss = 0.001557
grad_step = 000411, loss = 0.001321
grad_step = 000412, loss = 0.001116
grad_step = 000413, loss = 0.001082
grad_step = 000414, loss = 0.001199
grad_step = 000415, loss = 0.001272
grad_step = 000416, loss = 0.001181
grad_step = 000417, loss = 0.001075
grad_step = 000418, loss = 0.001094
grad_step = 000419, loss = 0.001173
grad_step = 000420, loss = 0.001177
grad_step = 000421, loss = 0.001101
grad_step = 000422, loss = 0.001056
grad_step = 000423, loss = 0.001084
grad_step = 000424, loss = 0.001127
grad_step = 000425, loss = 0.001135
grad_step = 000426, loss = 0.001108
grad_step = 000427, loss = 0.001078
grad_step = 000428, loss = 0.001055
grad_step = 000429, loss = 0.001057
grad_step = 000430, loss = 0.001080
grad_step = 000431, loss = 0.001095
grad_step = 000432, loss = 0.001087
grad_step = 000433, loss = 0.001062
grad_step = 000434, loss = 0.001048
grad_step = 000435, loss = 0.001046
grad_step = 000436, loss = 0.001050
grad_step = 000437, loss = 0.001058
grad_step = 000438, loss = 0.001064
grad_step = 000439, loss = 0.001062
grad_step = 000440, loss = 0.001052
grad_step = 000441, loss = 0.001042
grad_step = 000442, loss = 0.001036
grad_step = 000443, loss = 0.001035
grad_step = 000444, loss = 0.001036
grad_step = 000445, loss = 0.001039
grad_step = 000446, loss = 0.001043
grad_step = 000447, loss = 0.001046
grad_step = 000448, loss = 0.001045
grad_step = 000449, loss = 0.001042
grad_step = 000450, loss = 0.001039
grad_step = 000451, loss = 0.001036
grad_step = 000452, loss = 0.001032
grad_step = 000453, loss = 0.001028
grad_step = 000454, loss = 0.001026
grad_step = 000455, loss = 0.001025
grad_step = 000456, loss = 0.001023
grad_step = 000457, loss = 0.001022
grad_step = 000458, loss = 0.001021
grad_step = 000459, loss = 0.001021
grad_step = 000460, loss = 0.001021
grad_step = 000461, loss = 0.001021
grad_step = 000462, loss = 0.001021
grad_step = 000463, loss = 0.001021
grad_step = 000464, loss = 0.001023
grad_step = 000465, loss = 0.001026
grad_step = 000466, loss = 0.001031
grad_step = 000467, loss = 0.001041
grad_step = 000468, loss = 0.001057
grad_step = 000469, loss = 0.001085
grad_step = 000470, loss = 0.001123
grad_step = 000471, loss = 0.001187
grad_step = 000472, loss = 0.001251
grad_step = 000473, loss = 0.001329
grad_step = 000474, loss = 0.001339
grad_step = 000475, loss = 0.001298
grad_step = 000476, loss = 0.001175
grad_step = 000477, loss = 0.001057
grad_step = 000478, loss = 0.001011
grad_step = 000479, loss = 0.001049
grad_step = 000480, loss = 0.001113
grad_step = 000481, loss = 0.001128
grad_step = 000482, loss = 0.001081
grad_step = 000483, loss = 0.001024
grad_step = 000484, loss = 0.001009
grad_step = 000485, loss = 0.001039
grad_step = 000486, loss = 0.001070
grad_step = 000487, loss = 0.001069
grad_step = 000488, loss = 0.001037
grad_step = 000489, loss = 0.001008
grad_step = 000490, loss = 0.001004
grad_step = 000491, loss = 0.001020
grad_step = 000492, loss = 0.001037
grad_step = 000493, loss = 0.001039
grad_step = 000494, loss = 0.001027
grad_step = 000495, loss = 0.001009
grad_step = 000496, loss = 0.000998
grad_step = 000497, loss = 0.000997
grad_step = 000498, loss = 0.001004
grad_step = 000499, loss = 0.001014
grad_step = 000500, loss = 0.001018
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

  date_run                              2020-05-11 07:14:38.349206
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.273357
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 07:14:38.355349
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    0.1928
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 07:14:38.362788
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.151881
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 07:14:38.370764
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.92967
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

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json 

                        date_run  ...            metric_name
0   2020-05-11 07:14:04.166279  ...    mean_absolute_error
1   2020-05-11 07:14:04.170112  ...     mean_squared_error
2   2020-05-11 07:14:04.173880  ...  median_absolute_error
3   2020-05-11 07:14:04.177421  ...               r2_score
4   2020-05-11 07:14:14.664484  ...    mean_absolute_error
5   2020-05-11 07:14:14.668511  ...     mean_squared_error
6   2020-05-11 07:14:14.673267  ...  median_absolute_error
7   2020-05-11 07:14:14.676553  ...               r2_score
8   2020-05-11 07:14:38.349206  ...    mean_absolute_error
9   2020-05-11 07:14:38.355349  ...     mean_squared_error
10  2020-05-11 07:14:38.362788  ...  median_absolute_error
11  2020-05-11 07:14:38.370764  ...               r2_score

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
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do timeseries 





 ************************************************************************************************************************

  vision_mnist 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_cnn/mnist 

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 300758.80it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 392433.00it/s]  5%|▌         | 499712/9912422 [00:00<00:18, 522024.82it/s]  8%|▊         | 819200/9912422 [00:00<00:13, 689078.99it/s] 12%|█▏        | 1146880/9912422 [00:00<00:09, 886559.40it/s] 15%|█▍        | 1482752/9912422 [00:01<00:07, 1115340.06it/s] 18%|█▊        | 1826816/9912422 [00:01<00:05, 1386911.25it/s] 22%|██▏       | 2179072/9912422 [00:01<00:04, 1639191.26it/s] 26%|██▌       | 2531328/9912422 [00:01<00:03, 1879689.76it/s] 29%|██▉       | 2899968/9912422 [00:01<00:03, 2124148.39it/s] 33%|███▎      | 3268608/9912422 [00:01<00:02, 2329748.65it/s] 37%|███▋      | 3653632/9912422 [00:01<00:02, 2526533.38it/s] 41%|████      | 4038656/9912422 [00:01<00:02, 2665814.06it/s] 45%|████▍     | 4431872/9912422 [00:01<00:01, 2810762.63it/s] 48%|████▊     | 4743168/9912422 [00:02<00:01, 2888785.83it/s] 51%|█████     | 5054464/9912422 [00:02<00:01, 2921430.14it/s] 55%|█████▍    | 5447680/9912422 [00:02<00:01, 3011547.74it/s] 59%|█████▉    | 5840896/9912422 [00:02<00:01, 3235264.45it/s] 62%|██████▏   | 6176768/9912422 [00:02<00:01, 3254921.52it/s] 66%|██████▌   | 6512640/9912422 [00:02<00:01, 3192877.94it/s] 70%|██████▉   | 6905856/9912422 [00:02<00:00, 3243863.42it/s] 74%|███████▎  | 7307264/9912422 [00:02<00:00, 3438375.14it/s] 78%|███████▊  | 7733248/9912422 [00:02<00:00, 3447783.90it/s] 82%|████████▏ | 8159232/9912422 [00:03<00:00, 3457741.97it/s] 86%|████████▌ | 8544256/9912422 [00:03<00:00, 3549004.43it/s] 90%|████████▉ | 8904704/9912422 [00:03<00:00, 3427080.17it/s] 93%|█████████▎| 9265152/9912422 [00:03<00:00, 3361552.91it/s] 98%|█████████▊| 9674752/9912422 [00:03<00:00, 3543254.37it/s]9920512it [00:03, 2777954.43it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 147458.14it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 306531.13it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 399417.56it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 553759.83it/s]1654784it [00:00, 2824360.06it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53872.11it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ae4525be0> <class 'mlmodels.model_tch.torchhub.Model'>
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Processing...
Done!

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a81c79b00> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ae4525cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ae44e9f28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a96ef1da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a81c79b00> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a81c79d68> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a81c760f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a96ef1e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a81c760f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a81c76080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/cnn/mnist 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff7bbfa81d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=ab0b8d1dfeca96756b716c46dd909278c3621fb8d588f76d29aa58169171ca4c
  Stored in directory: /tmp/pip-ephem-wheel-cache-zc5iyhn4/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.0.2; however, version 20.1 is available.
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff75bcd5da0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 51s
   57344/17464789 [..............................] - ETA: 39s
  106496/17464789 [..............................] - ETA: 32s
  229376/17464789 [..............................] - ETA: 18s
  491520/17464789 [..............................] - ETA: 11s
  999424/17464789 [>.............................] - ETA: 6s 
 2023424/17464789 [==>...........................] - ETA: 3s
 4005888/17464789 [=====>........................] - ETA: 1s
 6889472/17464789 [==========>...................] - ETA: 0s
10018816/17464789 [================>.............] - ETA: 0s
13148160/17464789 [=====================>........] - ETA: 0s
16179200/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 07:16:10.585293: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 07:16:10.589415: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 07:16:10.589616: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561eb363d5b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 07:16:10.589631: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6053 - accuracy: 0.5040
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6758 - accuracy: 0.4994
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6837 - accuracy: 0.4989
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 4s - loss: 7.6137 - accuracy: 0.5035
12000/25000 [=============>................] - ETA: 4s - loss: 7.6117 - accuracy: 0.5036
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6277 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6250 - accuracy: 0.5027
15000/25000 [=================>............] - ETA: 3s - loss: 7.6319 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6427 - accuracy: 0.5016
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6495 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
25000/25000 [==============================] - 9s 378us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 07:16:27.309972
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 07:16:27.309972  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}] 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'} 

  #### Setup Model   ############################################## 
Using TensorFlow backend.
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
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 75)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 75, 40)            1720      
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
2020-05-11 07:16:33.654262: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 07:16:33.660077: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 07:16:33.660585: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558969faeee0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 07:16:33.660635: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fcdccc0dc18> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4566 - crf_viterbi_accuracy: 0.2267 - val_loss: 1.4457 - val_crf_viterbi_accuracy: 0.2667

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 40, 50)       250         input_2[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_2[0][0]                
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
dense_2 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fcda898d898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9733 - accuracy: 0.4800
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8736 - accuracy: 0.4865 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8353 - accuracy: 0.4890
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6697 - accuracy: 0.4998
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6232 - accuracy: 0.5028
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6272 - accuracy: 0.5026
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
11000/25000 [============>.................] - ETA: 4s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6265 - accuracy: 0.5026
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6381 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 3s - loss: 7.6707 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6494 - accuracy: 0.5011
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6530 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6561 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6467 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6886 - accuracy: 0.4986
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6858 - accuracy: 0.4988
25000/25000 [==============================] - 9s 377us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fcd6394a3c8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:38:13, 24.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<6:46:42, 35.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.67M/862M [00:00<4:44:03, 50.4kB/s].vector_cache/glove.6B.zip:   1%|          | 6.67M/862M [00:00<3:18:01, 72.0kB/s].vector_cache/glove.6B.zip:   1%|          | 10.7M/862M [00:00<2:18:04, 103kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.8M/862M [00:00<1:36:26, 147kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.2M/862M [00:00<1:07:21, 209kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.3M/862M [00:01<46:57, 298kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:01<32:44, 425kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.5M/862M [00:01<22:55, 604kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:01<16:06, 856kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:01<11:18, 1.21MB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:01<08:03, 1.70MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:01<05:47, 2.35MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:01<04:11, 3.24MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.6M/862M [00:01<03:09, 4.30MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:01<02:23, 5.66MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<03:31, 3.82MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<04:02, 3.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<03:36, 3.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:04<02:37, 5.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<1:20:58, 165kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<58:28, 229kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:06<40:59, 325kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<33:13, 400kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<25:14, 527kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:08<17:48, 745kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<16:33, 799kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<13:37, 971kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:10<09:42, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<10:29, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<09:19, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<07:21, 1.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:12<05:14, 2.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<12:54, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:15<12:51, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<09:27, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:15<06:47, 1.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<10:41, 1.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:16<09:27, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<06:53, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<07:31, 1.72MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<07:09, 1.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<05:08, 2.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<08:54, 1.44MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<07:32, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<05:27, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<07:38, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<07:23, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:22<05:18, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<08:27, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<07:03, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<05:05, 2.49MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<08:04, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:33, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<04:44, 2.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:14, 1.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<06:47, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:28<04:52, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<10:57, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:30<06:12, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<21:16, 585kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<15:49, 786kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<11:12, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<11:56, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<09:52, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<07:01, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<10:19, 1.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<08:15, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<05:52, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<14:50, 825kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<11:52, 1.03MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<08:25, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<10:31, 1.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<09:04, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<06:28, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:09, 1.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:03, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:48, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<07:21, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:08, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:04, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:09, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:46<03:41, 3.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<2:10:49, 90.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<1:33:15, 127kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<1:05:03, 182kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<1:00:12, 196kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<43:08, 274kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<31:47, 370kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<24:00, 490kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<16:51, 694kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<17:25, 671kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<13:54, 840kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<09:48, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<14:56, 778kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<12:04, 962kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<08:30, 1.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<17:50, 647kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<13:32, 853kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<09:31, 1.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<22:48, 503kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<16:55, 678kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<13:29, 846kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<10:17, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:53, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:31, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<05:22, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:52, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:50, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<05:35, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<10:33, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:12, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<05:49, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<16:43, 666kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<12:39, 879kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<08:55, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<17:24, 636kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<13:22, 827kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<09:24, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<19:17, 570kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<14:24, 763kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<11:40, 936kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<09:04, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<06:28, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<08:38, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<07:36, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<05:25, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<08:38, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<07:38, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<05:31, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<06:29, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<06:02, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:21<04:18, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<17:40, 603kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<13:43, 776kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<09:43, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<10:01, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<08:30, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:25<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<11:22, 925kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<09:27, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:27<06:41, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<12:44, 821kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:13, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:29<07:13, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<14:29, 716kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<11:52, 874kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<08:27, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<08:28, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<07:11, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:33<05:06, 2.01MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<09:42, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<07:46, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:35<05:30, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<14:44, 690kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<11:31, 882kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<08:06, 1.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<16:03, 629kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<12:22, 816kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:39<08:41, 1.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<22:43, 441kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<17:23, 577kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<12:13, 817kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<12:47, 780kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<10:21, 961kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:43<07:17, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<15:35, 635kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<11:57, 827kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:45<08:25, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<12:08, 810kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<09:57, 986kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<07:04, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:57, 1.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<06:54, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<04:57, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:49<03:55, 2.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<11:47:42, 13.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<8:15:47, 19.5kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<5:46:14, 27.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<4:03:57, 39.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<2:52:04, 55.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<2:00:04, 79.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<1:27:06, 109kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<1:02:21, 153kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<43:36, 218kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<33:19, 284kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<24:12, 391kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<16:58, 555kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<15:33, 604kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<11:34, 811kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<08:08, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<15:55, 586kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<11:52, 785kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<09:38, 961kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<07:57, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:03<05:37, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<18:45, 490kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<14:19, 641kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:05<10:03, 908kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<15:34, 586kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<11:58, 761kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<08:25, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<11:38, 778kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<08:56, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<07:32, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<06:00, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<04:16, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<08:19, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<06:28, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:02, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:31, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:56, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<07:24, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<06:13, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:26, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<06:21, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:33, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<04:00, 2.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:11, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:51, 1.78MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:29, 2.46MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:14, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:54, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<03:31, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:47, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:53, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<03:31, 2.40MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:07, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:30, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<03:13, 2.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:57, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:46, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<03:24, 2.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<08:29, 978kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:55, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<04:54, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<07:47, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:01, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:33<04:15, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<23:18, 350kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<16:51, 483kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<11:48, 686kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<13:39, 592kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<10:24, 776kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<07:20, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<08:16, 969kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:35, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<04:39, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:39, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:58, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<04:15, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:48, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:31, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<03:54, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<08:33, 914kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<06:54, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<04:52, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<15:56, 486kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<11:42, 661kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<09:18, 825kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:04, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:03, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:00, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<03:32, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<46:51, 161kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<33:17, 226kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<23:10, 322kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<1:36:45, 77.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<1:08:23, 109kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<47:41, 156kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<36:12, 204kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<26:11, 282kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<18:14, 402kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<23:48, 308kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<17:04, 429kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<11:55, 610kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<1:00:36, 120kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<43:08, 168kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<30:58, 232kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<22:29, 320kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<16:38, 428kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<12:13, 583kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<08:34, 825kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<10:17, 685kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<08:02, 877kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<05:38, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<09:50, 710kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<07:30, 930kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<05:15, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<2:32:59, 45.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<1:47:32, 64.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<1:14:56, 91.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<54:53, 125kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<39:08, 175kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<27:14, 249kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<24:40, 275kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<17:44, 382kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<12:22, 543kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<17:32, 383kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<13:03, 513kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:17<09:06, 729kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<33:19, 199kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<23:50, 278kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<16:36, 396kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<18:03, 364kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<13:21, 492kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<09:23, 696kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<08:37, 754kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<07:07, 912kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<05:02, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:38, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:42, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<03:20, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:37, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:32, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:26<03:12, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<11:43, 537kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<08:51, 710kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:08, 873kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:31, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<03:53, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<09:04, 679kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<07:03, 871kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<04:58, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<06:25, 947kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:11, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<03:40, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:57, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:48, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<06:31, 913kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:09, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:38, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:58, 1.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:11, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<02:58, 1.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<05:50, 997kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:40, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<03:18, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:01, 1.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:06, 1.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:40, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:14, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<02:18, 2.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:53, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:02, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<02:50, 1.95MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<10:08, 546kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:44, 716kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<05:27, 1.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<05:30, 994kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:34, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<03:13, 1.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:36, 962kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:28, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:13, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:33, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:07, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<02:12, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<29:57, 176kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<21:23, 246kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<14:52, 350kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<15:44, 330kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<11:34, 449kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<08:04, 637kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<09:29, 540kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:11, 712kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:42, 886kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:32, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<03:12, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:32, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:40, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:15, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:50, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<02:01, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:32, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<03:03, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:32, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:32, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:19, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:08, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:36, 2.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:15, 3.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:02, 4.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<13:38, 346kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<10:00, 471kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<06:59, 668kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<07:07, 652kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<05:33, 834kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<03:56, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:01, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<05:25, 843kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:03, 1.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:56, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<02:09, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:48, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:03, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:12, 2.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:40, 2.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:11, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:38, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:56, 2.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:28, 2.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:47, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:23, 1.82MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:45, 2.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:17, 3.35MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<06:09, 699kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:45, 902kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:23, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:22, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:58, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:06, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:28, 1.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:50, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:01, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:01, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:27, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:44, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:59, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:45, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:50, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:02, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:08, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<05:45, 675kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<04:23, 886kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<03:03, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<04:59, 767kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:54, 976kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:44, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:40, 804kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:40, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:35, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:06, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:38, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:52, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:32, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:13, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<01:34, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:11, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:36, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<01:49, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<08:45, 397kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<06:21, 546kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<04:26, 772kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:17, 793kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:16, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<02:20, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:25, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:53, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:21, 2.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:53, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:32, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:07, 2.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:37, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:27, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:03, 2.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:30, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:17, 2.42MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:55, 3.31MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:53, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:10, 2.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:44, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:31, 1.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:04, 2.73MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:12, 1.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:47, 1.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<01:15, 2.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<13:12, 216kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<09:24, 303kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<06:51, 406kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:57, 561kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:48, 713kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:50, 956kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:21, 1.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:55, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:20, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<24:49, 104kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<17:29, 147kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<12:05, 210kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<09:19, 269kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<06:41, 374kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<04:40, 530kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:52, 631kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:58, 821kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:05, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:10, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:42, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:12, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:05, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:43, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<01:11, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<04:12, 532kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<03:07, 714kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<02:10, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:21, 920kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:54, 1.13MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:21, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:24, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:09, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:48, 2.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:59, 1.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:33, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<01:04, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<02:46, 707kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:09, 910kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<01:28, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<08:41, 218kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<06:10, 305kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<04:14, 434kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<03:51, 473kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:53, 629kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<02:12, 794kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:41, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:09, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:57, 861kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:28, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:14, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:00, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:42, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:06, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:56, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:39, 2.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:28, 1.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:11, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:00, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:51, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:35, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:39, 805kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:16, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:02, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:48, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<00:21, 2.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<02:37, 405kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:57, 544kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<01:19, 771kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:23, 717kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:04, 922kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:43, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:55, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:43, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:29, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:50, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:40, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:27, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:44, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:33, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:22, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:16, 565kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:57, 754kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:37, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:06, 588kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:49, 781kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:36, 958kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:28, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:18, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:28, 1.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:23, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:18, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:14, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:05, 2.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:03, 3.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:49, 207kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:34, 286kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:16, 407kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:14, 417kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:09, 570kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:03, 808kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:03, 523kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 718kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 896/400000 [00:00<00:44, 8953.99it/s]  0%|          | 1686/400000 [00:00<00:46, 8607.80it/s]  1%|          | 2614/400000 [00:00<00:45, 8797.77it/s]  1%|          | 3544/400000 [00:00<00:44, 8942.62it/s]  1%|          | 4377/400000 [00:00<00:45, 8747.04it/s]  1%|▏         | 5201/400000 [00:00<00:45, 8586.48it/s]  1%|▏         | 5957/400000 [00:00<00:48, 8155.14it/s]  2%|▏         | 6871/400000 [00:00<00:46, 8426.14it/s]  2%|▏         | 7682/400000 [00:00<00:47, 8325.62it/s]  2%|▏         | 8484/400000 [00:01<00:47, 8165.00it/s]  2%|▏         | 9377/400000 [00:01<00:46, 8379.58it/s]  3%|▎         | 10202/400000 [00:01<00:47, 8234.89it/s]  3%|▎         | 11127/400000 [00:01<00:45, 8513.53it/s]  3%|▎         | 12063/400000 [00:01<00:44, 8749.90it/s]  3%|▎         | 13000/400000 [00:01<00:43, 8924.83it/s]  3%|▎         | 13904/400000 [00:01<00:43, 8958.02it/s]  4%|▎         | 14800/400000 [00:01<00:43, 8775.30it/s]  4%|▍         | 15679/400000 [00:01<00:44, 8699.45it/s]  4%|▍         | 16638/400000 [00:01<00:42, 8946.38it/s]  4%|▍         | 17535/400000 [00:02<00:43, 8799.61it/s]  5%|▍         | 18418/400000 [00:02<00:43, 8793.35it/s]  5%|▍         | 19299/400000 [00:02<00:44, 8510.30it/s]  5%|▌         | 20154/400000 [00:02<00:45, 8326.95it/s]  5%|▌         | 20990/400000 [00:02<00:46, 8089.44it/s]  5%|▌         | 21844/400000 [00:02<00:46, 8217.43it/s]  6%|▌         | 22690/400000 [00:02<00:45, 8287.51it/s]  6%|▌         | 23521/400000 [00:02<00:45, 8265.60it/s]  6%|▌         | 24435/400000 [00:02<00:44, 8508.43it/s]  6%|▋         | 25321/400000 [00:02<00:43, 8610.39it/s]  7%|▋         | 26280/400000 [00:03<00:42, 8880.51it/s]  7%|▋         | 27243/400000 [00:03<00:40, 9091.90it/s]  7%|▋         | 28157/400000 [00:03<00:43, 8563.88it/s]  7%|▋         | 29061/400000 [00:03<00:42, 8698.73it/s]  7%|▋         | 29938/400000 [00:03<00:43, 8507.20it/s]  8%|▊         | 30795/400000 [00:03<00:45, 8170.27it/s]  8%|▊         | 31619/400000 [00:03<00:45, 8102.38it/s]  8%|▊         | 32451/400000 [00:03<00:45, 8164.32it/s]  8%|▊         | 33316/400000 [00:03<00:44, 8302.68it/s]  9%|▊         | 34150/400000 [00:04<00:44, 8207.91it/s]  9%|▊         | 34997/400000 [00:04<00:44, 8283.45it/s]  9%|▉         | 35839/400000 [00:04<00:43, 8322.06it/s]  9%|▉         | 36692/400000 [00:04<00:43, 8382.41it/s]  9%|▉         | 37600/400000 [00:04<00:42, 8579.76it/s] 10%|▉         | 38460/400000 [00:04<00:42, 8463.10it/s] 10%|▉         | 39308/400000 [00:04<00:44, 8118.36it/s] 10%|█         | 40124/400000 [00:04<00:45, 7973.98it/s] 10%|█         | 40925/400000 [00:04<00:45, 7869.17it/s] 10%|█         | 41851/400000 [00:04<00:43, 8238.75it/s] 11%|█         | 42757/400000 [00:05<00:42, 8467.08it/s] 11%|█         | 43675/400000 [00:05<00:41, 8665.14it/s] 11%|█         | 44602/400000 [00:05<00:40, 8837.07it/s] 11%|█▏        | 45491/400000 [00:05<00:40, 8724.90it/s] 12%|█▏        | 46368/400000 [00:05<00:40, 8735.85it/s] 12%|█▏        | 47271/400000 [00:05<00:39, 8819.09it/s] 12%|█▏        | 48155/400000 [00:05<00:40, 8621.08it/s] 12%|█▏        | 49020/400000 [00:05<00:41, 8392.69it/s] 12%|█▏        | 49888/400000 [00:05<00:41, 8474.95it/s] 13%|█▎        | 50769/400000 [00:05<00:40, 8571.32it/s] 13%|█▎        | 51701/400000 [00:06<00:39, 8780.68it/s] 13%|█▎        | 52588/400000 [00:06<00:39, 8799.75it/s] 13%|█▎        | 53470/400000 [00:06<00:39, 8773.89it/s] 14%|█▎        | 54349/400000 [00:06<00:40, 8483.43it/s] 14%|█▍        | 55201/400000 [00:06<00:40, 8438.58it/s] 14%|█▍        | 56064/400000 [00:06<00:40, 8494.17it/s] 14%|█▍        | 56916/400000 [00:06<00:40, 8410.30it/s] 14%|█▍        | 57759/400000 [00:06<00:42, 8138.02it/s] 15%|█▍        | 58606/400000 [00:06<00:41, 8234.71it/s] 15%|█▍        | 59468/400000 [00:06<00:40, 8343.56it/s] 15%|█▌        | 60305/400000 [00:07<00:41, 8141.00it/s] 15%|█▌        | 61122/400000 [00:07<00:42, 7976.25it/s] 15%|█▌        | 61931/400000 [00:07<00:42, 8009.86it/s] 16%|█▌        | 62785/400000 [00:07<00:41, 8159.86it/s] 16%|█▌        | 63734/400000 [00:07<00:39, 8517.76it/s] 16%|█▌        | 64691/400000 [00:07<00:38, 8807.26it/s] 16%|█▋        | 65628/400000 [00:07<00:37, 8968.72it/s] 17%|█▋        | 66530/400000 [00:07<00:37, 8914.79it/s] 17%|█▋        | 67426/400000 [00:07<00:38, 8689.35it/s] 17%|█▋        | 68299/400000 [00:08<00:38, 8695.88it/s] 17%|█▋        | 69201/400000 [00:08<00:37, 8790.56it/s] 18%|█▊        | 70102/400000 [00:08<00:37, 8853.93it/s] 18%|█▊        | 70989/400000 [00:08<00:37, 8678.85it/s] 18%|█▊        | 71859/400000 [00:08<00:39, 8394.28it/s] 18%|█▊        | 72711/400000 [00:08<00:38, 8430.91it/s] 18%|█▊        | 73577/400000 [00:08<00:38, 8497.69it/s] 19%|█▊        | 74429/400000 [00:08<00:39, 8342.10it/s] 19%|█▉        | 75267/400000 [00:08<00:38, 8352.59it/s] 19%|█▉        | 76104/400000 [00:08<00:38, 8315.11it/s] 19%|█▉        | 77025/400000 [00:09<00:37, 8564.14it/s] 20%|█▉        | 78000/400000 [00:09<00:36, 8887.56it/s] 20%|█▉        | 78894/400000 [00:09<00:36, 8708.38it/s] 20%|█▉        | 79778/400000 [00:09<00:36, 8745.85it/s] 20%|██        | 80656/400000 [00:09<00:36, 8657.74it/s] 20%|██        | 81525/400000 [00:09<00:36, 8643.60it/s] 21%|██        | 82412/400000 [00:09<00:36, 8706.05it/s] 21%|██        | 83284/400000 [00:09<00:36, 8687.28it/s] 21%|██        | 84154/400000 [00:09<00:36, 8677.92it/s] 21%|██▏       | 85023/400000 [00:09<00:36, 8579.27it/s] 21%|██▏       | 85950/400000 [00:10<00:35, 8774.71it/s] 22%|██▏       | 86875/400000 [00:10<00:35, 8911.09it/s] 22%|██▏       | 87768/400000 [00:10<00:36, 8648.88it/s] 22%|██▏       | 88636/400000 [00:10<00:37, 8288.11it/s] 22%|██▏       | 89561/400000 [00:10<00:36, 8554.42it/s] 23%|██▎       | 90460/400000 [00:10<00:35, 8680.59it/s] 23%|██▎       | 91368/400000 [00:10<00:35, 8795.76it/s] 23%|██▎       | 92252/400000 [00:10<00:35, 8717.26it/s] 23%|██▎       | 93127/400000 [00:10<00:35, 8579.28it/s] 24%|██▎       | 94066/400000 [00:11<00:34, 8805.46it/s] 24%|██▎       | 94950/400000 [00:11<00:34, 8813.03it/s] 24%|██▍       | 95834/400000 [00:11<00:34, 8741.36it/s] 24%|██▍       | 96769/400000 [00:11<00:34, 8915.31it/s] 24%|██▍       | 97663/400000 [00:11<00:34, 8754.73it/s] 25%|██▍       | 98541/400000 [00:11<00:34, 8673.87it/s] 25%|██▍       | 99446/400000 [00:11<00:34, 8781.84it/s] 25%|██▌       | 100371/400000 [00:11<00:33, 8916.82it/s] 25%|██▌       | 101265/400000 [00:11<00:34, 8642.19it/s] 26%|██▌       | 102133/400000 [00:11<00:34, 8514.96it/s] 26%|██▌       | 103048/400000 [00:12<00:34, 8695.24it/s] 26%|██▌       | 103970/400000 [00:12<00:33, 8842.84it/s] 26%|██▌       | 104857/400000 [00:12<00:33, 8689.05it/s] 26%|██▋       | 105782/400000 [00:12<00:33, 8847.98it/s] 27%|██▋       | 106670/400000 [00:12<00:33, 8709.31it/s] 27%|██▋       | 107544/400000 [00:12<00:34, 8572.48it/s] 27%|██▋       | 108404/400000 [00:12<00:34, 8516.02it/s] 27%|██▋       | 109284/400000 [00:12<00:33, 8596.75it/s] 28%|██▊       | 110145/400000 [00:12<00:34, 8523.11it/s] 28%|██▊       | 110999/400000 [00:12<00:34, 8346.88it/s] 28%|██▊       | 111836/400000 [00:13<00:34, 8293.48it/s] 28%|██▊       | 112710/400000 [00:13<00:34, 8422.47it/s] 28%|██▊       | 113609/400000 [00:13<00:33, 8585.05it/s] 29%|██▊       | 114531/400000 [00:13<00:32, 8765.12it/s] 29%|██▉       | 115410/400000 [00:13<00:33, 8569.50it/s] 29%|██▉       | 116283/400000 [00:13<00:32, 8616.00it/s] 29%|██▉       | 117174/400000 [00:13<00:32, 8701.62it/s] 30%|██▉       | 118046/400000 [00:13<00:32, 8606.72it/s] 30%|██▉       | 118908/400000 [00:13<00:33, 8368.35it/s] 30%|██▉       | 119748/400000 [00:13<00:34, 8216.56it/s] 30%|███       | 120572/400000 [00:14<00:34, 8056.81it/s] 30%|███       | 121385/400000 [00:14<00:34, 8077.47it/s] 31%|███       | 122202/400000 [00:14<00:34, 8104.17it/s] 31%|███       | 123014/400000 [00:14<00:34, 7983.42it/s] 31%|███       | 123814/400000 [00:14<00:34, 7957.91it/s] 31%|███       | 124657/400000 [00:14<00:34, 8092.36it/s] 31%|███▏      | 125481/400000 [00:14<00:33, 8135.89it/s] 32%|███▏      | 126310/400000 [00:14<00:33, 8180.89it/s] 32%|███▏      | 127129/400000 [00:14<00:33, 8092.81it/s] 32%|███▏      | 127963/400000 [00:15<00:33, 8164.75it/s] 32%|███▏      | 128874/400000 [00:15<00:32, 8425.38it/s] 32%|███▏      | 129799/400000 [00:15<00:31, 8654.88it/s] 33%|███▎      | 130668/400000 [00:15<00:31, 8603.26it/s] 33%|███▎      | 131531/400000 [00:15<00:32, 8332.58it/s] 33%|███▎      | 132368/400000 [00:15<00:32, 8218.21it/s] 33%|███▎      | 133193/400000 [00:15<00:32, 8216.60it/s] 34%|███▎      | 134125/400000 [00:15<00:31, 8517.71it/s] 34%|███▍      | 135006/400000 [00:15<00:30, 8601.58it/s] 34%|███▍      | 135870/400000 [00:15<00:31, 8446.60it/s] 34%|███▍      | 136802/400000 [00:16<00:30, 8689.64it/s] 34%|███▍      | 137675/400000 [00:16<00:30, 8548.34it/s] 35%|███▍      | 138567/400000 [00:16<00:30, 8651.04it/s] 35%|███▍      | 139435/400000 [00:16<00:30, 8654.15it/s] 35%|███▌      | 140303/400000 [00:16<00:30, 8443.15it/s] 35%|███▌      | 141221/400000 [00:16<00:29, 8649.72it/s] 36%|███▌      | 142169/400000 [00:16<00:29, 8882.79it/s] 36%|███▌      | 143091/400000 [00:16<00:28, 8981.13it/s] 36%|███▌      | 143992/400000 [00:16<00:29, 8673.31it/s] 36%|███▌      | 144864/400000 [00:16<00:30, 8380.10it/s] 36%|███▋      | 145741/400000 [00:17<00:29, 8491.90it/s] 37%|███▋      | 146596/400000 [00:17<00:29, 8507.34it/s] 37%|███▋      | 147450/400000 [00:17<00:29, 8445.33it/s] 37%|███▋      | 148338/400000 [00:17<00:29, 8569.58it/s] 37%|███▋      | 149197/400000 [00:17<00:30, 8339.25it/s] 38%|███▊      | 150063/400000 [00:17<00:29, 8432.21it/s] 38%|███▊      | 151011/400000 [00:17<00:28, 8719.69it/s] 38%|███▊      | 151893/400000 [00:17<00:28, 8749.47it/s] 38%|███▊      | 152771/400000 [00:17<00:29, 8490.50it/s] 38%|███▊      | 153624/400000 [00:18<00:29, 8362.62it/s] 39%|███▊      | 154533/400000 [00:18<00:28, 8567.18it/s] 39%|███▉      | 155468/400000 [00:18<00:27, 8787.73it/s] 39%|███▉      | 156422/400000 [00:18<00:27, 9000.03it/s] 39%|███▉      | 157326/400000 [00:18<00:27, 8907.52it/s] 40%|███▉      | 158220/400000 [00:18<00:27, 8766.91it/s] 40%|███▉      | 159172/400000 [00:18<00:26, 8979.29it/s] 40%|████      | 160106/400000 [00:18<00:26, 9083.17it/s] 40%|████      | 161017/400000 [00:18<00:26, 8866.31it/s] 40%|████      | 161907/400000 [00:18<00:28, 8492.77it/s] 41%|████      | 162762/400000 [00:19<00:28, 8218.97it/s] 41%|████      | 163590/400000 [00:19<00:28, 8233.58it/s] 41%|████      | 164499/400000 [00:19<00:27, 8471.09it/s] 41%|████▏     | 165403/400000 [00:19<00:27, 8629.11it/s] 42%|████▏     | 166270/400000 [00:19<00:27, 8508.83it/s] 42%|████▏     | 167124/400000 [00:19<00:27, 8515.57it/s] 42%|████▏     | 167984/400000 [00:19<00:27, 8540.24it/s] 42%|████▏     | 168911/400000 [00:19<00:26, 8744.26it/s] 42%|████▏     | 169788/400000 [00:19<00:26, 8643.75it/s] 43%|████▎     | 170655/400000 [00:19<00:27, 8464.48it/s] 43%|████▎     | 171504/400000 [00:20<00:26, 8465.44it/s] 43%|████▎     | 172431/400000 [00:20<00:26, 8691.30it/s] 43%|████▎     | 173377/400000 [00:20<00:25, 8907.66it/s] 44%|████▎     | 174335/400000 [00:20<00:24, 9097.09it/s] 44%|████▍     | 175248/400000 [00:20<00:25, 8798.99it/s] 44%|████▍     | 176133/400000 [00:20<00:25, 8636.11it/s] 44%|████▍     | 177031/400000 [00:20<00:25, 8736.30it/s] 44%|████▍     | 177920/400000 [00:20<00:25, 8778.58it/s] 45%|████▍     | 178837/400000 [00:20<00:24, 8891.07it/s] 45%|████▍     | 179728/400000 [00:20<00:25, 8809.61it/s] 45%|████▌     | 180611/400000 [00:21<00:24, 8787.69it/s] 45%|████▌     | 181564/400000 [00:21<00:24, 8997.00it/s] 46%|████▌     | 182466/400000 [00:21<00:24, 8929.14it/s] 46%|████▌     | 183361/400000 [00:21<00:24, 8800.87it/s] 46%|████▌     | 184243/400000 [00:21<00:25, 8438.27it/s] 46%|████▋     | 185170/400000 [00:21<00:24, 8671.04it/s] 47%|████▋     | 186108/400000 [00:21<00:24, 8870.91it/s] 47%|████▋     | 187072/400000 [00:21<00:23, 9087.45it/s] 47%|████▋     | 187986/400000 [00:21<00:23, 9071.42it/s] 47%|████▋     | 188897/400000 [00:22<00:24, 8731.23it/s] 47%|████▋     | 189775/400000 [00:22<00:24, 8460.51it/s] 48%|████▊     | 190627/400000 [00:22<00:25, 8363.57it/s] 48%|████▊     | 191503/400000 [00:22<00:24, 8478.41it/s] 48%|████▊     | 192370/400000 [00:22<00:24, 8533.16it/s] 48%|████▊     | 193226/400000 [00:22<00:24, 8466.82it/s] 49%|████▊     | 194119/400000 [00:22<00:23, 8599.07it/s] 49%|████▊     | 194981/400000 [00:22<00:24, 8535.84it/s] 49%|████▉     | 195906/400000 [00:22<00:23, 8736.93it/s] 49%|████▉     | 196782/400000 [00:22<00:23, 8682.97it/s] 49%|████▉     | 197670/400000 [00:23<00:23, 8739.48it/s] 50%|████▉     | 198581/400000 [00:23<00:22, 8845.90it/s] 50%|████▉     | 199519/400000 [00:23<00:22, 8998.51it/s] 50%|█████     | 200472/400000 [00:23<00:21, 9150.12it/s] 50%|█████     | 201389/400000 [00:23<00:22, 8974.00it/s] 51%|█████     | 202289/400000 [00:23<00:22, 8908.01it/s] 51%|█████     | 203195/400000 [00:23<00:21, 8952.36it/s] 51%|█████     | 204107/400000 [00:23<00:21, 9000.52it/s] 51%|█████▏    | 205008/400000 [00:23<00:22, 8745.28it/s] 51%|█████▏    | 205885/400000 [00:23<00:22, 8460.56it/s] 52%|█████▏    | 206774/400000 [00:24<00:22, 8583.68it/s] 52%|█████▏    | 207636/400000 [00:24<00:23, 8177.32it/s] 52%|█████▏    | 208581/400000 [00:24<00:22, 8519.96it/s] 52%|█████▏    | 209478/400000 [00:24<00:22, 8649.53it/s] 53%|█████▎    | 210350/400000 [00:24<00:22, 8509.43it/s] 53%|█████▎    | 211227/400000 [00:24<00:21, 8585.37it/s] 53%|█████▎    | 212159/400000 [00:24<00:21, 8792.62it/s] 53%|█████▎    | 213110/400000 [00:24<00:20, 8994.36it/s] 54%|█████▎    | 214014/400000 [00:24<00:20, 8873.47it/s] 54%|█████▎    | 214905/400000 [00:25<00:21, 8530.64it/s] 54%|█████▍    | 215763/400000 [00:25<00:21, 8511.20it/s] 54%|█████▍    | 216649/400000 [00:25<00:21, 8611.90it/s] 54%|█████▍    | 217513/400000 [00:25<00:21, 8614.16it/s] 55%|█████▍    | 218377/400000 [00:25<00:21, 8540.82it/s] 55%|█████▍    | 219233/400000 [00:25<00:21, 8351.74it/s] 55%|█████▌    | 220109/400000 [00:25<00:21, 8467.78it/s] 55%|█████▌    | 221063/400000 [00:25<00:20, 8762.59it/s] 56%|█████▌    | 222027/400000 [00:25<00:19, 9005.83it/s] 56%|█████▌    | 222932/400000 [00:25<00:20, 8687.15it/s] 56%|█████▌    | 223807/400000 [00:26<00:20, 8609.17it/s] 56%|█████▌    | 224672/400000 [00:26<00:20, 8542.77it/s] 56%|█████▋    | 225556/400000 [00:26<00:20, 8627.78it/s] 57%|█████▋    | 226422/400000 [00:26<00:20, 8634.76it/s] 57%|█████▋    | 227288/400000 [00:26<00:20, 8464.28it/s] 57%|█████▋    | 228142/400000 [00:26<00:20, 8484.53it/s] 57%|█████▋    | 229021/400000 [00:26<00:19, 8572.66it/s] 57%|█████▋    | 229979/400000 [00:26<00:19, 8851.85it/s] 58%|█████▊    | 230897/400000 [00:26<00:18, 8944.84it/s] 58%|█████▊    | 231794/400000 [00:26<00:19, 8770.35it/s] 58%|█████▊    | 232674/400000 [00:27<00:19, 8497.26it/s] 58%|█████▊    | 233528/400000 [00:27<00:19, 8388.17it/s] 59%|█████▊    | 234402/400000 [00:27<00:19, 8490.26it/s] 59%|█████▉    | 235326/400000 [00:27<00:18, 8701.79it/s] 59%|█████▉    | 236281/400000 [00:27<00:18, 8938.86it/s] 59%|█████▉    | 237179/400000 [00:27<00:18, 8794.15it/s] 60%|█████▉    | 238062/400000 [00:27<00:18, 8769.45it/s] 60%|█████▉    | 238972/400000 [00:27<00:18, 8864.30it/s] 60%|█████▉    | 239895/400000 [00:27<00:17, 8968.77it/s] 60%|██████    | 240809/400000 [00:27<00:17, 9016.93it/s] 60%|██████    | 241712/400000 [00:28<00:18, 8750.07it/s] 61%|██████    | 242595/400000 [00:28<00:17, 8772.32it/s] 61%|██████    | 243549/400000 [00:28<00:17, 8988.17it/s] 61%|██████    | 244458/400000 [00:28<00:17, 9016.77it/s] 61%|██████▏   | 245362/400000 [00:28<00:17, 8968.00it/s] 62%|██████▏   | 246261/400000 [00:28<00:17, 8780.14it/s] 62%|██████▏   | 247141/400000 [00:28<00:17, 8748.19it/s] 62%|██████▏   | 248018/400000 [00:28<00:17, 8600.47it/s] 62%|██████▏   | 248915/400000 [00:28<00:17, 8707.68it/s] 62%|██████▏   | 249788/400000 [00:29<00:17, 8688.08it/s] 63%|██████▎   | 250658/400000 [00:29<00:17, 8597.41it/s] 63%|██████▎   | 251519/400000 [00:29<00:17, 8258.52it/s] 63%|██████▎   | 252425/400000 [00:29<00:17, 8482.64it/s] 63%|██████▎   | 253342/400000 [00:29<00:16, 8676.17it/s] 64%|██████▎   | 254214/400000 [00:29<00:17, 8199.82it/s] 64%|██████▍   | 255043/400000 [00:29<00:17, 8178.89it/s] 64%|██████▍   | 255953/400000 [00:29<00:17, 8432.94it/s] 64%|██████▍   | 256893/400000 [00:29<00:16, 8698.43it/s] 64%|██████▍   | 257770/400000 [00:29<00:16, 8646.40it/s] 65%|██████▍   | 258640/400000 [00:30<00:17, 8282.68it/s] 65%|██████▍   | 259475/400000 [00:30<00:17, 8085.42it/s] 65%|██████▌   | 260390/400000 [00:30<00:16, 8375.65it/s] 65%|██████▌   | 261250/400000 [00:30<00:16, 8439.62it/s] 66%|██████▌   | 262099/400000 [00:30<00:16, 8322.71it/s] 66%|██████▌   | 262939/400000 [00:30<00:16, 8343.98it/s] 66%|██████▌   | 263823/400000 [00:30<00:16, 8485.90it/s] 66%|██████▌   | 264766/400000 [00:30<00:15, 8747.72it/s] 66%|██████▋   | 265679/400000 [00:30<00:15, 8858.77it/s] 67%|██████▋   | 266573/400000 [00:30<00:15, 8880.36it/s] 67%|██████▋   | 267464/400000 [00:31<00:14, 8839.22it/s] 67%|██████▋   | 268365/400000 [00:31<00:14, 8886.51it/s] 67%|██████▋   | 269255/400000 [00:31<00:14, 8842.47it/s] 68%|██████▊   | 270141/400000 [00:31<00:15, 8560.29it/s] 68%|██████▊   | 271041/400000 [00:31<00:14, 8687.59it/s] 68%|██████▊   | 271912/400000 [00:31<00:15, 8538.32it/s] 68%|██████▊   | 272818/400000 [00:31<00:14, 8688.21it/s] 68%|██████▊   | 273767/400000 [00:31<00:14, 8913.52it/s] 69%|██████▊   | 274722/400000 [00:31<00:13, 9092.79it/s] 69%|██████▉   | 275673/400000 [00:32<00:13, 9214.06it/s] 69%|██████▉   | 276597/400000 [00:32<00:13, 8950.93it/s] 69%|██████▉   | 277496/400000 [00:32<00:14, 8744.91it/s] 70%|██████▉   | 278422/400000 [00:32<00:13, 8892.93it/s] 70%|██████▉   | 279373/400000 [00:32<00:13, 9069.03it/s] 70%|███████   | 280283/400000 [00:32<00:13, 8968.87it/s] 70%|███████   | 281183/400000 [00:32<00:13, 8730.24it/s] 71%|███████   | 282060/400000 [00:32<00:13, 8641.49it/s] 71%|███████   | 282972/400000 [00:32<00:13, 8777.26it/s] 71%|███████   | 283852/400000 [00:32<00:13, 8674.01it/s] 71%|███████   | 284739/400000 [00:33<00:13, 8731.23it/s] 71%|███████▏  | 285614/400000 [00:33<00:13, 8658.64it/s] 72%|███████▏  | 286481/400000 [00:33<00:13, 8450.29it/s] 72%|███████▏  | 287420/400000 [00:33<00:12, 8711.19it/s] 72%|███████▏  | 288333/400000 [00:33<00:12, 8831.47it/s] 72%|███████▏  | 289252/400000 [00:33<00:12, 8933.48it/s] 73%|███████▎  | 290148/400000 [00:33<00:12, 8760.31it/s] 73%|███████▎  | 291027/400000 [00:33<00:12, 8469.60it/s] 73%|███████▎  | 291887/400000 [00:33<00:12, 8506.72it/s] 73%|███████▎  | 292747/400000 [00:33<00:12, 8532.86it/s] 73%|███████▎  | 293679/400000 [00:34<00:12, 8752.91it/s] 74%|███████▎  | 294557/400000 [00:34<00:12, 8555.36it/s] 74%|███████▍  | 295416/400000 [00:34<00:12, 8503.28it/s] 74%|███████▍  | 296269/400000 [00:34<00:12, 8429.94it/s] 74%|███████▍  | 297114/400000 [00:34<00:12, 8372.28it/s] 74%|███████▍  | 297953/400000 [00:34<00:12, 8186.83it/s] 75%|███████▍  | 298774/400000 [00:34<00:12, 8103.45it/s] 75%|███████▍  | 299586/400000 [00:34<00:12, 7849.79it/s] 75%|███████▌  | 300442/400000 [00:34<00:12, 8048.71it/s] 75%|███████▌  | 301254/400000 [00:35<00:12, 8069.16it/s] 76%|███████▌  | 302077/400000 [00:35<00:12, 8116.09it/s] 76%|███████▌  | 302891/400000 [00:35<00:12, 7932.77it/s] 76%|███████▌  | 303689/400000 [00:35<00:12, 7946.77it/s] 76%|███████▌  | 304504/400000 [00:35<00:11, 8005.73it/s] 76%|███████▋  | 305306/400000 [00:35<00:12, 7740.46it/s] 77%|███████▋  | 306109/400000 [00:35<00:11, 7824.40it/s] 77%|███████▋  | 306979/400000 [00:35<00:11, 8065.90it/s] 77%|███████▋  | 307864/400000 [00:35<00:11, 8283.89it/s] 77%|███████▋  | 308739/400000 [00:35<00:10, 8416.03it/s] 77%|███████▋  | 309599/400000 [00:36<00:10, 8468.95it/s] 78%|███████▊  | 310468/400000 [00:36<00:10, 8533.61it/s] 78%|███████▊  | 311389/400000 [00:36<00:10, 8725.36it/s] 78%|███████▊  | 312274/400000 [00:36<00:10, 8760.98it/s] 78%|███████▊  | 313163/400000 [00:36<00:09, 8799.13it/s] 79%|███████▊  | 314045/400000 [00:36<00:09, 8765.35it/s] 79%|███████▊  | 314923/400000 [00:36<00:09, 8700.24it/s] 79%|███████▉  | 315815/400000 [00:36<00:09, 8763.23it/s] 79%|███████▉  | 316692/400000 [00:36<00:09, 8516.98it/s] 79%|███████▉  | 317546/400000 [00:36<00:09, 8376.99it/s] 80%|███████▉  | 318386/400000 [00:37<00:10, 8136.90it/s] 80%|███████▉  | 319203/400000 [00:37<00:10, 7961.23it/s] 80%|████████  | 320051/400000 [00:37<00:09, 8107.84it/s] 80%|████████  | 320921/400000 [00:37<00:09, 8274.88it/s] 80%|████████  | 321831/400000 [00:37<00:09, 8504.34it/s] 81%|████████  | 322685/400000 [00:37<00:09, 8482.59it/s] 81%|████████  | 323536/400000 [00:37<00:09, 8396.27it/s] 81%|████████  | 324478/400000 [00:37<00:08, 8678.04it/s] 81%|████████▏ | 325350/400000 [00:37<00:08, 8465.52it/s] 82%|████████▏ | 326222/400000 [00:38<00:08, 8538.35it/s] 82%|████████▏ | 327121/400000 [00:38<00:08, 8664.49it/s] 82%|████████▏ | 327990/400000 [00:38<00:08, 8526.03it/s] 82%|████████▏ | 328913/400000 [00:38<00:08, 8724.10it/s] 82%|████████▏ | 329869/400000 [00:38<00:07, 8958.07it/s] 83%|████████▎ | 330769/400000 [00:38<00:07, 8969.79it/s] 83%|████████▎ | 331669/400000 [00:38<00:07, 8759.52it/s] 83%|████████▎ | 332548/400000 [00:38<00:07, 8561.26it/s] 83%|████████▎ | 333408/400000 [00:38<00:07, 8479.01it/s] 84%|████████▎ | 334260/400000 [00:38<00:07, 8489.69it/s] 84%|████████▍ | 335207/400000 [00:39<00:07, 8761.71it/s] 84%|████████▍ | 336142/400000 [00:39<00:07, 8929.71it/s] 84%|████████▍ | 337039/400000 [00:39<00:07, 8879.92it/s] 84%|████████▍ | 337946/400000 [00:39<00:06, 8931.73it/s] 85%|████████▍ | 338868/400000 [00:39<00:06, 9015.47it/s] 85%|████████▍ | 339771/400000 [00:39<00:06, 8897.33it/s] 85%|████████▌ | 340664/400000 [00:39<00:06, 8906.59it/s] 85%|████████▌ | 341556/400000 [00:39<00:06, 8802.87it/s] 86%|████████▌ | 342494/400000 [00:39<00:06, 8966.79it/s] 86%|████████▌ | 343411/400000 [00:39<00:06, 9024.96it/s] 86%|████████▌ | 344327/400000 [00:40<00:06, 9062.29it/s] 86%|████████▋ | 345235/400000 [00:40<00:06, 8946.88it/s] 87%|████████▋ | 346131/400000 [00:40<00:06, 8820.12it/s] 87%|████████▋ | 347015/400000 [00:40<00:06, 8550.70it/s] 87%|████████▋ | 347873/400000 [00:40<00:06, 8337.74it/s] 87%|████████▋ | 348710/400000 [00:40<00:06, 8238.86it/s] 87%|████████▋ | 349585/400000 [00:40<00:06, 8383.04it/s] 88%|████████▊ | 350426/400000 [00:40<00:06, 8182.76it/s] 88%|████████▊ | 351247/400000 [00:40<00:06, 7941.92it/s] 88%|████████▊ | 352104/400000 [00:40<00:05, 8118.71it/s] 88%|████████▊ | 352928/400000 [00:41<00:05, 8153.29it/s] 88%|████████▊ | 353746/400000 [00:41<00:05, 8147.87it/s] 89%|████████▊ | 354626/400000 [00:41<00:05, 8331.40it/s] 89%|████████▉ | 355559/400000 [00:41<00:05, 8606.00it/s] 89%|████████▉ | 356491/400000 [00:41<00:04, 8805.89it/s] 89%|████████▉ | 357392/400000 [00:41<00:04, 8866.07it/s] 90%|████████▉ | 358282/400000 [00:41<00:04, 8834.61it/s] 90%|████████▉ | 359189/400000 [00:41<00:04, 8903.03it/s] 90%|█████████ | 360081/400000 [00:41<00:04, 8898.70it/s] 90%|█████████ | 360972/400000 [00:42<00:04, 8601.37it/s] 90%|█████████ | 361836/400000 [00:42<00:04, 8390.92it/s] 91%|█████████ | 362679/400000 [00:42<00:04, 8173.48it/s] 91%|█████████ | 363599/400000 [00:42<00:04, 8456.49it/s] 91%|█████████ | 364501/400000 [00:42<00:04, 8616.15it/s] 91%|█████████▏| 365395/400000 [00:42<00:03, 8708.69it/s] 92%|█████████▏| 366270/400000 [00:42<00:04, 8340.93it/s] 92%|█████████▏| 367110/400000 [00:42<00:04, 8161.29it/s] 92%|█████████▏| 368011/400000 [00:42<00:03, 8397.29it/s] 92%|█████████▏| 368856/400000 [00:42<00:03, 8377.61it/s] 92%|█████████▏| 369774/400000 [00:43<00:03, 8602.73it/s] 93%|█████████▎| 370639/400000 [00:43<00:03, 8562.92it/s] 93%|█████████▎| 371499/400000 [00:43<00:03, 8502.36it/s] 93%|█████████▎| 372452/400000 [00:43<00:03, 8784.33it/s] 93%|█████████▎| 373335/400000 [00:43<00:03, 8791.52it/s] 94%|█████████▎| 374217/400000 [00:43<00:02, 8796.75it/s] 94%|█████████▍| 375135/400000 [00:43<00:02, 8906.03it/s] 94%|█████████▍| 376028/400000 [00:43<00:02, 8608.91it/s] 94%|█████████▍| 376893/400000 [00:43<00:02, 8486.06it/s] 94%|█████████▍| 377754/400000 [00:43<00:02, 8522.82it/s] 95%|█████████▍| 378710/400000 [00:44<00:02, 8807.50it/s] 95%|█████████▍| 379646/400000 [00:44<00:02, 8964.63it/s] 95%|█████████▌| 380546/400000 [00:44<00:02, 8940.90it/s] 95%|█████████▌| 381448/400000 [00:44<00:02, 8961.07it/s] 96%|█████████▌| 382347/400000 [00:44<00:01, 8969.32it/s] 96%|█████████▌| 383293/400000 [00:44<00:01, 9109.01it/s] 96%|█████████▌| 384241/400000 [00:44<00:01, 9216.15it/s] 96%|█████████▋| 385164/400000 [00:44<00:01, 8926.55it/s] 97%|█████████▋| 386060/400000 [00:44<00:01, 8825.34it/s] 97%|█████████▋| 386945/400000 [00:44<00:01, 8827.44it/s] 97%|█████████▋| 387856/400000 [00:45<00:01, 8910.13it/s] 97%|█████████▋| 388749/400000 [00:45<00:01, 8824.48it/s] 97%|█████████▋| 389633/400000 [00:45<00:01, 8625.79it/s] 98%|█████████▊| 390583/400000 [00:45<00:01, 8868.80it/s] 98%|█████████▊| 391473/400000 [00:45<00:00, 8540.42it/s] 98%|█████████▊| 392348/400000 [00:45<00:00, 8601.40it/s] 98%|█████████▊| 393212/400000 [00:45<00:00, 8572.58it/s] 99%|█████████▊| 394096/400000 [00:45<00:00, 8651.05it/s] 99%|█████████▊| 394963/400000 [00:45<00:00, 8331.30it/s] 99%|█████████▉| 395800/400000 [00:46<00:00, 8204.40it/s] 99%|█████████▉| 396692/400000 [00:46<00:00, 8405.67it/s] 99%|█████████▉| 397616/400000 [00:46<00:00, 8637.64it/s]100%|█████████▉| 398494/400000 [00:46<00:00, 8677.85it/s]100%|█████████▉| 399365/400000 [00:46<00:00, 8671.95it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8600.35it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fcd6d6e0c88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011108879026595785 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011035641100893053 	 Accuracy: 61

  model saves at 61% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15868 out of table with 15838 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


### Running {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} 'model_pars' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text/ 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 

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
RuntimeError: index out of range: Tried to access index 15868 out of table with 15838 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

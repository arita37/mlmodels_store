
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f68f0779fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 08:13:07.918833
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 08:13:07.922891
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 08:13:07.927594
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 08:13:07.931794
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f68fc791400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355158.4375
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 285438.6562
Epoch 3/10

1/1 [==============================] - 0s 107ms/step - loss: 186453.7656
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 110148.1484
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 61835.5117
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 34757.3750
Epoch 7/10

1/1 [==============================] - 0s 114ms/step - loss: 19807.0234
Epoch 8/10

1/1 [==============================] - 0s 99ms/step - loss: 11949.2920
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 7895.4561
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 5617.6240

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.3578412   9.8010025   8.096823    7.2290373   9.524773    7.271768
   8.976969    9.465994    9.354199    8.872426   11.585571    9.366979
   7.5871835  10.150408    7.6101074   9.761005   10.2098      8.67892
   8.828112    9.075297    8.888285   10.189435    6.8328013   8.312818
   8.237783    8.546245    8.999425    8.723787    9.2178955   9.486509
   8.846366    7.185072    8.659405    9.890939    8.344561    8.515438
   8.599378   10.129311    9.379388    8.954229   10.173475    9.704949
  11.4726095   9.600686   10.037685    9.591718    9.436337    8.658448
   8.490586   10.316579    8.744871    8.318447    9.768379   11.297083
   9.016139    9.934886   10.415207    7.4405813   8.562617    9.830549
  -2.511943   -1.9740833  -1.2431766   0.35774836  1.3331532   0.3190751
  -0.25979534  0.07977638 -0.7421979  -0.46107504 -0.69764686  0.48764247
   0.27899283 -0.9878339   0.5888174   0.6592462   0.25929406 -1.4178016
  -0.9571916  -0.21634279 -0.03894138 -1.884909   -0.14119169 -0.34892973
  -0.2716974   2.3349953   0.08591112 -0.9417851   0.0141075   0.9524723
   0.9430388   1.4903597  -0.13670945  0.14669344 -0.7729776  -0.09164512
  -0.08675969  1.8835022  -0.2202293   0.18043509  0.6940243  -1.016241
   0.33675042 -0.65953577  0.32349262 -0.16792783 -1.2367225   0.4251321
   1.5520504  -1.4426863   0.9010506  -0.7321111  -1.1240212   0.78876823
   0.54689115 -1.0723596  -0.87975514 -0.78454435 -0.9458524   0.9072759
   0.40641516  1.6639448   1.3523889  -0.80040663  0.3024767   0.8806981
   1.4706095  -0.31450188 -0.71597934  0.16629258 -0.07856318 -0.5781239
  -0.840462    0.31963682  1.0271306   0.4663592   1.1667784   0.9596692
   0.23377782 -0.37709838  1.2621276  -0.18835297  0.21658033 -0.132617
   0.23528878 -1.1211269   0.07558845  1.0937665   0.4786356   0.4666924
  -0.4085086  -0.12549543 -2.7020998   0.60774714 -1.259244    0.51828015
   1.6013821   1.6381154   1.1004984   1.0226512  -0.867033   -1.9484143
  -2.677208   -0.6702566   1.7996926   0.47149146  0.8442404   1.091091
  -0.19853802 -0.42091542 -0.6606335  -0.0633665   0.28945214  1.6022993
   1.9161141   0.26913908  0.24915735  2.024284   -0.72301924 -1.1095301
   0.18424511  8.30584     9.807989   10.945752    9.079406    9.450262
   9.711135   10.406459    9.9480505   9.802516    8.97544     9.749349
  10.01556    10.814694    9.714277   10.419069    9.474203    9.87942
   9.511116    8.76735    10.674574    7.8906236   9.234447    9.170916
   9.795147    8.811393    8.993409    9.538235   10.60412     9.425315
   9.910474    9.781302   10.103666    8.746328   11.019707    9.116722
   8.437351    9.956281   11.171682    9.126477    9.012533   10.340333
   8.824614    9.272645    9.2792845   9.833218    8.630344    8.055214
   8.997592    9.938377    9.499625    9.005943    9.741136    8.214459
  10.579008   10.129174   10.862947    7.5375986   8.376798    8.8860655
   0.5313044   0.5217605   0.09224695  1.4097037   0.4144907   0.9107123
   2.5553293   0.78084797  0.18283284  0.97756577  0.4202801   0.7216549
   1.7852888   0.3931166   2.0843058   0.3119865   0.6944151   0.7757611
   1.3963789   1.0095215   2.1674619   2.278522    0.8628718   0.6024234
   0.8904603   0.06299973  1.7423599   2.3350737   0.7215265   1.7774501
   0.18995416  0.7967799   0.1498794   2.6064298   2.0102909   0.5416233
   0.32423055  1.3330756   2.1671996   0.460899    0.4189899   2.7618532
   1.047734    1.6608338   0.56798583  1.4097986   2.047721    2.0838227
   1.2727442   0.4329427   1.943338    0.19991362  0.7017523   0.5016237
   1.2124184   0.37813234  0.2966088   2.3454266   0.3625766   2.2945514
   2.3204217   1.5186294   2.237409    1.2097679   3.027925    1.3414588
   0.65797436  0.59347177  3.2622967   1.4204623   0.4680617   0.7540119
   3.037065    0.3250072   1.6956856   0.36454713  0.55337673  0.37514663
   1.4052292   1.2081134   1.5862929   1.0577657   0.47194362  0.06486326
   2.6247327   0.25053942  2.4575295   1.2863591   0.25822878  0.53964895
   2.1067147   0.7478497   0.19316101  0.99482673  0.9056399   2.3345428
   0.3184144   1.9342434   2.2262218   3.6686826   0.26165074  2.6904492
   1.6270082   0.23698807  1.4712446   0.37103623  0.65231836  2.4641929
   0.6113279   0.310826    1.380558    0.4874661   0.4329741   2.1234503
   1.9599359   1.4917763   1.3534656   0.6636282   1.7218597   0.11742246
   9.602644   -5.495061   -4.9522843 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 08:13:18.680399
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.8442
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 08:13:18.685127
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8643.94
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 08:13:18.689186
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.3742
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 08:13:18.693000
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -773.126
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140088348179256
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140087121051152
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140087120617544
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140087120618048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140087120618552
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140087120619056

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f68ea1c43c8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.579078
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.551417
grad_step = 000002, loss = 0.528192
grad_step = 000003, loss = 0.504533
grad_step = 000004, loss = 0.481143
grad_step = 000005, loss = 0.461192
grad_step = 000006, loss = 0.445714
grad_step = 000007, loss = 0.424277
grad_step = 000008, loss = 0.405178
grad_step = 000009, loss = 0.400468
grad_step = 000010, loss = 0.395336
grad_step = 000011, loss = 0.383048
grad_step = 000012, loss = 0.371064
grad_step = 000013, loss = 0.362028
grad_step = 000014, loss = 0.354403
grad_step = 000015, loss = 0.346314
grad_step = 000016, loss = 0.337080
grad_step = 000017, loss = 0.327216
grad_step = 000018, loss = 0.317702
grad_step = 000019, loss = 0.309122
grad_step = 000020, loss = 0.301267
grad_step = 000021, loss = 0.293759
grad_step = 000022, loss = 0.285778
grad_step = 000023, loss = 0.277112
grad_step = 000024, loss = 0.268539
grad_step = 000025, loss = 0.260816
grad_step = 000026, loss = 0.253909
grad_step = 000027, loss = 0.247175
grad_step = 000028, loss = 0.240018
grad_step = 000029, loss = 0.232447
grad_step = 000030, loss = 0.224965
grad_step = 000031, loss = 0.218048
grad_step = 000032, loss = 0.211710
grad_step = 000033, loss = 0.204169
grad_step = 000034, loss = 0.195788
grad_step = 000035, loss = 0.186904
grad_step = 000036, loss = 0.178067
grad_step = 000037, loss = 0.169769
grad_step = 000038, loss = 0.162928
grad_step = 000039, loss = 0.158034
grad_step = 000040, loss = 0.152057
grad_step = 000041, loss = 0.145619
grad_step = 000042, loss = 0.139970
grad_step = 000043, loss = 0.134399
grad_step = 000044, loss = 0.128768
grad_step = 000045, loss = 0.123383
grad_step = 000046, loss = 0.118112
grad_step = 000047, loss = 0.112986
grad_step = 000048, loss = 0.108297
grad_step = 000049, loss = 0.103573
grad_step = 000050, loss = 0.098625
grad_step = 000051, loss = 0.093981
grad_step = 000052, loss = 0.089792
grad_step = 000053, loss = 0.085849
grad_step = 000054, loss = 0.081940
grad_step = 000055, loss = 0.077995
grad_step = 000056, loss = 0.074116
grad_step = 000057, loss = 0.070426
grad_step = 000058, loss = 0.066911
grad_step = 000059, loss = 0.063569
grad_step = 000060, loss = 0.060408
grad_step = 000061, loss = 0.057319
grad_step = 000062, loss = 0.054236
grad_step = 000063, loss = 0.051237
grad_step = 000064, loss = 0.048411
grad_step = 000065, loss = 0.045744
grad_step = 000066, loss = 0.043143
grad_step = 000067, loss = 0.040598
grad_step = 000068, loss = 0.038155
grad_step = 000069, loss = 0.035822
grad_step = 000070, loss = 0.033612
grad_step = 000071, loss = 0.031510
grad_step = 000072, loss = 0.029478
grad_step = 000073, loss = 0.027507
grad_step = 000074, loss = 0.025629
grad_step = 000075, loss = 0.023874
grad_step = 000076, loss = 0.022237
grad_step = 000077, loss = 0.020681
grad_step = 000078, loss = 0.019180
grad_step = 000079, loss = 0.017754
grad_step = 000080, loss = 0.016436
grad_step = 000081, loss = 0.015205
grad_step = 000082, loss = 0.014057
grad_step = 000083, loss = 0.012999
grad_step = 000084, loss = 0.011998
grad_step = 000085, loss = 0.011051
grad_step = 000086, loss = 0.010201
grad_step = 000087, loss = 0.009415
grad_step = 000088, loss = 0.008686
grad_step = 000089, loss = 0.008027
grad_step = 000090, loss = 0.007420
grad_step = 000091, loss = 0.006859
grad_step = 000092, loss = 0.006355
grad_step = 000093, loss = 0.005898
grad_step = 000094, loss = 0.005485
grad_step = 000095, loss = 0.005116
grad_step = 000096, loss = 0.004781
grad_step = 000097, loss = 0.004478
grad_step = 000098, loss = 0.004210
grad_step = 000099, loss = 0.003972
grad_step = 000100, loss = 0.003759
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003570
grad_step = 000102, loss = 0.003400
grad_step = 000103, loss = 0.003253
grad_step = 000104, loss = 0.003123
grad_step = 000105, loss = 0.003005
grad_step = 000106, loss = 0.002902
grad_step = 000107, loss = 0.002810
grad_step = 000108, loss = 0.002730
grad_step = 000109, loss = 0.002659
grad_step = 000110, loss = 0.002595
grad_step = 000111, loss = 0.002537
grad_step = 000112, loss = 0.002485
grad_step = 000113, loss = 0.002441
grad_step = 000114, loss = 0.002400
grad_step = 000115, loss = 0.002362
grad_step = 000116, loss = 0.002329
grad_step = 000117, loss = 0.002299
grad_step = 000118, loss = 0.002271
grad_step = 000119, loss = 0.002247
grad_step = 000120, loss = 0.002225
grad_step = 000121, loss = 0.002204
grad_step = 000122, loss = 0.002186
grad_step = 000123, loss = 0.002169
grad_step = 000124, loss = 0.002154
grad_step = 000125, loss = 0.002141
grad_step = 000126, loss = 0.002128
grad_step = 000127, loss = 0.002117
grad_step = 000128, loss = 0.002106
grad_step = 000129, loss = 0.002097
grad_step = 000130, loss = 0.002090
grad_step = 000131, loss = 0.002081
grad_step = 000132, loss = 0.002073
grad_step = 000133, loss = 0.002068
grad_step = 000134, loss = 0.002061
grad_step = 000135, loss = 0.002055
grad_step = 000136, loss = 0.002051
grad_step = 000137, loss = 0.002046
grad_step = 000138, loss = 0.002040
grad_step = 000139, loss = 0.002036
grad_step = 000140, loss = 0.002032
grad_step = 000141, loss = 0.002028
grad_step = 000142, loss = 0.002023
grad_step = 000143, loss = 0.002020
grad_step = 000144, loss = 0.002017
grad_step = 000145, loss = 0.002014
grad_step = 000146, loss = 0.002010
grad_step = 000147, loss = 0.002006
grad_step = 000148, loss = 0.002002
grad_step = 000149, loss = 0.001999
grad_step = 000150, loss = 0.001996
grad_step = 000151, loss = 0.001993
grad_step = 000152, loss = 0.001993
grad_step = 000153, loss = 0.001997
grad_step = 000154, loss = 0.002005
grad_step = 000155, loss = 0.001997
grad_step = 000156, loss = 0.001982
grad_step = 000157, loss = 0.001976
grad_step = 000158, loss = 0.001984
grad_step = 000159, loss = 0.001985
grad_step = 000160, loss = 0.001973
grad_step = 000161, loss = 0.001965
grad_step = 000162, loss = 0.001969
grad_step = 000163, loss = 0.001971
grad_step = 000164, loss = 0.001966
grad_step = 000165, loss = 0.001957
grad_step = 000166, loss = 0.001955
grad_step = 000167, loss = 0.001958
grad_step = 000168, loss = 0.001958
grad_step = 000169, loss = 0.001952
grad_step = 000170, loss = 0.001945
grad_step = 000171, loss = 0.001944
grad_step = 000172, loss = 0.001945
grad_step = 000173, loss = 0.001944
grad_step = 000174, loss = 0.001941
grad_step = 000175, loss = 0.001936
grad_step = 000176, loss = 0.001932
grad_step = 000177, loss = 0.001930
grad_step = 000178, loss = 0.001930
grad_step = 000179, loss = 0.001930
grad_step = 000180, loss = 0.001930
grad_step = 000181, loss = 0.001929
grad_step = 000182, loss = 0.001926
grad_step = 000183, loss = 0.001923
grad_step = 000184, loss = 0.001919
grad_step = 000185, loss = 0.001915
grad_step = 000186, loss = 0.001911
grad_step = 000187, loss = 0.001909
grad_step = 000188, loss = 0.001907
grad_step = 000189, loss = 0.001905
grad_step = 000190, loss = 0.001905
grad_step = 000191, loss = 0.001905
grad_step = 000192, loss = 0.001907
grad_step = 000193, loss = 0.001913
grad_step = 000194, loss = 0.001924
grad_step = 000195, loss = 0.001929
grad_step = 000196, loss = 0.001927
grad_step = 000197, loss = 0.001904
grad_step = 000198, loss = 0.001886
grad_step = 000199, loss = 0.001886
grad_step = 000200, loss = 0.001896
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001900
grad_step = 000202, loss = 0.001889
grad_step = 000203, loss = 0.001876
grad_step = 000204, loss = 0.001872
grad_step = 000205, loss = 0.001877
grad_step = 000206, loss = 0.001881
grad_step = 000207, loss = 0.001876
grad_step = 000208, loss = 0.001867
grad_step = 000209, loss = 0.001860
grad_step = 000210, loss = 0.001860
grad_step = 000211, loss = 0.001862
grad_step = 000212, loss = 0.001862
grad_step = 000213, loss = 0.001857
grad_step = 000214, loss = 0.001851
grad_step = 000215, loss = 0.001846
grad_step = 000216, loss = 0.001845
grad_step = 000217, loss = 0.001845
grad_step = 000218, loss = 0.001845
grad_step = 000219, loss = 0.001843
grad_step = 000220, loss = 0.001840
grad_step = 000221, loss = 0.001836
grad_step = 000222, loss = 0.001832
grad_step = 000223, loss = 0.001828
grad_step = 000224, loss = 0.001824
grad_step = 000225, loss = 0.001822
grad_step = 000226, loss = 0.001819
grad_step = 000227, loss = 0.001817
grad_step = 000228, loss = 0.001816
grad_step = 000229, loss = 0.001815
grad_step = 000230, loss = 0.001817
grad_step = 000231, loss = 0.001824
grad_step = 000232, loss = 0.001840
grad_step = 000233, loss = 0.001879
grad_step = 000234, loss = 0.001895
grad_step = 000235, loss = 0.001917
grad_step = 000236, loss = 0.001856
grad_step = 000237, loss = 0.001806
grad_step = 000238, loss = 0.001795
grad_step = 000239, loss = 0.001821
grad_step = 000240, loss = 0.001839
grad_step = 000241, loss = 0.001811
grad_step = 000242, loss = 0.001783
grad_step = 000243, loss = 0.001782
grad_step = 000244, loss = 0.001796
grad_step = 000245, loss = 0.001806
grad_step = 000246, loss = 0.001801
grad_step = 000247, loss = 0.001788
grad_step = 000248, loss = 0.001767
grad_step = 000249, loss = 0.001762
grad_step = 000250, loss = 0.001770
grad_step = 000251, loss = 0.001774
grad_step = 000252, loss = 0.001772
grad_step = 000253, loss = 0.001765
grad_step = 000254, loss = 0.001754
grad_step = 000255, loss = 0.001745
grad_step = 000256, loss = 0.001743
grad_step = 000257, loss = 0.001746
grad_step = 000258, loss = 0.001748
grad_step = 000259, loss = 0.001747
grad_step = 000260, loss = 0.001747
grad_step = 000261, loss = 0.001749
grad_step = 000262, loss = 0.001744
grad_step = 000263, loss = 0.001741
grad_step = 000264, loss = 0.001738
grad_step = 000265, loss = 0.001737
grad_step = 000266, loss = 0.001735
grad_step = 000267, loss = 0.001734
grad_step = 000268, loss = 0.001734
grad_step = 000269, loss = 0.001739
grad_step = 000270, loss = 0.001740
grad_step = 000271, loss = 0.001745
grad_step = 000272, loss = 0.001742
grad_step = 000273, loss = 0.001743
grad_step = 000274, loss = 0.001729
grad_step = 000275, loss = 0.001715
grad_step = 000276, loss = 0.001697
grad_step = 000277, loss = 0.001685
grad_step = 000278, loss = 0.001682
grad_step = 000279, loss = 0.001685
grad_step = 000280, loss = 0.001692
grad_step = 000281, loss = 0.001701
grad_step = 000282, loss = 0.001717
grad_step = 000283, loss = 0.001729
grad_step = 000284, loss = 0.001752
grad_step = 000285, loss = 0.001746
grad_step = 000286, loss = 0.001743
grad_step = 000287, loss = 0.001711
grad_step = 000288, loss = 0.001681
grad_step = 000289, loss = 0.001658
grad_step = 000290, loss = 0.001655
grad_step = 000291, loss = 0.001668
grad_step = 000292, loss = 0.001683
grad_step = 000293, loss = 0.001692
grad_step = 000294, loss = 0.001682
grad_step = 000295, loss = 0.001667
grad_step = 000296, loss = 0.001651
grad_step = 000297, loss = 0.001641
grad_step = 000298, loss = 0.001636
grad_step = 000299, loss = 0.001633
grad_step = 000300, loss = 0.001632
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001634
grad_step = 000302, loss = 0.001644
grad_step = 000303, loss = 0.001659
grad_step = 000304, loss = 0.001697
grad_step = 000305, loss = 0.001722
grad_step = 000306, loss = 0.001789
grad_step = 000307, loss = 0.001769
grad_step = 000308, loss = 0.001769
grad_step = 000309, loss = 0.001702
grad_step = 000310, loss = 0.001636
grad_step = 000311, loss = 0.001612
grad_step = 000312, loss = 0.001644
grad_step = 000313, loss = 0.001689
grad_step = 000314, loss = 0.001671
grad_step = 000315, loss = 0.001622
grad_step = 000316, loss = 0.001593
grad_step = 000317, loss = 0.001609
grad_step = 000318, loss = 0.001631
grad_step = 000319, loss = 0.001624
grad_step = 000320, loss = 0.001613
grad_step = 000321, loss = 0.001611
grad_step = 000322, loss = 0.001616
grad_step = 000323, loss = 0.001597
grad_step = 000324, loss = 0.001576
grad_step = 000325, loss = 0.001571
grad_step = 000326, loss = 0.001581
grad_step = 000327, loss = 0.001590
grad_step = 000328, loss = 0.001588
grad_step = 000329, loss = 0.001593
grad_step = 000330, loss = 0.001605
grad_step = 000331, loss = 0.001635
grad_step = 000332, loss = 0.001651
grad_step = 000333, loss = 0.001681
grad_step = 000334, loss = 0.001660
grad_step = 000335, loss = 0.001647
grad_step = 000336, loss = 0.001581
grad_step = 000337, loss = 0.001546
grad_step = 000338, loss = 0.001552
grad_step = 000339, loss = 0.001577
grad_step = 000340, loss = 0.001596
grad_step = 000341, loss = 0.001579
grad_step = 000342, loss = 0.001566
grad_step = 000343, loss = 0.001553
grad_step = 000344, loss = 0.001540
grad_step = 000345, loss = 0.001531
grad_step = 000346, loss = 0.001536
grad_step = 000347, loss = 0.001549
grad_step = 000348, loss = 0.001554
grad_step = 000349, loss = 0.001547
grad_step = 000350, loss = 0.001534
grad_step = 000351, loss = 0.001530
grad_step = 000352, loss = 0.001530
grad_step = 000353, loss = 0.001529
grad_step = 000354, loss = 0.001519
grad_step = 000355, loss = 0.001512
grad_step = 000356, loss = 0.001511
grad_step = 000357, loss = 0.001517
grad_step = 000358, loss = 0.001527
grad_step = 000359, loss = 0.001547
grad_step = 000360, loss = 0.001574
grad_step = 000361, loss = 0.001648
grad_step = 000362, loss = 0.001591
grad_step = 000363, loss = 0.001548
grad_step = 000364, loss = 0.001503
grad_step = 000365, loss = 0.001533
grad_step = 000366, loss = 0.001567
grad_step = 000367, loss = 0.001506
grad_step = 000368, loss = 0.001505
grad_step = 000369, loss = 0.001537
grad_step = 000370, loss = 0.001503
grad_step = 000371, loss = 0.001486
grad_step = 000372, loss = 0.001508
grad_step = 000373, loss = 0.001499
grad_step = 000374, loss = 0.001477
grad_step = 000375, loss = 0.001476
grad_step = 000376, loss = 0.001487
grad_step = 000377, loss = 0.001482
grad_step = 000378, loss = 0.001465
grad_step = 000379, loss = 0.001464
grad_step = 000380, loss = 0.001473
grad_step = 000381, loss = 0.001470
grad_step = 000382, loss = 0.001459
grad_step = 000383, loss = 0.001452
grad_step = 000384, loss = 0.001453
grad_step = 000385, loss = 0.001463
grad_step = 000386, loss = 0.001472
grad_step = 000387, loss = 0.001490
grad_step = 000388, loss = 0.001499
grad_step = 000389, loss = 0.001539
grad_step = 000390, loss = 0.001590
grad_step = 000391, loss = 0.001627
grad_step = 000392, loss = 0.001606
grad_step = 000393, loss = 0.001532
grad_step = 000394, loss = 0.001441
grad_step = 000395, loss = 0.001425
grad_step = 000396, loss = 0.001471
grad_step = 000397, loss = 0.001502
grad_step = 000398, loss = 0.001485
grad_step = 000399, loss = 0.001433
grad_step = 000400, loss = 0.001406
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001424
grad_step = 000402, loss = 0.001451
grad_step = 000403, loss = 0.001444
grad_step = 000404, loss = 0.001412
grad_step = 000405, loss = 0.001394
grad_step = 000406, loss = 0.001398
grad_step = 000407, loss = 0.001409
grad_step = 000408, loss = 0.001412
grad_step = 000409, loss = 0.001398
grad_step = 000410, loss = 0.001388
grad_step = 000411, loss = 0.001380
grad_step = 000412, loss = 0.001377
grad_step = 000413, loss = 0.001378
grad_step = 000414, loss = 0.001380
grad_step = 000415, loss = 0.001383
grad_step = 000416, loss = 0.001377
grad_step = 000417, loss = 0.001369
grad_step = 000418, loss = 0.001358
grad_step = 000419, loss = 0.001351
grad_step = 000420, loss = 0.001349
grad_step = 000421, loss = 0.001351
grad_step = 000422, loss = 0.001353
grad_step = 000423, loss = 0.001352
grad_step = 000424, loss = 0.001351
grad_step = 000425, loss = 0.001348
grad_step = 000426, loss = 0.001346
grad_step = 000427, loss = 0.001341
grad_step = 000428, loss = 0.001338
grad_step = 000429, loss = 0.001332
grad_step = 000430, loss = 0.001328
grad_step = 000431, loss = 0.001324
grad_step = 000432, loss = 0.001322
grad_step = 000433, loss = 0.001320
grad_step = 000434, loss = 0.001320
grad_step = 000435, loss = 0.001321
grad_step = 000436, loss = 0.001327
grad_step = 000437, loss = 0.001336
grad_step = 000438, loss = 0.001367
grad_step = 000439, loss = 0.001389
grad_step = 000440, loss = 0.001454
grad_step = 000441, loss = 0.001416
grad_step = 000442, loss = 0.001386
grad_step = 000443, loss = 0.001316
grad_step = 000444, loss = 0.001290
grad_step = 000445, loss = 0.001314
grad_step = 000446, loss = 0.001337
grad_step = 000447, loss = 0.001335
grad_step = 000448, loss = 0.001296
grad_step = 000449, loss = 0.001277
grad_step = 000450, loss = 0.001288
grad_step = 000451, loss = 0.001305
grad_step = 000452, loss = 0.001313
grad_step = 000453, loss = 0.001291
grad_step = 000454, loss = 0.001270
grad_step = 000455, loss = 0.001256
grad_step = 000456, loss = 0.001256
grad_step = 000457, loss = 0.001266
grad_step = 000458, loss = 0.001276
grad_step = 000459, loss = 0.001288
grad_step = 000460, loss = 0.001283
grad_step = 000461, loss = 0.001280
grad_step = 000462, loss = 0.001264
grad_step = 000463, loss = 0.001252
grad_step = 000464, loss = 0.001239
grad_step = 000465, loss = 0.001231
grad_step = 000466, loss = 0.001226
grad_step = 000467, loss = 0.001224
grad_step = 000468, loss = 0.001223
grad_step = 000469, loss = 0.001224
grad_step = 000470, loss = 0.001227
grad_step = 000471, loss = 0.001230
grad_step = 000472, loss = 0.001237
grad_step = 000473, loss = 0.001242
grad_step = 000474, loss = 0.001257
grad_step = 000475, loss = 0.001257
grad_step = 000476, loss = 0.001272
grad_step = 000477, loss = 0.001259
grad_step = 000478, loss = 0.001255
grad_step = 000479, loss = 0.001234
grad_step = 000480, loss = 0.001223
grad_step = 000481, loss = 0.001224
grad_step = 000482, loss = 0.001239
grad_step = 000483, loss = 0.001268
grad_step = 000484, loss = 0.001288
grad_step = 000485, loss = 0.001308
grad_step = 000486, loss = 0.001293
grad_step = 000487, loss = 0.001268
grad_step = 000488, loss = 0.001221
grad_step = 000489, loss = 0.001182
grad_step = 000490, loss = 0.001163
grad_step = 000491, loss = 0.001166
grad_step = 000492, loss = 0.001183
grad_step = 000493, loss = 0.001199
grad_step = 000494, loss = 0.001213
grad_step = 000495, loss = 0.001209
grad_step = 000496, loss = 0.001204
grad_step = 000497, loss = 0.001195
grad_step = 000498, loss = 0.001218
grad_step = 000499, loss = 0.001248
grad_step = 000500, loss = 0.001334
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001309
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

  date_run                              2020-05-11 08:13:43.256724
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.286203
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 08:13:43.263269
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.19699
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 08:13:43.269604
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.158879
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 08:13:43.275503
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.99333
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
0   2020-05-11 08:13:07.918833  ...    mean_absolute_error
1   2020-05-11 08:13:07.922891  ...     mean_squared_error
2   2020-05-11 08:13:07.927594  ...  median_absolute_error
3   2020-05-11 08:13:07.931794  ...               r2_score
4   2020-05-11 08:13:18.680399  ...    mean_absolute_error
5   2020-05-11 08:13:18.685127  ...     mean_squared_error
6   2020-05-11 08:13:18.689186  ...  median_absolute_error
7   2020-05-11 08:13:18.693000  ...               r2_score
8   2020-05-11 08:13:43.256724  ...    mean_absolute_error
9   2020-05-11 08:13:43.263269  ...     mean_squared_error
10  2020-05-11 08:13:43.269604  ...  median_absolute_error
11  2020-05-11 08:13:43.275503  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 20%|██        | 2023424/9912422 [00:00<00:00, 20004090.40it/s]9920512it [00:00, 37084879.88it/s]                             
0it [00:00, ?it/s]32768it [00:00, 587100.08it/s]
0it [00:00, ?it/s]  3%|▎         | 57344/1648877 [00:00<00:02, 568302.61it/s]1654784it [00:00, 10303794.89it/s]                         
0it [00:00, ?it/s]8192it [00:00, 150768.72it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3fd0901fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f6e01def0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3fd088def0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f6daf50f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3fd0901fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f83285e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3fd0901fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f77738748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3fd08c9c18> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f83285e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f6e01dfd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8ad1fe01d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5430ca5b261b6ba18a60a4abf3006562351e6d072e445359db93ec4f4c25aa84
  Stored in directory: /tmp/pip-ephem-wheel-cache-iyzfbpac/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8a69bc81d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3325952/17464789 [====>.........................] - ETA: 0s
11419648/17464789 [==================>...........] - ETA: 0s
16441344/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 08:15:11.529176: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 08:15:11.533318: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 08:15:11.533440: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5627d70d1770 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 08:15:11.533455: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6896 - accuracy: 0.4985
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7126 - accuracy: 0.4970 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5874 - accuracy: 0.5052
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6141 - accuracy: 0.5034
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6496 - accuracy: 0.5011
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6544 - accuracy: 0.5008
11000/25000 [============>.................] - ETA: 4s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 4s - loss: 7.6257 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6253 - accuracy: 0.5027
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 3s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6350 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6351 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6545 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6681 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6877 - accuracy: 0.4986
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 08:15:28.849450
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 08:15:28.849450  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 08:15:35.408879: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 08:15:35.414814: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 08:15:35.414999: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5648d09eb170 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 08:15:35.415015: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1072ee4dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7395 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.6228 - val_crf_viterbi_accuracy: 0.2533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f10691868d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 11s - loss: 7.5363 - accuracy: 0.5085
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5235 - accuracy: 0.5093 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6237 - accuracy: 0.5028
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6184 - accuracy: 0.5031
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6257 - accuracy: 0.5027
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
11000/25000 [============>.................] - ETA: 4s - loss: 7.6847 - accuracy: 0.4988
12000/25000 [=============>................] - ETA: 4s - loss: 7.6883 - accuracy: 0.4986
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6808 - accuracy: 0.4991
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6852 - accuracy: 0.4988
15000/25000 [=================>............] - ETA: 3s - loss: 7.6820 - accuracy: 0.4990
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6802 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6481 - accuracy: 0.5012
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6298 - accuracy: 0.5024
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6396 - accuracy: 0.5018
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6332 - accuracy: 0.5022
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6380 - accuracy: 0.5019
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6538 - accuracy: 0.5008
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f10680535f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:02:06, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:21:21, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:30:08, 20.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:03:34, 29.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.20M/862M [00:01<5:40:14, 42.2kB/s].vector_cache/glove.6B.zip:   1%|          | 4.51M/862M [00:01<3:57:25, 60.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.58M/862M [00:01<2:45:18, 86.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.8M/862M [00:01<1:55:15, 123kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.4M/862M [00:01<1:20:12, 175kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.3M/862M [00:01<56:06, 249kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:01<39:05, 356kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:02<27:24, 506kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.5M/862M [00:02<19:07, 719kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<13:29, 1.02MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<09:28, 1.44MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<06:40, 2.03MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<05:49, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<05:58, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:17, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<04:55, 2.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<05:49, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<06:04, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:07<04:41, 2.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<05:44, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<05:29, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<04:08, 3.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<05:47, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<06:59, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<05:37, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:11<04:06, 3.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<18:47, 700kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<14:30, 906kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<10:28, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<10:23, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<09:55, 1.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<07:37, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:15<05:29, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<1:37:21, 134kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<1:09:28, 187kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<48:52, 266kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<37:07, 349kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:19<27:17, 474kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<19:20, 667kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<16:34, 777kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<14:13, 905kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:21<10:37, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:21<07:34, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<1:36:45, 132kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<1:09:01, 185kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<48:29, 263kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<36:51, 346kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<28:22, 449kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:25<20:29, 621kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<16:21, 775kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<12:46, 992kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<09:15, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:22, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:51, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<05:48, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:02, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:28, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:10, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<14:25, 864kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:23, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:16, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:41, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:42, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:38, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<04:49, 2.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:18, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:05, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<05:13, 2.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:31, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:48, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:22, 2.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:57, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:23, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:02, 3.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:42, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:28, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:02, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:41, 3.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:06, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:54, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:05, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:22, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:53, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:25, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:43, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:12, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:53, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:30, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:03, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:49, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:27, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:01, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:48, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:49, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:32, 3.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:39, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:33, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:52, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:06, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:45, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:19, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:50, 3.00MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<41:50, 275kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<30:28, 377kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<21:31, 533kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<17:43, 645kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<14:43, 776kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<10:52, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<09:25, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:45, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:41, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:37, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:47, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:19, 2.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:09, 3.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:16:25, 147kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<54:25, 206kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<38:15, 293kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<26:50, 416kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<1:00:57, 183kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<43:46, 255kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<30:51, 361kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<24:09, 459kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<19:10, 578kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<13:58, 793kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<09:51, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<10:44:41, 17.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<7:32:09, 24.4kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<5:16:01, 34.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<3:43:03, 49.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<2:38:18, 69.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<1:51:15, 98.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:19:16, 137kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<56:37, 192kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<39:47, 273kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<27:53, 388kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:39:51, 108kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:12:04, 150kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<50:52, 212kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<35:40, 302kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<29:02, 370kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<21:26, 501kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<15:14, 703kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<13:08, 812kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<11:31, 927kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:33, 1.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:15, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:49, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:19, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:30, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:45, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<03:25, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<10:07:21, 17.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<7:05:57, 24.6kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<4:57:40, 35.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<3:30:05, 49.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<2:29:08, 69.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<1:44:43, 99.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<1:13:06, 142kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:03:50, 162kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<45:44, 226kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<32:10, 320kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<24:51, 413kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<19:28, 527kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<14:05, 728kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<10:03, 1.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<09:54, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:59, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:48, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:25, 1.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:32, 1.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:08, 2.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:12, 1.93MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<05:41, 1.77MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:29, 2.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:14, 3.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<10:29, 952kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<08:21, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:05, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:34, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:37, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:03, 1.96MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:45<03:41, 2.68MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:23, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:29, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:05, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:08, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:34, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:26, 2.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:42, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:15, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:05, 2.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:00, 3.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:59, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:10, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:51, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:56, 1.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:24, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:16, 2.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<03:05, 3.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<1:10:08, 136kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<49:54, 191kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<35:05, 270kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<26:40, 354kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<20:35, 459kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<14:51, 634kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<11:52, 790kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<09:15, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:42, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:50, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:41, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:09, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:42, 2.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<1:08:49, 134kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<49:06, 188kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<34:28, 267kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<26:11, 350kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<20:13, 453kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<14:32, 630kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<10:13, 891kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<16:35, 549kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<12:31, 726kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:58, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:22, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:41, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:46, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<04:08, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:04, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:33, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:48, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:26, 1.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:42, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:28, 2.55MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:31, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:54, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:08, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:38, 2.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:43, 3.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:18<02:01, 4.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<36:27, 238kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<27:17, 318kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<19:27, 446kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<13:39, 632kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<15:45, 547kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<11:55, 723kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<08:32, 1.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:56, 1.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:17, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:31, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:13, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:32, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:22, 2.50MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:20, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:53, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:55, 2.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:01, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:30, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:31, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:33, 3.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<09:18, 890kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<07:21, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:21, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:39, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:42, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:21, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:07, 2.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<08:57, 909kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:58, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:05, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:23, 1.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:24, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:08, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:58, 2.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<14:51, 539kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<11:12, 714kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<07:59, 998kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:26, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:49, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:10, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:52, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:00, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:50, 2.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:47, 2.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:18, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:12, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:49, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:27, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:54, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:55, 2.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:51, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:18, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:24, 3.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:02, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:17, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:11, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:00, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:20, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:21, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:27, 3.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:38, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:59, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:48, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<02:34, 2.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:31, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:12, 2.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:25, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:24, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:48, 1.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<02:59, 2.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<02:10, 3.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<6:55:42, 17.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<4:51:29, 24.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<3:23:27, 35.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<2:23:23, 49.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<1:41:00, 70.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:10:37, 100kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<50:51, 139kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<36:16, 194kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<25:27, 276kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<19:23, 360kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<14:16, 488kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<10:07, 686kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<08:41, 795kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<07:29, 923kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:34, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:59, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:53, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:43, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:42, 2.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:16, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:34, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:37, 2.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<01:57, 3.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<18:45, 358kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<13:47, 486kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<09:47, 681kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:22, 793kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<07:16, 912kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:22, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<03:52, 1.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:47, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:55, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:53, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:06, 3.09MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<17:10, 378kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<13:20, 487kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<09:36, 675kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<06:45, 954kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<11:37, 553kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<08:41, 739kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<06:12, 1.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<04:24, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<19:08, 332kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<14:41, 433kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<10:32, 601kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<07:24, 850kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<09:27, 666kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<07:14, 867kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<05:12, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:05, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:11, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:03, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:35, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:45, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:53, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:04, 2.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:41, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:37, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:21, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:45, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:50, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:56, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:08, 2.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:09, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:30, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:36, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:11, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:25, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:39, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:54, 3.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<05:55, 982kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:43, 1.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:26, 1.68MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:44, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:11, 1.79MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:22, 2.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:59, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:14, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:31, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:48, 3.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<07:45, 722kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<05:59, 934kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:17, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:17, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:28, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:31, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:50, 2.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<11:37, 471kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<09:14, 592kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<06:43, 810kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<04:43, 1.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<5:14:04, 17.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<3:40:02, 24.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<2:33:25, 35.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<1:46:48, 50.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<1:28:09, 60.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<1:02:44, 84.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<44:02, 121kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<30:42, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<23:46, 221kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<17:09, 307kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<12:03, 434kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<09:36, 541kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<07:46, 667kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:40, 914kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:01, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:38, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:46, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:44, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:05, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:11, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:28, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<01:46, 2.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<25:31, 195kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<18:21, 272kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<12:53, 385kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<10:07, 486kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<07:34, 649kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<05:22, 909kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:52, 994kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<04:24, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:19, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:04, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:33, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:54, 2.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:25, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:38, 2.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:14, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:31, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:59, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:07, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:57, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:27, 3.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:04, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:23, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:53, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:21, 3.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<4:15:58, 17.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<2:59:23, 24.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<2:04:56, 35.3kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<1:27:46, 49.8kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<1:02:20, 70.1kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<43:42, 99.7kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<30:22, 142kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<23:48, 181kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<17:05, 252kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<11:59, 356kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<09:19, 454kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<06:56, 609kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:55, 854kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:24, 947kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:56, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:56, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<02:04, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<05:10, 793kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<04:02, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:54, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:58, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:56, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:13, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:36, 2.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:46, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:21, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:44, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:07, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:52, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:24, 2.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:52, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:05, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:37, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:10, 3.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:43, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:16, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:41, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:01, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:47, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:20, 2.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:46, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:58, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:32, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:06, 3.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:34, 994kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:51, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:04, 1.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:15, 1.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:17, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:46, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:16, 2.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<25:14, 135kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<17:59, 189kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<12:34, 269kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<09:28, 353kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<07:18, 457kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<05:14, 634kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<03:40, 896kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<04:27, 734kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:26, 948kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<02:28, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:27, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:02, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:29, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:46, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:33, 2.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:09, 2.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:31, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:41, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:19, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<00:56, 3.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<08:21, 359kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<06:08, 487kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<04:20, 684kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:41, 795kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:52, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:04, 1.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:06, 1.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:45, 1.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:17, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:33, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:38, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:15, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:53, 3.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:27, 611kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:23, 801kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:25, 1.11MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<02:18, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:52, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:22, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:33, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:36, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:15, 2.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<00:53, 2.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<18:39, 135kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<13:17, 189kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<09:16, 269kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<06:58, 352kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<05:22, 456kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:50, 634kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:40, 895kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:08, 757kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:26, 973kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:45, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:45, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:24, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:03, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:45, 2.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<10:49, 208kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<08:01, 280kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<05:40, 393kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<03:55, 557kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:13, 516kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<03:09, 686kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<02:14, 956kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:02, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:36, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:08, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:49, 2.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<15:49, 129kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<11:15, 181kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<07:49, 257kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<05:50, 337kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<04:29, 439kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<03:13, 607kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:30, 761kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:56, 977kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:23, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:22, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:08, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:50, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:59, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:03, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:49, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:34, 2.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:48, 943kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:24, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:00, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:04, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:54, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:40, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:49, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:53, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:41, 2.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:30, 3.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:52, 1.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:45, 1.94MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:33, 2.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:42, 1.99MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:47, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:48<00:26, 3.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:57, 695kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:29, 902kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:03, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:01, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:58, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:30, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:34, 777kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:13, 995kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:51, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:51, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:49, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:38, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:35, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:22, 2.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:33, 1.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:18, 3.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:20, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:26, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:22, 2.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:16, 3.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:24, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:22, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:16, 2.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:24, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:18, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:12, 3.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<02:34, 261kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:51, 358kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<01:15, 506kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:58, 618kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:48, 748kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:34, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:23, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:26, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:21, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:14, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:10, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<01:59, 234kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:28, 313kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:01, 438kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:39, 621kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:43, 548kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:32, 724kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:21, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:18, 1.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:06, 2.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<01:03, 246kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:44, 339kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:28, 478kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:19, 588kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:14, 774kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:08, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.98MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 739/400000 [00:00<00:54, 7386.57it/s]  0%|          | 1457/400000 [00:00<00:54, 7322.31it/s]  1%|          | 2233/400000 [00:00<00:53, 7447.46it/s]  1%|          | 3017/400000 [00:00<00:52, 7560.85it/s]  1%|          | 3792/400000 [00:00<00:52, 7616.16it/s]  1%|          | 4542/400000 [00:00<00:52, 7580.51it/s]  1%|▏         | 5255/400000 [00:00<00:53, 7436.65it/s]  1%|▏         | 5984/400000 [00:00<00:53, 7390.67it/s]  2%|▏         | 6727/400000 [00:00<00:53, 7400.11it/s]  2%|▏         | 7486/400000 [00:01<00:52, 7454.07it/s]  2%|▏         | 8223/400000 [00:01<00:52, 7428.58it/s]  2%|▏         | 8976/400000 [00:01<00:52, 7458.71it/s]  2%|▏         | 9740/400000 [00:01<00:51, 7509.94it/s]  3%|▎         | 10518/400000 [00:01<00:51, 7587.66it/s]  3%|▎         | 11291/400000 [00:01<00:50, 7627.46it/s]  3%|▎         | 12057/400000 [00:01<00:50, 7635.61it/s]  3%|▎         | 12819/400000 [00:01<00:51, 7539.02it/s]  3%|▎         | 13572/400000 [00:01<00:51, 7470.41it/s]  4%|▎         | 14319/400000 [00:01<00:51, 7425.59it/s]  4%|▍         | 15062/400000 [00:02<00:52, 7358.57it/s]  4%|▍         | 15798/400000 [00:02<00:54, 7082.45it/s]  4%|▍         | 16509/400000 [00:02<00:54, 7068.90it/s]  4%|▍         | 17234/400000 [00:02<00:53, 7121.16it/s]  4%|▍         | 17975/400000 [00:02<00:53, 7204.68it/s]  5%|▍         | 18715/400000 [00:02<00:52, 7260.98it/s]  5%|▍         | 19458/400000 [00:02<00:52, 7308.87it/s]  5%|▌         | 20190/400000 [00:02<00:52, 7300.45it/s]  5%|▌         | 20932/400000 [00:02<00:51, 7333.45it/s]  5%|▌         | 21679/400000 [00:02<00:51, 7371.76it/s]  6%|▌         | 22417/400000 [00:03<00:51, 7363.06it/s]  6%|▌         | 23154/400000 [00:03<00:51, 7334.93it/s]  6%|▌         | 23888/400000 [00:03<00:51, 7316.49it/s]  6%|▌         | 24623/400000 [00:03<00:51, 7326.10it/s]  6%|▋         | 25364/400000 [00:03<00:51, 7344.58it/s]  7%|▋         | 26104/400000 [00:03<00:50, 7358.68it/s]  7%|▋         | 26851/400000 [00:03<00:50, 7389.96it/s]  7%|▋         | 27591/400000 [00:03<00:51, 7206.45it/s]  7%|▋         | 28337/400000 [00:03<00:51, 7279.63it/s]  7%|▋         | 29075/400000 [00:03<00:50, 7307.64it/s]  7%|▋         | 29840/400000 [00:04<00:49, 7404.20it/s]  8%|▊         | 30609/400000 [00:04<00:49, 7485.29it/s]  8%|▊         | 31359/400000 [00:04<00:49, 7414.18it/s]  8%|▊         | 32102/400000 [00:04<00:49, 7396.29it/s]  8%|▊         | 32880/400000 [00:04<00:48, 7506.28it/s]  8%|▊         | 33641/400000 [00:04<00:48, 7535.42it/s]  9%|▊         | 34396/400000 [00:04<00:48, 7486.41it/s]  9%|▉         | 35146/400000 [00:04<00:48, 7457.39it/s]  9%|▉         | 35903/400000 [00:04<00:48, 7490.61it/s]  9%|▉         | 36665/400000 [00:04<00:48, 7526.89it/s]  9%|▉         | 37418/400000 [00:05<00:49, 7264.94it/s] 10%|▉         | 38187/400000 [00:05<00:48, 7386.88it/s] 10%|▉         | 38928/400000 [00:05<00:48, 7384.23it/s] 10%|▉         | 39668/400000 [00:05<00:49, 7335.47it/s] 10%|█         | 40403/400000 [00:05<00:49, 7192.69it/s] 10%|█         | 41158/400000 [00:05<00:49, 7293.98it/s] 10%|█         | 41909/400000 [00:05<00:48, 7355.02it/s] 11%|█         | 42646/400000 [00:05<00:48, 7334.25it/s] 11%|█         | 43399/400000 [00:05<00:48, 7391.65it/s] 11%|█         | 44150/400000 [00:05<00:47, 7425.58it/s] 11%|█         | 44894/400000 [00:06<00:47, 7410.52it/s] 11%|█▏        | 45645/400000 [00:06<00:47, 7438.97it/s] 12%|█▏        | 46390/400000 [00:06<00:47, 7440.27it/s] 12%|█▏        | 47148/400000 [00:06<00:47, 7479.15it/s] 12%|█▏        | 47897/400000 [00:06<00:47, 7482.26it/s] 12%|█▏        | 48661/400000 [00:06<00:46, 7527.86it/s] 12%|█▏        | 49414/400000 [00:06<00:46, 7479.10it/s] 13%|█▎        | 50163/400000 [00:06<00:47, 7429.49it/s] 13%|█▎        | 50907/400000 [00:06<00:47, 7405.68it/s] 13%|█▎        | 51648/400000 [00:06<00:48, 7158.90it/s] 13%|█▎        | 52405/400000 [00:07<00:47, 7276.52it/s] 13%|█▎        | 53135/400000 [00:07<00:48, 7090.89it/s] 13%|█▎        | 53889/400000 [00:07<00:47, 7218.84it/s] 14%|█▎        | 54614/400000 [00:07<00:47, 7200.96it/s] 14%|█▍        | 55355/400000 [00:07<00:47, 7259.87it/s] 14%|█▍        | 56095/400000 [00:07<00:47, 7299.69it/s] 14%|█▍        | 56829/400000 [00:07<00:46, 7309.75it/s] 14%|█▍        | 57561/400000 [00:07<00:47, 7284.16it/s] 15%|█▍        | 58290/400000 [00:07<00:46, 7279.34it/s] 15%|█▍        | 59019/400000 [00:08<00:47, 7250.42it/s] 15%|█▍        | 59753/400000 [00:08<00:46, 7275.44it/s] 15%|█▌        | 60488/400000 [00:08<00:46, 7297.56it/s] 15%|█▌        | 61218/400000 [00:08<00:46, 7282.69it/s] 15%|█▌        | 61947/400000 [00:08<00:47, 7072.40it/s] 16%|█▌        | 62656/400000 [00:08<00:48, 6961.65it/s] 16%|█▌        | 63384/400000 [00:08<00:47, 7053.95it/s] 16%|█▌        | 64121/400000 [00:08<00:47, 7145.26it/s] 16%|█▌        | 64849/400000 [00:08<00:46, 7185.04it/s] 16%|█▋        | 65569/400000 [00:08<00:46, 7144.56it/s] 17%|█▋        | 66301/400000 [00:09<00:46, 7195.22it/s] 17%|█▋        | 67030/400000 [00:09<00:46, 7220.87it/s] 17%|█▋        | 67768/400000 [00:09<00:45, 7265.39it/s] 17%|█▋        | 68495/400000 [00:09<00:45, 7258.33it/s] 17%|█▋        | 69222/400000 [00:09<00:45, 7255.50it/s] 17%|█▋        | 69956/400000 [00:09<00:45, 7280.03it/s] 18%|█▊        | 70702/400000 [00:09<00:44, 7333.06it/s] 18%|█▊        | 71439/400000 [00:09<00:44, 7342.22it/s] 18%|█▊        | 72174/400000 [00:09<00:44, 7324.41it/s] 18%|█▊        | 72907/400000 [00:09<00:45, 7235.65it/s] 18%|█▊        | 73634/400000 [00:10<00:45, 7244.48it/s] 19%|█▊        | 74359/400000 [00:10<00:45, 7221.60it/s] 19%|█▉        | 75101/400000 [00:10<00:44, 7279.92it/s] 19%|█▉        | 75830/400000 [00:10<00:44, 7275.43it/s] 19%|█▉        | 76564/400000 [00:10<00:44, 7292.10it/s] 19%|█▉        | 77304/400000 [00:10<00:44, 7322.83it/s] 20%|█▉        | 78038/400000 [00:10<00:43, 7327.72it/s] 20%|█▉        | 78771/400000 [00:10<00:43, 7313.29it/s] 20%|█▉        | 79506/400000 [00:10<00:43, 7323.21it/s] 20%|██        | 80239/400000 [00:10<00:43, 7302.98it/s] 20%|██        | 80977/400000 [00:11<00:43, 7324.49it/s] 20%|██        | 81722/400000 [00:11<00:43, 7361.38it/s] 21%|██        | 82459/400000 [00:11<00:43, 7362.96it/s] 21%|██        | 83198/400000 [00:11<00:42, 7370.43it/s] 21%|██        | 83936/400000 [00:11<00:43, 7225.81it/s] 21%|██        | 84664/400000 [00:11<00:43, 7240.85it/s] 21%|██▏       | 85389/400000 [00:11<00:43, 7159.65it/s] 22%|██▏       | 86121/400000 [00:11<00:43, 7204.29it/s] 22%|██▏       | 86842/400000 [00:11<00:44, 7103.98it/s] 22%|██▏       | 87568/400000 [00:11<00:43, 7147.71it/s] 22%|██▏       | 88313/400000 [00:12<00:43, 7234.81it/s] 22%|██▏       | 89064/400000 [00:12<00:42, 7313.26it/s] 22%|██▏       | 89808/400000 [00:12<00:42, 7348.92it/s] 23%|██▎       | 90546/400000 [00:12<00:42, 7356.66it/s] 23%|██▎       | 91283/400000 [00:12<00:42, 7306.80it/s] 23%|██▎       | 92014/400000 [00:12<00:42, 7300.56it/s] 23%|██▎       | 92759/400000 [00:12<00:41, 7343.01it/s] 23%|██▎       | 93502/400000 [00:12<00:41, 7367.58it/s] 24%|██▎       | 94239/400000 [00:12<00:41, 7368.22it/s] 24%|██▎       | 94976/400000 [00:12<00:41, 7286.84it/s] 24%|██▍       | 95722/400000 [00:13<00:41, 7336.83it/s] 24%|██▍       | 96456/400000 [00:13<00:41, 7267.04it/s] 24%|██▍       | 97204/400000 [00:13<00:41, 7327.21it/s] 24%|██▍       | 97938/400000 [00:13<00:41, 7290.35it/s] 25%|██▍       | 98668/400000 [00:13<00:41, 7269.22it/s] 25%|██▍       | 99398/400000 [00:13<00:41, 7277.66it/s] 25%|██▌       | 100141/400000 [00:13<00:40, 7322.33it/s] 25%|██▌       | 100889/400000 [00:13<00:40, 7367.34it/s] 25%|██▌       | 101631/400000 [00:13<00:40, 7382.71it/s] 26%|██▌       | 102370/400000 [00:13<00:40, 7308.18it/s] 26%|██▌       | 103115/400000 [00:14<00:40, 7349.87it/s] 26%|██▌       | 103856/400000 [00:14<00:40, 7366.65it/s] 26%|██▌       | 104593/400000 [00:14<00:40, 7324.93it/s] 26%|██▋       | 105326/400000 [00:14<00:40, 7326.07it/s] 27%|██▋       | 106059/400000 [00:14<00:40, 7244.24it/s] 27%|██▋       | 106784/400000 [00:14<00:40, 7220.93it/s] 27%|██▋       | 107507/400000 [00:14<00:40, 7218.61it/s] 27%|██▋       | 108230/400000 [00:14<00:40, 7198.19it/s] 27%|██▋       | 108974/400000 [00:14<00:40, 7268.78it/s] 27%|██▋       | 109702/400000 [00:14<00:41, 6956.56it/s] 28%|██▊       | 110401/400000 [00:15<00:42, 6864.02it/s] 28%|██▊       | 111138/400000 [00:15<00:41, 7007.77it/s] 28%|██▊       | 111895/400000 [00:15<00:40, 7166.40it/s] 28%|██▊       | 112643/400000 [00:15<00:39, 7256.36it/s] 28%|██▊       | 113384/400000 [00:15<00:39, 7301.67it/s] 29%|██▊       | 114126/400000 [00:15<00:38, 7336.35it/s] 29%|██▊       | 114874/400000 [00:15<00:38, 7377.95it/s] 29%|██▉       | 115617/400000 [00:15<00:38, 7391.41it/s] 29%|██▉       | 116365/400000 [00:15<00:38, 7415.10it/s] 29%|██▉       | 117107/400000 [00:15<00:38, 7395.00it/s] 29%|██▉       | 117847/400000 [00:16<00:38, 7365.91it/s] 30%|██▉       | 118596/400000 [00:16<00:38, 7401.60it/s] 30%|██▉       | 119338/400000 [00:16<00:37, 7405.93it/s] 30%|███       | 120085/400000 [00:16<00:37, 7424.36it/s] 30%|███       | 120828/400000 [00:16<00:37, 7367.99it/s] 30%|███       | 121578/400000 [00:16<00:37, 7404.88it/s] 31%|███       | 122319/400000 [00:16<00:37, 7398.40it/s] 31%|███       | 123059/400000 [00:16<00:37, 7388.93it/s] 31%|███       | 123798/400000 [00:16<00:37, 7372.36it/s] 31%|███       | 124536/400000 [00:17<00:37, 7303.00it/s] 31%|███▏      | 125267/400000 [00:17<00:37, 7287.72it/s] 32%|███▏      | 126002/400000 [00:17<00:37, 7304.49it/s] 32%|███▏      | 126733/400000 [00:17<00:37, 7305.85it/s] 32%|███▏      | 127472/400000 [00:17<00:37, 7329.64it/s] 32%|███▏      | 128206/400000 [00:17<00:37, 7312.98it/s] 32%|███▏      | 128938/400000 [00:17<00:37, 7292.72it/s] 32%|███▏      | 129677/400000 [00:17<00:36, 7319.04it/s] 33%|███▎      | 130409/400000 [00:17<00:37, 7235.69it/s] 33%|███▎      | 131148/400000 [00:17<00:36, 7280.10it/s] 33%|███▎      | 131882/400000 [00:18<00:36, 7297.28it/s] 33%|███▎      | 132612/400000 [00:18<00:36, 7271.78it/s] 33%|███▎      | 133362/400000 [00:18<00:36, 7338.67it/s] 34%|███▎      | 134097/400000 [00:18<00:36, 7296.89it/s] 34%|███▎      | 134827/400000 [00:18<00:36, 7288.69it/s] 34%|███▍      | 135559/400000 [00:18<00:36, 7295.95it/s] 34%|███▍      | 136294/400000 [00:18<00:36, 7310.27it/s] 34%|███▍      | 137035/400000 [00:18<00:35, 7339.67it/s] 34%|███▍      | 137778/400000 [00:18<00:35, 7365.75it/s] 35%|███▍      | 138515/400000 [00:18<00:35, 7319.01it/s] 35%|███▍      | 139248/400000 [00:19<00:35, 7322.04it/s] 35%|███▍      | 139985/400000 [00:19<00:35, 7336.28it/s] 35%|███▌      | 140726/400000 [00:19<00:35, 7357.56it/s] 35%|███▌      | 141464/400000 [00:19<00:35, 7362.21it/s] 36%|███▌      | 142201/400000 [00:19<00:35, 7253.24it/s] 36%|███▌      | 142936/400000 [00:19<00:35, 7279.50it/s] 36%|███▌      | 143674/400000 [00:19<00:35, 7309.07it/s] 36%|███▌      | 144416/400000 [00:19<00:34, 7340.20it/s] 36%|███▋      | 145151/400000 [00:19<00:34, 7337.75it/s] 36%|███▋      | 145889/400000 [00:19<00:34, 7349.78it/s] 37%|███▋      | 146625/400000 [00:20<00:36, 6992.79it/s] 37%|███▋      | 147352/400000 [00:20<00:35, 7062.31it/s] 37%|███▋      | 148092/400000 [00:20<00:35, 7158.22it/s] 37%|███▋      | 148846/400000 [00:20<00:34, 7268.24it/s] 37%|███▋      | 149599/400000 [00:20<00:34, 7344.24it/s] 38%|███▊      | 150347/400000 [00:20<00:33, 7381.59it/s] 38%|███▊      | 151087/400000 [00:20<00:33, 7375.93it/s] 38%|███▊      | 151826/400000 [00:20<00:33, 7330.16it/s] 38%|███▊      | 152560/400000 [00:20<00:34, 7276.42it/s] 38%|███▊      | 153314/400000 [00:20<00:33, 7352.41it/s] 39%|███▊      | 154066/400000 [00:21<00:33, 7400.92it/s] 39%|███▊      | 154807/400000 [00:21<00:33, 7299.93it/s] 39%|███▉      | 155545/400000 [00:21<00:33, 7322.10it/s] 39%|███▉      | 156281/400000 [00:21<00:33, 7332.26it/s] 39%|███▉      | 157032/400000 [00:21<00:32, 7382.98it/s] 39%|███▉      | 157779/400000 [00:21<00:32, 7408.25it/s] 40%|███▉      | 158525/400000 [00:21<00:32, 7421.92it/s] 40%|███▉      | 159269/400000 [00:21<00:32, 7427.14it/s] 40%|████      | 160012/400000 [00:21<00:32, 7425.65it/s] 40%|████      | 160755/400000 [00:21<00:32, 7422.71it/s] 40%|████      | 161499/400000 [00:22<00:32, 7427.70it/s] 41%|████      | 162242/400000 [00:22<00:32, 7413.02it/s] 41%|████      | 162987/400000 [00:22<00:31, 7421.33it/s] 41%|████      | 163736/400000 [00:22<00:31, 7441.51it/s] 41%|████      | 164481/400000 [00:22<00:31, 7434.00it/s] 41%|████▏     | 165225/400000 [00:22<00:31, 7403.67it/s] 41%|████▏     | 165966/400000 [00:22<00:31, 7400.52it/s] 42%|████▏     | 166712/400000 [00:22<00:31, 7416.48it/s] 42%|████▏     | 167460/400000 [00:22<00:31, 7434.64it/s] 42%|████▏     | 168214/400000 [00:22<00:31, 7464.43it/s] 42%|████▏     | 168966/400000 [00:23<00:30, 7477.51it/s] 42%|████▏     | 169714/400000 [00:23<00:31, 7371.54it/s] 43%|████▎     | 170452/400000 [00:23<00:31, 7365.60it/s] 43%|████▎     | 171206/400000 [00:23<00:30, 7414.19it/s] 43%|████▎     | 171952/400000 [00:23<00:30, 7426.55it/s] 43%|████▎     | 172695/400000 [00:23<00:30, 7427.12it/s] 43%|████▎     | 173438/400000 [00:23<00:30, 7354.68it/s] 44%|████▎     | 174187/400000 [00:23<00:30, 7392.64it/s] 44%|████▎     | 174927/400000 [00:23<00:30, 7384.93it/s] 44%|████▍     | 175666/400000 [00:23<00:30, 7375.80it/s] 44%|████▍     | 176404/400000 [00:24<00:30, 7377.04it/s] 44%|████▍     | 177142/400000 [00:24<00:30, 7282.72it/s] 44%|████▍     | 177881/400000 [00:24<00:30, 7312.12it/s] 45%|████▍     | 178613/400000 [00:24<00:30, 7293.56it/s] 45%|████▍     | 179348/400000 [00:24<00:30, 7309.22it/s] 45%|████▌     | 180082/400000 [00:24<00:30, 7317.57it/s] 45%|████▌     | 180814/400000 [00:24<00:30, 7290.06it/s] 45%|████▌     | 181560/400000 [00:24<00:29, 7339.92it/s] 46%|████▌     | 182304/400000 [00:24<00:29, 7368.15it/s] 46%|████▌     | 183049/400000 [00:24<00:29, 7390.31it/s] 46%|████▌     | 183789/400000 [00:25<00:29, 7218.97it/s] 46%|████▌     | 184513/400000 [00:25<00:29, 7224.78it/s] 46%|████▋     | 185251/400000 [00:25<00:29, 7270.23it/s] 46%|████▋     | 185992/400000 [00:25<00:29, 7308.73it/s] 47%|████▋     | 186739/400000 [00:25<00:28, 7354.47it/s] 47%|████▋     | 187489/400000 [00:25<00:28, 7395.94it/s] 47%|████▋     | 188229/400000 [00:25<00:28, 7396.78it/s] 47%|████▋     | 188969/400000 [00:25<00:28, 7396.00it/s] 47%|████▋     | 189722/400000 [00:25<00:28, 7433.78it/s] 48%|████▊     | 190471/400000 [00:25<00:28, 7447.85it/s] 48%|████▊     | 191221/400000 [00:26<00:27, 7462.89it/s] 48%|████▊     | 191968/400000 [00:26<00:28, 7362.15it/s] 48%|████▊     | 192705/400000 [00:26<00:28, 7354.40it/s] 48%|████▊     | 193441/400000 [00:26<00:28, 7328.82it/s] 49%|████▊     | 194195/400000 [00:26<00:27, 7388.42it/s] 49%|████▊     | 194941/400000 [00:26<00:27, 7407.10it/s] 49%|████▉     | 195682/400000 [00:26<00:27, 7391.57it/s] 49%|████▉     | 196422/400000 [00:26<00:27, 7373.15it/s] 49%|████▉     | 197160/400000 [00:26<00:27, 7374.19it/s] 49%|████▉     | 197899/400000 [00:26<00:27, 7377.32it/s] 50%|████▉     | 198647/400000 [00:27<00:27, 7405.89it/s] 50%|████▉     | 199388/400000 [00:27<00:27, 7379.80it/s] 50%|█████     | 200139/400000 [00:27<00:26, 7417.31it/s] 50%|█████     | 200896/400000 [00:27<00:26, 7461.21it/s] 50%|█████     | 201644/400000 [00:27<00:26, 7464.71it/s] 51%|█████     | 202391/400000 [00:27<00:26, 7443.11it/s] 51%|█████     | 203136/400000 [00:27<00:26, 7377.60it/s] 51%|█████     | 203881/400000 [00:27<00:26, 7398.87it/s] 51%|█████     | 204629/400000 [00:27<00:26, 7421.35it/s] 51%|█████▏    | 205376/400000 [00:27<00:26, 7433.94it/s] 52%|█████▏    | 206134/400000 [00:28<00:25, 7473.41it/s] 52%|█████▏    | 206882/400000 [00:28<00:25, 7453.12it/s] 52%|█████▏    | 207628/400000 [00:28<00:25, 7444.55it/s] 52%|█████▏    | 208373/400000 [00:28<00:25, 7425.81it/s] 52%|█████▏    | 209117/400000 [00:28<00:25, 7427.28it/s] 52%|█████▏    | 209870/400000 [00:28<00:25, 7454.83it/s] 53%|█████▎    | 210616/400000 [00:28<00:25, 7412.25it/s] 53%|█████▎    | 211364/400000 [00:28<00:25, 7428.74it/s] 53%|█████▎    | 212114/400000 [00:28<00:25, 7448.19it/s] 53%|█████▎    | 212864/400000 [00:29<00:25, 7462.42it/s] 53%|█████▎    | 213611/400000 [00:29<00:24, 7456.89it/s] 54%|█████▎    | 214357/400000 [00:29<00:25, 7360.80it/s] 54%|█████▍    | 215105/400000 [00:29<00:25, 7395.37it/s] 54%|█████▍    | 215854/400000 [00:29<00:24, 7423.07it/s] 54%|█████▍    | 216602/400000 [00:29<00:24, 7439.35it/s] 54%|█████▍    | 217351/400000 [00:29<00:24, 7453.01it/s] 55%|█████▍    | 218097/400000 [00:29<00:24, 7332.25it/s] 55%|█████▍    | 218837/400000 [00:29<00:24, 7351.52it/s] 55%|█████▍    | 219573/400000 [00:29<00:24, 7351.47it/s] 55%|█████▌    | 220309/400000 [00:30<00:24, 7285.05it/s] 55%|█████▌    | 221038/400000 [00:30<00:24, 7190.82it/s] 55%|█████▌    | 221760/400000 [00:30<00:24, 7198.57it/s] 56%|█████▌    | 222524/400000 [00:30<00:24, 7323.37it/s] 56%|█████▌    | 223273/400000 [00:30<00:23, 7371.31it/s] 56%|█████▌    | 224024/400000 [00:30<00:23, 7411.68it/s] 56%|█████▌    | 224767/400000 [00:30<00:23, 7416.12it/s] 56%|█████▋    | 225509/400000 [00:30<00:23, 7389.07it/s] 57%|█████▋    | 226265/400000 [00:30<00:23, 7437.43it/s] 57%|█████▋    | 227028/400000 [00:30<00:23, 7493.99it/s] 57%|█████▋    | 227796/400000 [00:31<00:22, 7546.95it/s] 57%|█████▋    | 228551/400000 [00:31<00:22, 7516.59it/s] 57%|█████▋    | 229303/400000 [00:31<00:23, 7415.58it/s] 58%|█████▊    | 230046/400000 [00:31<00:22, 7409.33it/s] 58%|█████▊    | 230798/400000 [00:31<00:22, 7441.00it/s] 58%|█████▊    | 231557/400000 [00:31<00:22, 7482.92it/s] 58%|█████▊    | 232311/400000 [00:31<00:22, 7499.50it/s] 58%|█████▊    | 233062/400000 [00:31<00:22, 7445.24it/s] 58%|█████▊    | 233807/400000 [00:31<00:22, 7415.72it/s] 59%|█████▊    | 234589/400000 [00:31<00:21, 7531.58it/s] 59%|█████▉    | 235343/400000 [00:32<00:21, 7497.11it/s] 59%|█████▉    | 236094/400000 [00:32<00:21, 7459.71it/s] 59%|█████▉    | 236841/400000 [00:32<00:21, 7447.88it/s] 59%|█████▉    | 237589/400000 [00:32<00:21, 7456.78it/s] 60%|█████▉    | 238342/400000 [00:32<00:21, 7478.41it/s] 60%|█████▉    | 239111/400000 [00:32<00:21, 7539.42it/s] 60%|█████▉    | 239880/400000 [00:32<00:21, 7581.09it/s] 60%|██████    | 240639/400000 [00:32<00:21, 7558.68it/s] 60%|██████    | 241396/400000 [00:32<00:21, 7546.21it/s] 61%|██████    | 242152/400000 [00:32<00:20, 7550.26it/s] 61%|██████    | 242908/400000 [00:33<00:20, 7523.88it/s] 61%|██████    | 243666/400000 [00:33<00:20, 7539.94it/s] 61%|██████    | 244421/400000 [00:33<00:20, 7508.75it/s] 61%|██████▏   | 245205/400000 [00:33<00:20, 7602.76it/s] 61%|██████▏   | 245966/400000 [00:33<00:20, 7528.83it/s] 62%|██████▏   | 246729/400000 [00:33<00:20, 7556.40it/s] 62%|██████▏   | 247485/400000 [00:33<00:20, 7551.06it/s] 62%|██████▏   | 248247/400000 [00:33<00:20, 7569.51it/s] 62%|██████▏   | 249018/400000 [00:33<00:19, 7602.46it/s] 62%|██████▏   | 249793/400000 [00:33<00:19, 7645.14it/s] 63%|██████▎   | 250563/400000 [00:34<00:19, 7659.38it/s] 63%|██████▎   | 251332/400000 [00:34<00:19, 7666.21it/s] 63%|██████▎   | 252100/400000 [00:34<00:19, 7668.69it/s] 63%|██████▎   | 252867/400000 [00:34<00:19, 7626.87it/s] 63%|██████▎   | 253630/400000 [00:34<00:19, 7595.57it/s] 64%|██████▎   | 254390/400000 [00:34<00:19, 7558.31it/s] 64%|██████▍   | 255163/400000 [00:34<00:19, 7608.95it/s] 64%|██████▍   | 255925/400000 [00:34<00:19, 7575.31it/s] 64%|██████▍   | 256683/400000 [00:34<00:18, 7546.26it/s] 64%|██████▍   | 257438/400000 [00:34<00:19, 7497.87it/s] 65%|██████▍   | 258188/400000 [00:35<00:19, 7448.73it/s] 65%|██████▍   | 258968/400000 [00:35<00:18, 7550.12it/s] 65%|██████▍   | 259766/400000 [00:35<00:18, 7673.45it/s] 65%|██████▌   | 260543/400000 [00:35<00:18, 7701.63it/s] 65%|██████▌   | 261325/400000 [00:35<00:17, 7736.17it/s] 66%|██████▌   | 262158/400000 [00:35<00:17, 7903.86it/s] 66%|██████▌   | 262950/400000 [00:35<00:17, 7875.59it/s] 66%|██████▌   | 263739/400000 [00:35<00:17, 7828.20it/s] 66%|██████▌   | 264523/400000 [00:35<00:17, 7773.34it/s] 66%|██████▋   | 265301/400000 [00:35<00:17, 7774.99it/s] 67%|██████▋   | 266094/400000 [00:36<00:17, 7819.40it/s] 67%|██████▋   | 266877/400000 [00:36<00:17, 7645.03it/s] 67%|██████▋   | 267643/400000 [00:36<00:17, 7554.10it/s] 67%|██████▋   | 268418/400000 [00:36<00:17, 7609.78it/s] 67%|██████▋   | 269203/400000 [00:36<00:17, 7677.30it/s] 67%|██████▋   | 269988/400000 [00:36<00:16, 7726.11it/s] 68%|██████▊   | 270762/400000 [00:36<00:16, 7683.71it/s] 68%|██████▊   | 271531/400000 [00:36<00:16, 7591.96it/s] 68%|██████▊   | 272308/400000 [00:36<00:16, 7643.12it/s] 68%|██████▊   | 273086/400000 [00:36<00:16, 7681.30it/s] 68%|██████▊   | 273881/400000 [00:37<00:16, 7759.35it/s] 69%|██████▊   | 274674/400000 [00:37<00:16, 7808.16it/s] 69%|██████▉   | 275456/400000 [00:37<00:15, 7792.74it/s] 69%|██████▉   | 276263/400000 [00:37<00:15, 7872.37it/s] 69%|██████▉   | 277081/400000 [00:37<00:15, 7961.53it/s] 69%|██████▉   | 277915/400000 [00:37<00:15, 8070.49it/s] 70%|██████▉   | 278756/400000 [00:37<00:14, 8167.52it/s] 70%|██████▉   | 279590/400000 [00:37<00:14, 8214.04it/s] 70%|███████   | 280413/400000 [00:37<00:14, 8030.08it/s] 70%|███████   | 281218/400000 [00:37<00:14, 8000.04it/s] 71%|███████   | 282019/400000 [00:38<00:15, 7859.17it/s] 71%|███████   | 282807/400000 [00:38<00:14, 7836.72it/s] 71%|███████   | 283592/400000 [00:38<00:15, 7754.95it/s] 71%|███████   | 284384/400000 [00:38<00:14, 7802.39it/s] 71%|███████▏  | 285188/400000 [00:38<00:14, 7872.11it/s] 71%|███████▏  | 285976/400000 [00:38<00:14, 7823.11it/s] 72%|███████▏  | 286765/400000 [00:38<00:14, 7839.93it/s] 72%|███████▏  | 287550/400000 [00:38<00:14, 7768.24it/s] 72%|███████▏  | 288328/400000 [00:38<00:14, 7694.57it/s] 72%|███████▏  | 289121/400000 [00:39<00:14, 7761.52it/s] 72%|███████▏  | 289913/400000 [00:39<00:14, 7806.40it/s] 73%|███████▎  | 290695/400000 [00:39<00:14, 7799.27it/s] 73%|███████▎  | 291476/400000 [00:39<00:14, 7745.73it/s] 73%|███████▎  | 292252/400000 [00:39<00:13, 7748.92it/s] 73%|███████▎  | 293048/400000 [00:39<00:13, 7809.18it/s] 73%|███████▎  | 293851/400000 [00:39<00:13, 7872.09it/s] 74%|███████▎  | 294671/400000 [00:39<00:13, 7965.56it/s] 74%|███████▍  | 295474/400000 [00:39<00:13, 7984.11it/s] 74%|███████▍  | 296273/400000 [00:39<00:13, 7782.59it/s] 74%|███████▍  | 297053/400000 [00:40<00:13, 7768.25it/s] 74%|███████▍  | 297831/400000 [00:40<00:13, 7752.81it/s] 75%|███████▍  | 298627/400000 [00:40<00:12, 7811.68it/s] 75%|███████▍  | 299438/400000 [00:40<00:12, 7895.78it/s] 75%|███████▌  | 300229/400000 [00:40<00:12, 7883.01it/s] 75%|███████▌  | 301056/400000 [00:40<00:12, 7994.09it/s] 75%|███████▌  | 301877/400000 [00:40<00:12, 8055.50it/s] 76%|███████▌  | 302684/400000 [00:40<00:12, 7994.42it/s] 76%|███████▌  | 303484/400000 [00:40<00:12, 7812.74it/s] 76%|███████▌  | 304267/400000 [00:40<00:12, 7717.58it/s] 76%|███████▋  | 305040/400000 [00:41<00:12, 7604.17it/s] 76%|███████▋  | 305824/400000 [00:41<00:12, 7673.18it/s] 77%|███████▋  | 306655/400000 [00:41<00:11, 7851.51it/s] 77%|███████▋  | 307470/400000 [00:41<00:11, 7937.81it/s] 77%|███████▋  | 308266/400000 [00:41<00:11, 7917.36it/s] 77%|███████▋  | 309114/400000 [00:41<00:11, 8077.19it/s] 77%|███████▋  | 309958/400000 [00:41<00:11, 8181.80it/s] 78%|███████▊  | 310778/400000 [00:41<00:10, 8128.17it/s] 78%|███████▊  | 311592/400000 [00:41<00:10, 8076.19it/s] 78%|███████▊  | 312401/400000 [00:41<00:10, 8012.20it/s] 78%|███████▊  | 313234/400000 [00:42<00:10, 8104.12it/s] 79%|███████▊  | 314123/400000 [00:42<00:10, 8324.36it/s] 79%|███████▊  | 314958/400000 [00:42<00:10, 8231.32it/s] 79%|███████▉  | 315792/400000 [00:42<00:10, 8259.64it/s] 79%|███████▉  | 316620/400000 [00:42<00:10, 8143.11it/s] 79%|███████▉  | 317491/400000 [00:42<00:09, 8303.42it/s] 80%|███████▉  | 318323/400000 [00:42<00:09, 8245.04it/s] 80%|███████▉  | 319149/400000 [00:42<00:09, 8246.57it/s] 80%|███████▉  | 319975/400000 [00:42<00:09, 8237.89it/s] 80%|████████  | 320800/400000 [00:42<00:09, 8161.21it/s] 80%|████████  | 321667/400000 [00:43<00:09, 8307.17it/s] 81%|████████  | 322533/400000 [00:43<00:09, 8406.96it/s] 81%|████████  | 323375/400000 [00:43<00:09, 8370.12it/s] 81%|████████  | 324233/400000 [00:43<00:08, 8429.61it/s] 81%|████████▏ | 325077/400000 [00:43<00:09, 8160.60it/s] 81%|████████▏ | 325896/400000 [00:43<00:09, 7998.72it/s] 82%|████████▏ | 326699/400000 [00:43<00:09, 7950.72it/s] 82%|████████▏ | 327496/400000 [00:43<00:09, 7944.87it/s] 82%|████████▏ | 328298/400000 [00:43<00:09, 7966.46it/s] 82%|████████▏ | 329096/400000 [00:44<00:09, 7845.20it/s] 82%|████████▏ | 329889/400000 [00:44<00:08, 7869.72it/s] 83%|████████▎ | 330686/400000 [00:44<00:08, 7898.78it/s] 83%|████████▎ | 331477/400000 [00:44<00:08, 7880.53it/s] 83%|████████▎ | 332267/400000 [00:44<00:08, 7884.98it/s] 83%|████████▎ | 333056/400000 [00:44<00:08, 7818.19it/s] 83%|████████▎ | 333851/400000 [00:44<00:08, 7855.09it/s] 84%|████████▎ | 334644/400000 [00:44<00:08, 7876.86it/s] 84%|████████▍ | 335441/400000 [00:44<00:08, 7904.33it/s] 84%|████████▍ | 336232/400000 [00:44<00:08, 7858.63it/s] 84%|████████▍ | 337019/400000 [00:45<00:08, 7640.11it/s] 84%|████████▍ | 337785/400000 [00:45<00:08, 7593.60it/s] 85%|████████▍ | 338546/400000 [00:45<00:08, 7551.43it/s] 85%|████████▍ | 339302/400000 [00:45<00:08, 7545.28it/s] 85%|████████▌ | 340064/400000 [00:45<00:07, 7567.42it/s] 85%|████████▌ | 340822/400000 [00:45<00:07, 7526.25it/s] 85%|████████▌ | 341584/400000 [00:45<00:07, 7553.99it/s] 86%|████████▌ | 342353/400000 [00:45<00:07, 7593.94it/s] 86%|████████▌ | 343138/400000 [00:45<00:07, 7668.59it/s] 86%|████████▌ | 343917/400000 [00:45<00:07, 7704.42it/s] 86%|████████▌ | 344688/400000 [00:46<00:07, 7623.75it/s] 86%|████████▋ | 345470/400000 [00:46<00:07, 7678.89it/s] 87%|████████▋ | 346275/400000 [00:46<00:06, 7785.09it/s] 87%|████████▋ | 347070/400000 [00:46<00:06, 7832.48it/s] 87%|████████▋ | 347854/400000 [00:46<00:06, 7717.71it/s] 87%|████████▋ | 348627/400000 [00:46<00:06, 7605.28it/s] 87%|████████▋ | 349389/400000 [00:46<00:06, 7563.08it/s] 88%|████████▊ | 350153/400000 [00:46<00:06, 7584.66it/s] 88%|████████▊ | 350918/400000 [00:46<00:06, 7602.66it/s] 88%|████████▊ | 351697/400000 [00:46<00:06, 7655.81it/s] 88%|████████▊ | 352463/400000 [00:47<00:06, 7633.34it/s] 88%|████████▊ | 353231/400000 [00:47<00:06, 7632.11it/s] 88%|████████▊ | 353995/400000 [00:47<00:06, 7572.22it/s] 89%|████████▊ | 354753/400000 [00:47<00:06, 7362.27it/s] 89%|████████▉ | 355493/400000 [00:47<00:06, 7372.54it/s] 89%|████████▉ | 356246/400000 [00:47<00:05, 7416.91it/s] 89%|████████▉ | 357007/400000 [00:47<00:05, 7473.00it/s] 89%|████████▉ | 357755/400000 [00:47<00:05, 7443.57it/s] 90%|████████▉ | 358535/400000 [00:47<00:05, 7545.91it/s] 90%|████████▉ | 359303/400000 [00:47<00:05, 7585.00it/s] 90%|█████████ | 360085/400000 [00:48<00:05, 7652.41it/s] 90%|█████████ | 360851/400000 [00:48<00:05, 7577.44it/s] 90%|█████████ | 361625/400000 [00:48<00:05, 7624.64it/s] 91%|█████████ | 362412/400000 [00:48<00:04, 7695.27it/s] 91%|█████████ | 363182/400000 [00:48<00:04, 7525.58it/s] 91%|█████████ | 363936/400000 [00:48<00:04, 7269.90it/s] 91%|█████████ | 364666/400000 [00:48<00:04, 7174.65it/s] 91%|█████████▏| 365415/400000 [00:48<00:04, 7264.59it/s] 92%|█████████▏| 366174/400000 [00:48<00:04, 7356.27it/s] 92%|█████████▏| 366943/400000 [00:48<00:04, 7451.44it/s] 92%|█████████▏| 367699/400000 [00:49<00:04, 7480.32it/s] 92%|█████████▏| 368491/400000 [00:49<00:04, 7605.61it/s] 92%|█████████▏| 369280/400000 [00:49<00:03, 7687.43it/s] 93%|█████████▎| 370087/400000 [00:49<00:03, 7797.35it/s] 93%|█████████▎| 370892/400000 [00:49<00:03, 7869.42it/s] 93%|█████████▎| 371680/400000 [00:49<00:03, 7357.00it/s] 93%|█████████▎| 372424/400000 [00:49<00:03, 7265.01it/s] 93%|█████████▎| 373156/400000 [00:49<00:03, 7127.30it/s] 93%|█████████▎| 373874/400000 [00:49<00:03, 7088.35it/s] 94%|█████████▎| 374586/400000 [00:50<00:03, 6802.56it/s] 94%|█████████▍| 375304/400000 [00:50<00:03, 6910.28it/s] 94%|█████████▍| 376059/400000 [00:50<00:03, 7088.77it/s] 94%|█████████▍| 376772/400000 [00:50<00:03, 7060.74it/s] 94%|█████████▍| 377481/400000 [00:50<00:03, 7026.48it/s] 95%|█████████▍| 378205/400000 [00:50<00:03, 7086.07it/s] 95%|█████████▍| 378954/400000 [00:50<00:02, 7202.13it/s] 95%|█████████▍| 379693/400000 [00:50<00:02, 7255.82it/s] 95%|█████████▌| 380467/400000 [00:50<00:02, 7392.96it/s] 95%|█████████▌| 381282/400000 [00:50<00:02, 7604.36it/s] 96%|█████████▌| 382055/400000 [00:51<00:02, 7641.41it/s] 96%|█████████▌| 382821/400000 [00:51<00:02, 7553.63it/s] 96%|█████████▌| 383587/400000 [00:51<00:02, 7585.19it/s] 96%|█████████▌| 384371/400000 [00:51<00:02, 7659.30it/s] 96%|█████████▋| 385138/400000 [00:51<00:01, 7545.57it/s] 96%|█████████▋| 385894/400000 [00:51<00:01, 7447.64it/s] 97%|█████████▋| 386640/400000 [00:51<00:01, 7439.71it/s] 97%|█████████▋| 387407/400000 [00:51<00:01, 7505.87it/s] 97%|█████████▋| 388160/400000 [00:51<00:01, 7512.97it/s] 97%|█████████▋| 388943/400000 [00:51<00:01, 7603.35it/s] 97%|█████████▋| 389753/400000 [00:52<00:01, 7744.08it/s] 98%|█████████▊| 390529/400000 [00:52<00:01, 7672.69it/s] 98%|█████████▊| 391342/400000 [00:52<00:01, 7802.74it/s] 98%|█████████▊| 392128/400000 [00:52<00:01, 7817.40it/s] 98%|█████████▊| 392911/400000 [00:52<00:00, 7807.08it/s] 98%|█████████▊| 393693/400000 [00:52<00:00, 7745.79it/s] 99%|█████████▊| 394469/400000 [00:52<00:00, 7639.05it/s] 99%|█████████▉| 395264/400000 [00:52<00:00, 7727.73it/s] 99%|█████████▉| 396043/400000 [00:52<00:00, 7744.68it/s] 99%|█████████▉| 396872/400000 [00:52<00:00, 7896.86it/s] 99%|█████████▉| 397663/400000 [00:53<00:00, 7883.60it/s]100%|█████████▉| 398453/400000 [00:53<00:00, 7815.93it/s]100%|█████████▉| 399236/400000 [00:53<00:00, 7607.67it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7527.57it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7492.56it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f10693742b0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010728031865858042 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.010933782145331137 	 Accuracy: 67

  model saves at 67% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15615 out of table with 15592 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15615 out of table with 15592 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 


  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/f5820f82e3fe7adb475287ba35fe87545042e303', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'f5820f82e3fe7adb475287ba35fe87545042e303', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/f5820f82e3fe7adb475287ba35fe87545042e303

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/f5820f82e3fe7adb475287ba35fe87545042e303

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f41bd8c3470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 04:12:48.528604
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 04:12:48.534012
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 04:12:48.537118
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 04:12:48.540654
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f41a93860b8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353539.5938
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 277787.2500
Epoch 3/10

1/1 [==============================] - 0s 93ms/step - loss: 171908.7344
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 96195.8516
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 53522.2969
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 31098.0000
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 19371.0977
Epoch 8/10

1/1 [==============================] - 0s 112ms/step - loss: 12966.7559
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 9258.0566
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 7003.6738

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.09712499 -0.7261611  -0.4366829   0.8035354  -0.20829698 -0.15000248
   1.009196    1.492562   -0.51555765 -0.13454121  1.4143229   1.1612588
  -0.54256266 -0.5361172   0.62728906  0.68725157  1.2948029   1.6967661
  -1.1192117  -0.34470868  0.43541354  0.02652121 -0.5211103  -0.2791253
   0.7203839  -1.1446157   0.49516317 -0.3623047  -0.57987815 -0.10866466
   0.7349768   0.58671606 -0.6263287  -1.0083951  -0.69345355  0.28277072
   0.26205885  0.12508428  0.7546903  -0.4314193   0.30314356  0.18538103
   1.0217012  -0.5619132   1.9986465  -1.382239   -0.75064015 -0.965696
   0.3910244  -1.1752331  -0.29875523 -1.3581424  -0.37806046  0.45794672
   0.70751053 -0.6074442   0.7245408   0.96167207  0.23336598  1.3045998
  -1.6787983  -0.04149212  1.7403432   0.08849141  1.1703764   1.0876794
  -0.3142022   0.7493909  -0.86933374  0.77979136  1.0068643  -0.74369276
  -0.01322806 -1.2584366  -0.92368054 -0.61377186  0.997681   -0.20226479
   1.5988688   0.4657212   0.70809674  1.2669196   0.11463444 -0.55408525
  -0.65873617 -0.7164189   0.10567451  0.8484578  -0.7137823  -0.7304545
  -1.4005747   0.78037155 -0.10377614 -1.4774777   0.66050404  0.10209993
   1.6032183   0.93449914 -0.4662829   0.6613559  -0.088238    0.4444095
  -1.4234239   0.1310498  -0.20527068 -1.2987818  -0.11314091 -0.18485953
  -1.2707131  -1.0499105   0.9593271  -0.20419931 -0.3942583  -0.40118712
   1.5367541   0.28501534 -0.38020247 -0.08064923 -0.57491165 -1.6898603
   0.04350552  8.993751    7.9630003   8.390953    7.944247    6.6524763
   9.091011    6.681841    7.410612    7.4370384   6.607836    9.633975
   9.124375    8.674491    8.309268    7.616537    8.483586    8.317769
   8.684448    7.1276217   7.79858     7.675224    7.268443    6.39229
   7.3156567   8.171615    7.4744287   6.938418    8.348772    8.393451
   6.4154277   7.8552046   8.120842    7.3316703   9.919419    7.4388494
   6.1908607   9.738302    8.296146    9.346045    8.245488    8.758602
   8.446337    7.275013    8.846326    7.7380958   8.173527    8.560205
   9.2232      7.611072    7.8758883   7.8514767   9.408113    6.6881256
   8.293615    7.5810847   9.196832    8.453534    8.610645    8.570343
   0.60533154  1.0459621   0.5158897   1.0040966   1.1613342   0.14157796
   2.2047243   0.8332521   0.8597304   1.5067911   1.2399094   0.40086615
   0.7918937   1.3706245   0.3104568   0.6083219   0.9024151   2.1496506
   2.3945806   1.0740459   2.0512805   1.5747845   0.5323188   0.51159525
   0.26925057  1.0479169   0.65172124  1.1698655   1.9318454   1.1509656
   1.2548639   1.6514608   2.463112    0.43991905  2.0222673   1.3051058
   1.7494735   0.7658545   0.59083676  1.5961031   0.36549997  0.5386425
   0.33741665  0.55798787  0.48271418  1.9158986   2.5416741   1.159275
   0.5064173   1.1015828   0.6340326   0.56278116  0.22936535  0.63763726
   1.8701446   1.9367492   0.7476149   0.42450106  1.0074593   1.3061697
   2.5981011   1.4896376   0.14467883  2.0879245   0.2817698   1.7090404
   0.536657    1.910228    1.8473958   0.3305512   1.7913338   1.0470576
   0.7190654   0.33033317  1.8414986   1.201508    1.7224491   0.2294401
   1.2814336   2.5423374   0.25703514  2.3270178   0.3138641   1.9077871
   1.93604     0.8636659   0.3432834   0.970824    0.47192085  1.028572
   0.732705    1.8440628   2.7601247   1.2065436   1.0651035   1.718682
   0.5703602   1.7364671   0.47344136  1.8156278   1.0437617   0.69024765
   0.44402444  1.8118105   0.9610895   1.6332731   0.40294373  0.59667194
   1.1593165   0.7809879   0.11171877  0.6418474   1.1353313   2.1599236
   2.2218313   1.3915237   1.2559353   1.7192261   0.22814786  0.81891483
   0.11048561  7.1379175   9.878812    8.1165495   6.920423    6.969763
   8.290126    7.655974    7.504012    7.612212    8.2731085   9.482368
   7.5233936   8.225931    9.376968    7.9609723   9.010319    9.138634
   8.783797    8.5584755   8.037142    7.610405    9.07357     7.3580837
   9.14783     7.988044    8.492333    8.195294    7.554464    8.03692
   9.803979    7.807564    7.9351077   7.5665913   8.262043    8.2297
   8.178507    7.715693    6.9289618   7.245529    8.676476    8.053303
   8.277156    7.399674    7.9730387   7.532358    8.216639    8.166237
   9.155754    8.925288    8.889087    7.5494556   8.294759    8.217019
   9.2361355   8.655865    7.445109    6.886811    6.947976    9.383545
  -4.460682   -8.2093725  11.882656  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 04:12:58.923138
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.7828
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 04:12:58.926839
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8817.41
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 04:12:58.929753
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.7883
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 04:12:58.933076
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -788.661
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139919231237032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139918699389840
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139918699390344
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139918699390848
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139918699391352
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139918699391856

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f41b1a95ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.612589
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.580787
grad_step = 000002, loss = 0.562221
grad_step = 000003, loss = 0.542752
grad_step = 000004, loss = 0.522014
grad_step = 000005, loss = 0.499984
grad_step = 000006, loss = 0.477461
grad_step = 000007, loss = 0.458131
grad_step = 000008, loss = 0.443661
grad_step = 000009, loss = 0.426852
grad_step = 000010, loss = 0.407855
grad_step = 000011, loss = 0.392545
grad_step = 000012, loss = 0.381770
grad_step = 000013, loss = 0.372452
grad_step = 000014, loss = 0.362667
grad_step = 000015, loss = 0.352382
grad_step = 000016, loss = 0.342320
grad_step = 000017, loss = 0.333151
grad_step = 000018, loss = 0.324231
grad_step = 000019, loss = 0.314242
grad_step = 000020, loss = 0.303718
grad_step = 000021, loss = 0.293699
grad_step = 000022, loss = 0.284199
grad_step = 000023, loss = 0.274708
grad_step = 000024, loss = 0.264942
grad_step = 000025, loss = 0.255297
grad_step = 000026, loss = 0.246228
grad_step = 000027, loss = 0.237624
grad_step = 000028, loss = 0.229164
grad_step = 000029, loss = 0.220733
grad_step = 000030, loss = 0.212429
grad_step = 000031, loss = 0.204278
grad_step = 000032, loss = 0.196325
grad_step = 000033, loss = 0.188593
grad_step = 000034, loss = 0.180821
grad_step = 000035, loss = 0.173077
grad_step = 000036, loss = 0.165607
grad_step = 000037, loss = 0.158513
grad_step = 000038, loss = 0.151610
grad_step = 000039, loss = 0.144687
grad_step = 000040, loss = 0.138032
grad_step = 000041, loss = 0.131790
grad_step = 000042, loss = 0.125702
grad_step = 000043, loss = 0.119736
grad_step = 000044, loss = 0.114034
grad_step = 000045, loss = 0.108548
grad_step = 000046, loss = 0.103208
grad_step = 000047, loss = 0.098061
grad_step = 000048, loss = 0.093128
grad_step = 000049, loss = 0.088369
grad_step = 000050, loss = 0.083842
grad_step = 000051, loss = 0.079463
grad_step = 000052, loss = 0.075267
grad_step = 000053, loss = 0.071339
grad_step = 000054, loss = 0.067551
grad_step = 000055, loss = 0.063899
grad_step = 000056, loss = 0.060447
grad_step = 000057, loss = 0.057164
grad_step = 000058, loss = 0.054014
grad_step = 000059, loss = 0.051013
grad_step = 000060, loss = 0.048145
grad_step = 000061, loss = 0.045437
grad_step = 000062, loss = 0.042849
grad_step = 000063, loss = 0.040391
grad_step = 000064, loss = 0.038077
grad_step = 000065, loss = 0.035866
grad_step = 000066, loss = 0.033773
grad_step = 000067, loss = 0.031804
grad_step = 000068, loss = 0.029926
grad_step = 000069, loss = 0.028148
grad_step = 000070, loss = 0.026464
grad_step = 000071, loss = 0.024874
grad_step = 000072, loss = 0.023366
grad_step = 000073, loss = 0.021937
grad_step = 000074, loss = 0.020598
grad_step = 000075, loss = 0.019326
grad_step = 000076, loss = 0.018130
grad_step = 000077, loss = 0.017006
grad_step = 000078, loss = 0.015943
grad_step = 000079, loss = 0.014946
grad_step = 000080, loss = 0.014006
grad_step = 000081, loss = 0.013123
grad_step = 000082, loss = 0.012293
grad_step = 000083, loss = 0.011516
grad_step = 000084, loss = 0.010786
grad_step = 000085, loss = 0.010104
grad_step = 000086, loss = 0.009466
grad_step = 000087, loss = 0.008870
grad_step = 000088, loss = 0.008315
grad_step = 000089, loss = 0.007798
grad_step = 000090, loss = 0.007316
grad_step = 000091, loss = 0.006869
grad_step = 000092, loss = 0.006454
grad_step = 000093, loss = 0.006069
grad_step = 000094, loss = 0.005713
grad_step = 000095, loss = 0.005385
grad_step = 000096, loss = 0.005082
grad_step = 000097, loss = 0.004804
grad_step = 000098, loss = 0.004547
grad_step = 000099, loss = 0.004313
grad_step = 000100, loss = 0.004099
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003902
grad_step = 000102, loss = 0.003723
grad_step = 000103, loss = 0.003560
grad_step = 000104, loss = 0.003411
grad_step = 000105, loss = 0.003277
grad_step = 000106, loss = 0.003155
grad_step = 000107, loss = 0.003045
grad_step = 000108, loss = 0.002946
grad_step = 000109, loss = 0.002857
grad_step = 000110, loss = 0.002777
grad_step = 000111, loss = 0.002705
grad_step = 000112, loss = 0.002640
grad_step = 000113, loss = 0.002583
grad_step = 000114, loss = 0.002531
grad_step = 000115, loss = 0.002486
grad_step = 000116, loss = 0.002446
grad_step = 000117, loss = 0.002411
grad_step = 000118, loss = 0.002379
grad_step = 000119, loss = 0.002348
grad_step = 000120, loss = 0.002324
grad_step = 000121, loss = 0.002303
grad_step = 000122, loss = 0.002282
grad_step = 000123, loss = 0.002263
grad_step = 000124, loss = 0.002249
grad_step = 000125, loss = 0.002236
grad_step = 000126, loss = 0.002223
grad_step = 000127, loss = 0.002211
grad_step = 000128, loss = 0.002203
grad_step = 000129, loss = 0.002197
grad_step = 000130, loss = 0.002196
grad_step = 000131, loss = 0.002204
grad_step = 000132, loss = 0.002219
grad_step = 000133, loss = 0.002240
grad_step = 000134, loss = 0.002218
grad_step = 000135, loss = 0.002175
grad_step = 000136, loss = 0.002160
grad_step = 000137, loss = 0.002185
grad_step = 000138, loss = 0.002202
grad_step = 000139, loss = 0.002184
grad_step = 000140, loss = 0.002169
grad_step = 000141, loss = 0.002183
grad_step = 000142, loss = 0.002192
grad_step = 000143, loss = 0.002176
grad_step = 000144, loss = 0.002137
grad_step = 000145, loss = 0.002129
grad_step = 000146, loss = 0.002140
grad_step = 000147, loss = 0.002136
grad_step = 000148, loss = 0.002130
grad_step = 000149, loss = 0.002131
grad_step = 000150, loss = 0.002134
grad_step = 000151, loss = 0.002126
grad_step = 000152, loss = 0.002110
grad_step = 000153, loss = 0.002100
grad_step = 000154, loss = 0.002104
grad_step = 000155, loss = 0.002111
grad_step = 000156, loss = 0.002108
grad_step = 000157, loss = 0.002101
grad_step = 000158, loss = 0.002098
grad_step = 000159, loss = 0.002097
grad_step = 000160, loss = 0.002092
grad_step = 000161, loss = 0.002086
grad_step = 000162, loss = 0.002081
grad_step = 000163, loss = 0.002077
grad_step = 000164, loss = 0.002076
grad_step = 000165, loss = 0.002077
grad_step = 000166, loss = 0.002079
grad_step = 000167, loss = 0.002078
grad_step = 000168, loss = 0.002077
grad_step = 000169, loss = 0.002077
grad_step = 000170, loss = 0.002084
grad_step = 000171, loss = 0.002099
grad_step = 000172, loss = 0.002130
grad_step = 000173, loss = 0.002156
grad_step = 000174, loss = 0.002180
grad_step = 000175, loss = 0.002139
grad_step = 000176, loss = 0.002082
grad_step = 000177, loss = 0.002049
grad_step = 000178, loss = 0.002075
grad_step = 000179, loss = 0.002118
grad_step = 000180, loss = 0.002123
grad_step = 000181, loss = 0.002108
grad_step = 000182, loss = 0.002116
grad_step = 000183, loss = 0.002119
grad_step = 000184, loss = 0.002118
grad_step = 000185, loss = 0.002076
grad_step = 000186, loss = 0.002048
grad_step = 000187, loss = 0.002056
grad_step = 000188, loss = 0.002076
grad_step = 000189, loss = 0.002080
grad_step = 000190, loss = 0.002045
grad_step = 000191, loss = 0.002028
grad_step = 000192, loss = 0.002041
grad_step = 000193, loss = 0.002054
grad_step = 000194, loss = 0.002049
grad_step = 000195, loss = 0.002025
grad_step = 000196, loss = 0.002015
grad_step = 000197, loss = 0.002023
grad_step = 000198, loss = 0.002031
grad_step = 000199, loss = 0.002027
grad_step = 000200, loss = 0.002012
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002003
grad_step = 000202, loss = 0.002007
grad_step = 000203, loss = 0.002013
grad_step = 000204, loss = 0.002011
grad_step = 000205, loss = 0.002001
grad_step = 000206, loss = 0.001993
grad_step = 000207, loss = 0.001991
grad_step = 000208, loss = 0.001993
grad_step = 000209, loss = 0.001995
grad_step = 000210, loss = 0.001993
grad_step = 000211, loss = 0.001990
grad_step = 000212, loss = 0.001984
grad_step = 000213, loss = 0.001978
grad_step = 000214, loss = 0.001974
grad_step = 000215, loss = 0.001972
grad_step = 000216, loss = 0.001972
grad_step = 000217, loss = 0.001973
grad_step = 000218, loss = 0.001973
grad_step = 000219, loss = 0.001974
grad_step = 000220, loss = 0.001976
grad_step = 000221, loss = 0.001979
grad_step = 000222, loss = 0.001986
grad_step = 000223, loss = 0.001992
grad_step = 000224, loss = 0.002005
grad_step = 000225, loss = 0.002004
grad_step = 000226, loss = 0.002001
grad_step = 000227, loss = 0.001975
grad_step = 000228, loss = 0.001953
grad_step = 000229, loss = 0.001943
grad_step = 000230, loss = 0.001949
grad_step = 000231, loss = 0.001961
grad_step = 000232, loss = 0.001965
grad_step = 000233, loss = 0.001964
grad_step = 000234, loss = 0.001957
grad_step = 000235, loss = 0.001958
grad_step = 000236, loss = 0.001976
grad_step = 000237, loss = 0.001999
grad_step = 000238, loss = 0.002042
grad_step = 000239, loss = 0.002045
grad_step = 000240, loss = 0.002046
grad_step = 000241, loss = 0.001983
grad_step = 000242, loss = 0.001938
grad_step = 000243, loss = 0.001930
grad_step = 000244, loss = 0.001961
grad_step = 000245, loss = 0.001993
grad_step = 000246, loss = 0.001968
grad_step = 000247, loss = 0.001923
grad_step = 000248, loss = 0.001898
grad_step = 000249, loss = 0.001908
grad_step = 000250, loss = 0.001923
grad_step = 000251, loss = 0.001911
grad_step = 000252, loss = 0.001907
grad_step = 000253, loss = 0.001928
grad_step = 000254, loss = 0.001945
grad_step = 000255, loss = 0.001906
grad_step = 000256, loss = 0.001866
grad_step = 000257, loss = 0.001878
grad_step = 000258, loss = 0.001915
grad_step = 000259, loss = 0.001923
grad_step = 000260, loss = 0.001933
grad_step = 000261, loss = 0.001995
grad_step = 000262, loss = 0.002142
grad_step = 000263, loss = 0.002137
grad_step = 000264, loss = 0.002038
grad_step = 000265, loss = 0.001868
grad_step = 000266, loss = 0.001884
grad_step = 000267, loss = 0.001984
grad_step = 000268, loss = 0.001931
grad_step = 000269, loss = 0.001855
grad_step = 000270, loss = 0.001878
grad_step = 000271, loss = 0.001908
grad_step = 000272, loss = 0.001868
grad_step = 000273, loss = 0.001822
grad_step = 000274, loss = 0.001858
grad_step = 000275, loss = 0.001893
grad_step = 000276, loss = 0.001844
grad_step = 000277, loss = 0.001809
grad_step = 000278, loss = 0.001826
grad_step = 000279, loss = 0.001845
grad_step = 000280, loss = 0.001831
grad_step = 000281, loss = 0.001800
grad_step = 000282, loss = 0.001801
grad_step = 000283, loss = 0.001822
grad_step = 000284, loss = 0.001816
grad_step = 000285, loss = 0.001791
grad_step = 000286, loss = 0.001778
grad_step = 000287, loss = 0.001788
grad_step = 000288, loss = 0.001799
grad_step = 000289, loss = 0.001792
grad_step = 000290, loss = 0.001780
grad_step = 000291, loss = 0.001772
grad_step = 000292, loss = 0.001770
grad_step = 000293, loss = 0.001772
grad_step = 000294, loss = 0.001778
grad_step = 000295, loss = 0.001787
grad_step = 000296, loss = 0.001788
grad_step = 000297, loss = 0.001792
grad_step = 000298, loss = 0.001802
grad_step = 000299, loss = 0.001836
grad_step = 000300, loss = 0.001875
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001941
grad_step = 000302, loss = 0.001959
grad_step = 000303, loss = 0.001942
grad_step = 000304, loss = 0.001888
grad_step = 000305, loss = 0.001840
grad_step = 000306, loss = 0.001844
grad_step = 000307, loss = 0.001834
grad_step = 000308, loss = 0.001834
grad_step = 000309, loss = 0.001788
grad_step = 000310, loss = 0.001754
grad_step = 000311, loss = 0.001767
grad_step = 000312, loss = 0.001796
grad_step = 000313, loss = 0.001804
grad_step = 000314, loss = 0.001756
grad_step = 000315, loss = 0.001734
grad_step = 000316, loss = 0.001749
grad_step = 000317, loss = 0.001760
grad_step = 000318, loss = 0.001764
grad_step = 000319, loss = 0.001745
grad_step = 000320, loss = 0.001725
grad_step = 000321, loss = 0.001718
grad_step = 000322, loss = 0.001732
grad_step = 000323, loss = 0.001749
grad_step = 000324, loss = 0.001738
grad_step = 000325, loss = 0.001720
grad_step = 000326, loss = 0.001711
grad_step = 000327, loss = 0.001710
grad_step = 000328, loss = 0.001710
grad_step = 000329, loss = 0.001711
grad_step = 000330, loss = 0.001715
grad_step = 000331, loss = 0.001713
grad_step = 000332, loss = 0.001705
grad_step = 000333, loss = 0.001700
grad_step = 000334, loss = 0.001699
grad_step = 000335, loss = 0.001697
grad_step = 000336, loss = 0.001695
grad_step = 000337, loss = 0.001695
grad_step = 000338, loss = 0.001698
grad_step = 000339, loss = 0.001701
grad_step = 000340, loss = 0.001704
grad_step = 000341, loss = 0.001714
grad_step = 000342, loss = 0.001730
grad_step = 000343, loss = 0.001760
grad_step = 000344, loss = 0.001792
grad_step = 000345, loss = 0.001847
grad_step = 000346, loss = 0.001856
grad_step = 000347, loss = 0.001853
grad_step = 000348, loss = 0.001773
grad_step = 000349, loss = 0.001727
grad_step = 000350, loss = 0.001730
grad_step = 000351, loss = 0.001782
grad_step = 000352, loss = 0.001820
grad_step = 000353, loss = 0.001787
grad_step = 000354, loss = 0.001721
grad_step = 000355, loss = 0.001677
grad_step = 000356, loss = 0.001680
grad_step = 000357, loss = 0.001706
grad_step = 000358, loss = 0.001705
grad_step = 000359, loss = 0.001685
grad_step = 000360, loss = 0.001671
grad_step = 000361, loss = 0.001679
grad_step = 000362, loss = 0.001699
grad_step = 000363, loss = 0.001696
grad_step = 000364, loss = 0.001678
grad_step = 000365, loss = 0.001653
grad_step = 000366, loss = 0.001643
grad_step = 000367, loss = 0.001649
grad_step = 000368, loss = 0.001658
grad_step = 000369, loss = 0.001661
grad_step = 000370, loss = 0.001654
grad_step = 000371, loss = 0.001647
grad_step = 000372, loss = 0.001644
grad_step = 000373, loss = 0.001646
grad_step = 000374, loss = 0.001648
grad_step = 000375, loss = 0.001647
grad_step = 000376, loss = 0.001642
grad_step = 000377, loss = 0.001634
grad_step = 000378, loss = 0.001626
grad_step = 000379, loss = 0.001621
grad_step = 000380, loss = 0.001619
grad_step = 000381, loss = 0.001618
grad_step = 000382, loss = 0.001620
grad_step = 000383, loss = 0.001622
grad_step = 000384, loss = 0.001625
grad_step = 000385, loss = 0.001628
grad_step = 000386, loss = 0.001633
grad_step = 000387, loss = 0.001639
grad_step = 000388, loss = 0.001649
grad_step = 000389, loss = 0.001667
grad_step = 000390, loss = 0.001694
grad_step = 000391, loss = 0.001750
grad_step = 000392, loss = 0.001825
grad_step = 000393, loss = 0.001955
grad_step = 000394, loss = 0.001989
grad_step = 000395, loss = 0.001961
grad_step = 000396, loss = 0.001790
grad_step = 000397, loss = 0.001840
grad_step = 000398, loss = 0.002025
grad_step = 000399, loss = 0.002159
grad_step = 000400, loss = 0.001871
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001631
grad_step = 000402, loss = 0.001690
grad_step = 000403, loss = 0.001842
grad_step = 000404, loss = 0.001800
grad_step = 000405, loss = 0.001648
grad_step = 000406, loss = 0.001679
grad_step = 000407, loss = 0.001735
grad_step = 000408, loss = 0.001674
grad_step = 000409, loss = 0.001639
grad_step = 000410, loss = 0.001656
grad_step = 000411, loss = 0.001657
grad_step = 000412, loss = 0.001638
grad_step = 000413, loss = 0.001628
grad_step = 000414, loss = 0.001629
grad_step = 000415, loss = 0.001615
grad_step = 000416, loss = 0.001613
grad_step = 000417, loss = 0.001619
grad_step = 000418, loss = 0.001606
grad_step = 000419, loss = 0.001597
grad_step = 000420, loss = 0.001596
grad_step = 000421, loss = 0.001598
grad_step = 000422, loss = 0.001601
grad_step = 000423, loss = 0.001583
grad_step = 000424, loss = 0.001570
grad_step = 000425, loss = 0.001582
grad_step = 000426, loss = 0.001590
grad_step = 000427, loss = 0.001576
grad_step = 000428, loss = 0.001559
grad_step = 000429, loss = 0.001566
grad_step = 000430, loss = 0.001573
grad_step = 000431, loss = 0.001566
grad_step = 000432, loss = 0.001560
grad_step = 000433, loss = 0.001559
grad_step = 000434, loss = 0.001556
grad_step = 000435, loss = 0.001552
grad_step = 000436, loss = 0.001552
grad_step = 000437, loss = 0.001554
grad_step = 000438, loss = 0.001551
grad_step = 000439, loss = 0.001546
grad_step = 000440, loss = 0.001542
grad_step = 000441, loss = 0.001540
grad_step = 000442, loss = 0.001541
grad_step = 000443, loss = 0.001542
grad_step = 000444, loss = 0.001539
grad_step = 000445, loss = 0.001535
grad_step = 000446, loss = 0.001534
grad_step = 000447, loss = 0.001533
grad_step = 000448, loss = 0.001531
grad_step = 000449, loss = 0.001528
grad_step = 000450, loss = 0.001526
grad_step = 000451, loss = 0.001524
grad_step = 000452, loss = 0.001523
grad_step = 000453, loss = 0.001522
grad_step = 000454, loss = 0.001520
grad_step = 000455, loss = 0.001518
grad_step = 000456, loss = 0.001516
grad_step = 000457, loss = 0.001515
grad_step = 000458, loss = 0.001515
grad_step = 000459, loss = 0.001515
grad_step = 000460, loss = 0.001516
grad_step = 000461, loss = 0.001522
grad_step = 000462, loss = 0.001538
grad_step = 000463, loss = 0.001583
grad_step = 000464, loss = 0.001689
grad_step = 000465, loss = 0.001948
grad_step = 000466, loss = 0.002172
grad_step = 000467, loss = 0.002335
grad_step = 000468, loss = 0.001748
grad_step = 000469, loss = 0.001561
grad_step = 000470, loss = 0.001844
grad_step = 000471, loss = 0.001709
grad_step = 000472, loss = 0.001555
grad_step = 000473, loss = 0.001728
grad_step = 000474, loss = 0.001635
grad_step = 000475, loss = 0.001521
grad_step = 000476, loss = 0.001640
grad_step = 000477, loss = 0.001584
grad_step = 000478, loss = 0.001539
grad_step = 000479, loss = 0.001596
grad_step = 000480, loss = 0.001527
grad_step = 000481, loss = 0.001531
grad_step = 000482, loss = 0.001584
grad_step = 000483, loss = 0.001521
grad_step = 000484, loss = 0.001523
grad_step = 000485, loss = 0.001542
grad_step = 000486, loss = 0.001502
grad_step = 000487, loss = 0.001527
grad_step = 000488, loss = 0.001532
grad_step = 000489, loss = 0.001487
grad_step = 000490, loss = 0.001508
grad_step = 000491, loss = 0.001516
grad_step = 000492, loss = 0.001487
grad_step = 000493, loss = 0.001503
grad_step = 000494, loss = 0.001504
grad_step = 000495, loss = 0.001476
grad_step = 000496, loss = 0.001489
grad_step = 000497, loss = 0.001495
grad_step = 000498, loss = 0.001476
grad_step = 000499, loss = 0.001483
grad_step = 000500, loss = 0.001484
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001468
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

  date_run                              2020-05-10 04:13:21.480800
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.247621
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 04:13:21.485811
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.156189
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 04:13:21.491858
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149577
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 04:13:21.497842
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.37334
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

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json 

                        date_run  ...            metric_name
0   2020-05-10 04:12:48.528604  ...    mean_absolute_error
1   2020-05-10 04:12:48.534012  ...     mean_squared_error
2   2020-05-10 04:12:48.537118  ...  median_absolute_error
3   2020-05-10 04:12:48.540654  ...               r2_score
4   2020-05-10 04:12:58.923138  ...    mean_absolute_error
5   2020-05-10 04:12:58.926839  ...     mean_squared_error
6   2020-05-10 04:12:58.929753  ...  median_absolute_error
7   2020-05-10 04:12:58.933076  ...               r2_score
8   2020-05-10 04:13:21.480800  ...    mean_absolute_error
9   2020-05-10 04:13:21.485811  ...     mean_squared_error
10  2020-05-10 04:13:21.491858  ...  median_absolute_error
11  2020-05-10 04:13:21.497842  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 152199.57it/s] 74%|███████▍  | 7315456/9912422 [00:00<00:11, 217232.96it/s]9920512it [00:00, 43276165.00it/s]                           
0it [00:00, ?it/s]32768it [00:00, 579546.84it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148760.52it/s]1654784it [00:00, 10758122.30it/s]                         
0it [00:00, ?it/s]8192it [00:00, 199537.38it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71f6da0128> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71934009b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71f5c73e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7193400da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71f5cbcf60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71f5cbc6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7193402080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71a867fb70> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7193400da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f71a867fb70> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7193402048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6ceb0d6208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f4cf5355245f127a9836ba2e21084d60241bbc6e71d6783d8b468ab775d3f971
  Stored in directory: /tmp/pip-ephem-wheel-cache-7_ixc1m4/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6ce1245048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2334720/17464789 [===>..........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
15867904/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 04:14:46.646660: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 04:14:46.650979: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-10 04:14:46.651103: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a6dabc0260 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 04:14:46.651116: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 10s - loss: 7.3830 - accuracy: 0.5185
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6308 - accuracy: 0.5023 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5018 - accuracy: 0.5107
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5194 - accuracy: 0.5096
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5542 - accuracy: 0.5073
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6343 - accuracy: 0.5021
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6912 - accuracy: 0.4984
11000/25000 [============>.................] - ETA: 4s - loss: 7.7224 - accuracy: 0.4964
12000/25000 [=============>................] - ETA: 4s - loss: 7.6998 - accuracy: 0.4978
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7067 - accuracy: 0.4974
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7050 - accuracy: 0.4975
15000/25000 [=================>............] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6684 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6421 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6549 - accuracy: 0.5008
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 377us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 04:15:02.975027
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 04:15:02.975027  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 04:15:08.863349: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 04:15:08.867798: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-10 04:15:08.867933: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b730a076e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 04:15:08.867946: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fa78dfa9d30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 986ms/step - loss: 1.6423 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.6514 - val_crf_viterbi_accuracy: 0.3867

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa78540d6a0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4520 - accuracy: 0.5140
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4673 - accuracy: 0.5130 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6002 - accuracy: 0.5043
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6283 - accuracy: 0.5025
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6850 - accuracy: 0.4988
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6455 - accuracy: 0.5014
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6649 - accuracy: 0.5001
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6605 - accuracy: 0.5004
11000/25000 [============>.................] - ETA: 4s - loss: 7.6485 - accuracy: 0.5012
12000/25000 [=============>................] - ETA: 4s - loss: 7.6321 - accuracy: 0.5023
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6383 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6338 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6298 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6071 - accuracy: 0.5039
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6291 - accuracy: 0.5024
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6295 - accuracy: 0.5024
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6352 - accuracy: 0.5020
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6313 - accuracy: 0.5023
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6455 - accuracy: 0.5014
25000/25000 [==============================] - 9s 369us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fa73f109048> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

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
NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_classifier.py", line 39, in <module>
    from util_transformer import (convert_examples_to_features, output_modes,
ModuleNotFoundError: No module named 'util_transformer'

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
NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
    output_path      = out_pars["model_path"]
KeyError: 'model_path'
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<95:39:51, 2.50kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<67:11:35, 3.56kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<47:04:52, 5.09kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<32:56:41, 7.26kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<22:59:29, 10.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.21M/862M [00:03<15:59:29, 14.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:04<11:07:14, 21.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:04<7:44:00, 30.2kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:04<5:22:42, 43.2kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.4M/862M [00:04<3:44:37, 61.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:04<2:36:39, 88.0kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:04<1:49:03, 126kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:04<1:16:09, 179kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:04<53:12, 255kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:05<37:22, 361kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.0M/862M [00:07<27:56, 481kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:07<23:11, 579kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:07<17:08, 783kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:07<12:11, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:09<14:16, 937kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:09<11:36, 1.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:09<08:31, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:11<08:40, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:11<08:53, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:11<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:11<04:58, 2.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:13<35:25, 373kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:13<26:07, 506kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:13<18:35, 710kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:15<16:04, 819kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:15<12:33, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:15<09:06, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:17<09:27, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:17<07:56, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:17<05:53, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:19<07:11, 1.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:19<07:40, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:19<05:56, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:19<04:17, 3.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:21<20:47, 623kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:21<15:51, 816kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:21<11:21, 1.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:22<10:58, 1.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:23<10:24, 1.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:23<07:49, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.1M/862M [00:23<05:38, 2.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:24<10:01, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:25<08:20, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:25<06:09, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:26<07:15, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:27<07:48, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:27<06:06, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<04:25, 2.86MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<1:33:44, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<1:06:54, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<47:03, 269kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<35:48, 352kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<27:44, 454kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<20:03, 628kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<14:08, 887kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<1:39:30, 126kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<1:10:43, 177kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<49:44, 252kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<37:36, 332kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<28:51, 432kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<20:43, 601kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:35<14:37, 850kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<17:51, 695kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<13:46, 900kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<09:56, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<09:50, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<09:24, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<07:07, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:39<05:07, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<11:49, 1.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<09:34, 1.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<07:00, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<07:44, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<08:00, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<06:08, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:42<04:25, 2.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<11:57, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<09:37, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<07:02, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<07:44, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<07:52, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:06, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<06:12, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:33, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<04:11, 2.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:43, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:00, 2.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<03:45, 3.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<02:48, 4.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:52<1:30:22, 131kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<1:05:38, 181kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<46:27, 255kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<34:18, 344kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<25:11, 468kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<17:53, 657kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<15:14, 769kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<13:02, 898kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<09:42, 1.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<08:39, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<07:15, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<05:21, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<06:27, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<06:57, 1.66MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<05:28, 2.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<03:57, 2.92MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<1:09:11, 166kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<49:35, 232kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<34:55, 329kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<27:04, 423kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<21:19, 537kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<15:29, 738kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<10:54, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<1:21:09, 140kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<57:57, 196kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<40:46, 278kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<31:04, 364kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<24:08, 468kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<17:28, 646kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<12:18, 913kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<34:35, 325kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<25:13, 445kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<17:55, 625kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<14:59, 745kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<12:51, 869kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<09:34, 1.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<06:48, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<31:32, 352kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<23:13, 478kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<16:30, 671kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<14:03, 785kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<12:11, 905kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<09:06, 1.21MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:28, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<30:51, 355kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<22:43, 482kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<16:09, 677kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<13:47, 791kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<11:58, 910kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<08:51, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:20<06:20, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<08:53, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<07:21, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<05:25, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<06:15, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<05:29, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<04:07, 2.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<05:21, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<06:01, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<04:46, 2.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<03:26, 3.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<26:58, 394kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<19:59, 531kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<14:14, 744kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<12:22, 853kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<10:54, 967kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<08:05, 1.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<05:52, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<07:04, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<05:51, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<04:18, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<03:11, 3.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:33<29:12, 357kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:33<22:38, 460kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<16:18, 638kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<11:30, 900kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<13:09, 787kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<10:16, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<07:24, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<07:32, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:37<07:29, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<05:46, 1.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:09, 2.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<33:35, 304kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<24:35, 415kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<17:26, 584kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<14:29, 700kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<12:18, 823kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<09:08, 1.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:30, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<36:30, 276kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<26:34, 379kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<18:49, 533kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<15:26, 648kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<12:50, 779kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<09:25, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<06:42, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:47<09:10, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:47<08:31, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<06:24, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<04:36, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<07:42, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<06:26, 1.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<04:44, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<05:33, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<05:58, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<04:39, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<03:21, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<08:26, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<06:55, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:53<05:02, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<05:45, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<06:03, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<04:42, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<03:22, 2.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<14:22, 667kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<11:02, 868kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:57<07:57, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:59<07:44, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<06:24, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<04:40, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<05:28, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<04:48, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<03:34, 2.64MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<04:39, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<05:15, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<04:09, 2.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<03:00, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:05<1:08:43, 136kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<49:03, 190kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<34:29, 269kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<26:10, 353kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<20:16, 456kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<14:39, 630kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<10:19, 890kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<1:12:50, 126kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<51:55, 177kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<36:29, 251kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<27:33, 331kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<21:12, 429kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<15:14, 597kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<10:44, 843kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<12:00, 753kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<09:20, 967kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<06:45, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<06:47, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<06:23, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<04:57, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<03:36, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<05:29, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<04:45, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<03:33, 2.50MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<05:02, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<03:57, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<02:52, 3.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<22:53, 383kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<16:55, 518kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<12:02, 725kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<10:24, 836kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<09:03, 960kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<06:44, 1.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:24<30:06, 287kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<21:57, 393kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<15:31, 554kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:26<12:49, 668kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<10:47, 794kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<07:59, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<05:39, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<24:25, 348kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<17:57, 473kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<12:43, 665kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<10:50, 777kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<09:22, 899kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:30<06:58, 1.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<04:57, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<1:03:12, 132kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<45:04, 185kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<31:39, 263kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<23:57, 346kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<18:32, 447kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<13:22, 618kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<09:24, 873kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<1:05:17, 126kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<46:32, 176kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<32:41, 250kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<24:39, 331kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<18:58, 429kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<13:38, 596kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<09:37, 841kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:40<10:01, 807kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<07:51, 1.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<05:41, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:42<05:49, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<04:27, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:12, 2.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:44<27:24, 290kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<20:00, 397kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<14:08, 560kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<11:40, 675kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<09:50, 800kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<07:14, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<05:08, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<08:01, 972kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<06:25, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<04:39, 1.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<05:03, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<05:10, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:50<03:59, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<02:52, 2.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<06:43, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<05:22, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<03:57, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<04:30, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<04:45, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<03:43, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<02:41, 2.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<15:36, 483kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<11:41, 643kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<08:21, 897kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<07:34, 986kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<06:05, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<04:25, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:00<04:46, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:54, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<03:48, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<02:44, 2.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<54:17, 135kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<38:44, 189kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<27:12, 268kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:03<20:37, 352kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<15:58, 454kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<11:32, 628kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<08:06, 886kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<57:02, 126kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:06<40:38, 177kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<28:32, 251kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<21:31, 331kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<16:34, 430kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<11:53, 597kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<08:22, 844kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<09:35, 736kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<07:26, 947kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<05:22, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<05:21, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<04:29, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<03:18, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<03:55, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<04:12, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:18, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<02:23, 2.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<23:06, 296kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<16:52, 406kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<11:56, 571kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<09:53, 685kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<08:18, 816kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<06:09, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<04:21, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<50:59, 132kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<36:15, 185kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:19<25:27, 262kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<19:15, 345kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<14:08, 469kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<10:01, 659kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<08:31, 771kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<07:21, 893kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<05:29, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:53, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<18:22, 354kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<13:31, 480kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<09:35, 674kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<08:11, 786kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<07:05, 908kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<05:14, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<03:44, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<05:12, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<04:18, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<03:10, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<03:40, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<03:51, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<02:57, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:08, 2.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<05:13, 1.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<04:18, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<03:09, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<03:37, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<03:47, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<02:56, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<02:07, 2.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<05:22, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<05:06, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:50, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:45, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<04:43, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:55, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<02:53, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:37, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<02:50, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:03, 2.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<19:09, 307kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<14:00, 420kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<09:54, 591kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:45<08:14, 706kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:45<07:00, 830kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<05:08, 1.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<03:40, 1.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<04:33, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:47<03:46, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<02:46, 2.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<03:15, 1.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<03:28, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:49<02:43, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<01:57, 2.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<14:51, 378kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<10:59, 510kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<07:48, 716kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<06:42, 827kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<05:52, 945kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:53<04:22, 1.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<03:07, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<04:33, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<03:45, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<02:45, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<03:22, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:57<02:36, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<01:53, 2.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<03:29, 1.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<03:00, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<02:13, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<02:46, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<03:03, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:24, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<01:44, 3.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<10:47, 482kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<08:04, 644kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<05:45, 899kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<05:12, 986kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<04:43, 1.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<03:32, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<02:31, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<04:33, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<03:42, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<02:43, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<03:04, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<02:39, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<01:59, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<02:30, 1.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<02:47, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<02:10, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<01:34, 3.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<04:05, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<03:22, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<02:27, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<02:48, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<02:28, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<01:50, 2.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<02:22, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<02:09, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<01:36, 2.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<02:10, 2.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<02:00, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<01:30, 3.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<02:27, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<01:57, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:25, 3.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<14:34, 310kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<10:39, 423kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<07:31, 595kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<06:15, 710kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<04:50, 918kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<03:28, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:26<03:26, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<03:20, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<02:31, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:49, 2.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:46, 1.55MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:24, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:45, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:30<02:12, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<01:55, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:26, 2.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:58, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:48, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:32<01:21, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<01:54, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<02:12, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:45, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:15, 3.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<10:13, 395kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<07:35, 531kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<05:22, 746kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<04:36, 860kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<04:02, 983kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<03:01, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:08, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<36:03, 108kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<26:02, 150kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<18:20, 212kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<12:48, 301kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<10:04, 380kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<07:27, 513kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<05:16, 719kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<04:32, 829kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<03:58, 947kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<02:56, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<02:05, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<03:07, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<02:33, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<01:52, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<02:08, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<01:52, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:48<01:23, 2.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:47, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<02:00, 1.77MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<01:33, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<01:07, 3.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<02:31, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<02:08, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<01:33, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:52, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<02:02, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<01:34, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<01:08, 2.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<02:00, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<01:45, 1.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<01:17, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:39, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:51, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<01:27, 2.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:02, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<19:15, 167kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<13:47, 233kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<09:39, 329kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<07:24, 424kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<05:30, 570kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<03:53, 799kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<03:24, 903kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<03:02, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<02:15, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<01:36, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<02:25, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<02:00, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:28, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:41, 1.74MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:29, 1.98MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:06, 2.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<01:26, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<01:35, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:14, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<00:52, 3.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:11<03:40, 761kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<02:51, 977kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<02:03, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<02:03, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:06, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<08:46, 304kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:15<06:21, 418kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<04:30, 587kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:08, 829kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<07:05, 367kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<05:30, 471kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<03:57, 653kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<02:46, 919kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<02:50, 888kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<02:15, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:37, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<01:40, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<01:41, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<01:17, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<00:55, 2.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<02:02, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<01:40, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<01:13, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:22, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:26, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<01:06, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:47, 2.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:44, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:26, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<01:03, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:14, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:19, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:02, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:44, 2.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<05:22, 393kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<03:58, 531kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:31<02:48, 743kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<02:24, 852kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<02:06, 968kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<01:33, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:05, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:35<02:14, 883kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<01:45, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<01:15, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<01:18, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<01:19, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<01:00, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:42, 2.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<01:39, 1.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<01:21, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<00:58, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<01:05, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<01:07, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<00:52, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:37, 2.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<04:22, 389kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<03:13, 525kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<02:16, 738kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<01:55, 849kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<01:30, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:45<01:04, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<01:06, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<01:06, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:50, 1.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:35, 2.53MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<11:07, 135kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<07:54, 189kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:49<05:28, 268kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<04:03, 352kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<03:08, 454kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<02:14, 630kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<01:33, 888kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:40, 812kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:18, 1.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:55, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:55, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:46, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<00:33, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:40, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:34, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<00:23, 2.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<01:57, 590kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<01:28, 775kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<01:02, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:57, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:53, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:40, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:28, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:46, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:38, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:27, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:31, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:28, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:20, 2.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:25, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:29, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:22, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:15, 3.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<02:13, 364kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<01:38, 492kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<01:10, 676kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:49, 945kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:33, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<21:52, 34.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<15:18, 48.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:10<10:26, 68.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<06:59, 96.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<05:00, 134kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<03:28, 189kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<02:15, 269kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<05:53, 103kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<04:09, 144kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<02:47, 205kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<01:57, 275kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<01:28, 362kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<01:02, 504kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:39, 713kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<01:56, 241kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<01:23, 332kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:55, 469kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:41, 580kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:30, 765kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:20, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:17, 1.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:16, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:12, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:55, 286kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:39, 392kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:25, 552kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:17, 667kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:14, 793kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:26<00:10, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:05, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:26, 287kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:18, 394kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:10, 554kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:05, 670kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:04, 796kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:02, 1.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 1.51MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 854/400000 [00:00<00:46, 8532.64it/s]  0%|          | 1651/400000 [00:00<00:47, 8347.64it/s]  1%|          | 2408/400000 [00:00<00:49, 8094.64it/s]  1%|          | 3305/400000 [00:00<00:47, 8335.82it/s]  1%|          | 4201/400000 [00:00<00:46, 8513.56it/s]  1%|▏         | 5046/400000 [00:00<00:46, 8492.68it/s]  1%|▏         | 5976/400000 [00:00<00:45, 8717.58it/s]  2%|▏         | 6781/400000 [00:00<00:47, 8357.63it/s]  2%|▏         | 7573/400000 [00:00<00:47, 8197.97it/s]  2%|▏         | 8393/400000 [00:01<00:47, 8196.77it/s]  2%|▏         | 9192/400000 [00:01<00:48, 8101.39it/s]  3%|▎         | 10028/400000 [00:01<00:47, 8175.81it/s]  3%|▎         | 10836/400000 [00:01<00:48, 8068.01it/s]  3%|▎         | 11703/400000 [00:01<00:47, 8239.26it/s]  3%|▎         | 12571/400000 [00:01<00:46, 8364.26it/s]  3%|▎         | 13458/400000 [00:01<00:45, 8508.21it/s]  4%|▎         | 14338/400000 [00:01<00:44, 8590.18it/s]  4%|▍         | 15197/400000 [00:01<00:45, 8443.46it/s]  4%|▍         | 16042/400000 [00:01<00:45, 8426.87it/s]  4%|▍         | 16890/400000 [00:02<00:45, 8441.27it/s]  4%|▍         | 17735/400000 [00:02<00:45, 8348.26it/s]  5%|▍         | 18583/400000 [00:02<00:45, 8385.05it/s]  5%|▍         | 19422/400000 [00:02<00:45, 8345.16it/s]  5%|▌         | 20313/400000 [00:02<00:44, 8505.45it/s]  5%|▌         | 21218/400000 [00:02<00:43, 8659.29it/s]  6%|▌         | 22086/400000 [00:02<00:44, 8411.33it/s]  6%|▌         | 22930/400000 [00:02<00:45, 8259.85it/s]  6%|▌         | 23764/400000 [00:02<00:45, 8283.34it/s]  6%|▌         | 24648/400000 [00:02<00:44, 8441.12it/s]  6%|▋         | 25495/400000 [00:03<00:44, 8394.84it/s]  7%|▋         | 26336/400000 [00:03<00:44, 8371.46it/s]  7%|▋         | 27175/400000 [00:03<00:44, 8355.99it/s]  7%|▋         | 28012/400000 [00:03<00:45, 8134.46it/s]  7%|▋         | 28834/400000 [00:03<00:45, 8157.71it/s]  7%|▋         | 29708/400000 [00:03<00:44, 8323.65it/s]  8%|▊         | 30545/400000 [00:03<00:44, 8337.00it/s]  8%|▊         | 31380/400000 [00:03<00:44, 8325.45it/s]  8%|▊         | 32214/400000 [00:03<00:46, 7965.95it/s]  8%|▊         | 33049/400000 [00:03<00:45, 8074.88it/s]  8%|▊         | 33860/400000 [00:04<00:45, 7962.88it/s]  9%|▊         | 34659/400000 [00:04<00:46, 7885.25it/s]  9%|▉         | 35450/400000 [00:04<00:46, 7885.84it/s]  9%|▉         | 36252/400000 [00:04<00:45, 7923.06it/s]  9%|▉         | 37112/400000 [00:04<00:44, 8112.17it/s]  9%|▉         | 37972/400000 [00:04<00:43, 8249.62it/s] 10%|▉         | 38855/400000 [00:04<00:42, 8414.92it/s] 10%|▉         | 39699/400000 [00:04<00:42, 8408.42it/s] 10%|█         | 40542/400000 [00:04<00:43, 8308.39it/s] 10%|█         | 41375/400000 [00:04<00:44, 8133.51it/s] 11%|█         | 42199/400000 [00:05<00:43, 8163.56it/s] 11%|█         | 43026/400000 [00:05<00:43, 8192.97it/s] 11%|█         | 43906/400000 [00:05<00:42, 8364.20it/s] 11%|█         | 44744/400000 [00:05<00:43, 8242.58it/s] 11%|█▏        | 45648/400000 [00:05<00:41, 8466.06it/s] 12%|█▏        | 46498/400000 [00:05<00:43, 8172.21it/s] 12%|█▏        | 47320/400000 [00:05<00:43, 8178.04it/s] 12%|█▏        | 48168/400000 [00:05<00:42, 8265.96it/s] 12%|█▏        | 48997/400000 [00:05<00:42, 8262.87it/s] 12%|█▏        | 49825/400000 [00:06<00:42, 8143.87it/s] 13%|█▎        | 50754/400000 [00:06<00:41, 8454.71it/s] 13%|█▎        | 51604/400000 [00:06<00:41, 8343.33it/s] 13%|█▎        | 52469/400000 [00:06<00:41, 8432.09it/s] 13%|█▎        | 53315/400000 [00:06<00:41, 8386.43it/s] 14%|█▎        | 54156/400000 [00:06<00:41, 8252.73it/s] 14%|█▍        | 55032/400000 [00:06<00:41, 8395.64it/s] 14%|█▍        | 55874/400000 [00:06<00:41, 8310.83it/s] 14%|█▍        | 56713/400000 [00:06<00:41, 8330.03it/s] 14%|█▍        | 57548/400000 [00:06<00:41, 8307.29it/s] 15%|█▍        | 58400/400000 [00:07<00:40, 8369.80it/s] 15%|█▍        | 59288/400000 [00:07<00:40, 8515.04it/s] 15%|█▌        | 60141/400000 [00:07<00:40, 8491.88it/s] 15%|█▌        | 60991/400000 [00:07<00:40, 8418.83it/s] 15%|█▌        | 61834/400000 [00:07<00:40, 8338.39it/s] 16%|█▌        | 62728/400000 [00:07<00:39, 8509.41it/s] 16%|█▌        | 63596/400000 [00:07<00:39, 8559.34it/s] 16%|█▌        | 64456/400000 [00:07<00:39, 8570.19it/s] 16%|█▋        | 65321/400000 [00:07<00:38, 8591.71it/s] 17%|█▋        | 66181/400000 [00:07<00:39, 8512.16it/s] 17%|█▋        | 67033/400000 [00:08<00:39, 8483.91it/s] 17%|█▋        | 67882/400000 [00:08<00:39, 8399.44it/s] 17%|█▋        | 68733/400000 [00:08<00:39, 8429.87it/s] 17%|█▋        | 69577/400000 [00:08<00:39, 8351.27it/s] 18%|█▊        | 70413/400000 [00:08<00:40, 8192.70it/s] 18%|█▊        | 71234/400000 [00:08<00:40, 8086.68it/s] 18%|█▊        | 72123/400000 [00:08<00:39, 8310.55it/s] 18%|█▊        | 72980/400000 [00:08<00:39, 8384.35it/s] 18%|█▊        | 73821/400000 [00:08<00:38, 8386.24it/s] 19%|█▊        | 74668/400000 [00:08<00:38, 8408.64it/s] 19%|█▉        | 75553/400000 [00:09<00:38, 8534.30it/s] 19%|█▉        | 76444/400000 [00:09<00:37, 8642.56it/s] 19%|█▉        | 77310/400000 [00:09<00:37, 8541.40it/s] 20%|█▉        | 78203/400000 [00:09<00:37, 8652.57it/s] 20%|█▉        | 79070/400000 [00:09<00:37, 8498.73it/s] 20%|█▉        | 79940/400000 [00:09<00:37, 8556.72it/s] 20%|██        | 80827/400000 [00:09<00:36, 8648.29it/s] 20%|██        | 81700/400000 [00:09<00:36, 8670.43it/s] 21%|██        | 82614/400000 [00:09<00:36, 8804.85it/s] 21%|██        | 83496/400000 [00:09<00:37, 8535.41it/s] 21%|██        | 84353/400000 [00:10<00:38, 8153.96it/s] 21%|██▏       | 85174/400000 [00:10<00:39, 8039.67it/s] 21%|██▏       | 85983/400000 [00:10<00:40, 7841.24it/s] 22%|██▏       | 86825/400000 [00:10<00:39, 8006.00it/s] 22%|██▏       | 87696/400000 [00:10<00:38, 8201.95it/s] 22%|██▏       | 88598/400000 [00:10<00:36, 8431.15it/s] 22%|██▏       | 89494/400000 [00:10<00:36, 8580.73it/s] 23%|██▎       | 90356/400000 [00:10<00:36, 8513.75it/s] 23%|██▎       | 91211/400000 [00:10<00:37, 8285.86it/s] 23%|██▎       | 92075/400000 [00:11<00:36, 8387.08it/s] 23%|██▎       | 92929/400000 [00:11<00:36, 8432.21it/s] 23%|██▎       | 93817/400000 [00:11<00:35, 8560.52it/s] 24%|██▎       | 94694/400000 [00:11<00:35, 8620.85it/s] 24%|██▍       | 95558/400000 [00:11<00:35, 8607.98it/s] 24%|██▍       | 96420/400000 [00:11<00:35, 8582.79it/s] 24%|██▍       | 97284/400000 [00:11<00:35, 8599.37it/s] 25%|██▍       | 98145/400000 [00:11<00:35, 8563.40it/s] 25%|██▍       | 99002/400000 [00:11<00:35, 8469.81it/s] 25%|██▍       | 99850/400000 [00:11<00:36, 8203.03it/s] 25%|██▌       | 100673/400000 [00:12<00:37, 7926.65it/s] 25%|██▌       | 101472/400000 [00:12<00:37, 7944.74it/s] 26%|██▌       | 102304/400000 [00:12<00:36, 8052.09it/s] 26%|██▌       | 103139/400000 [00:12<00:36, 8136.10it/s] 26%|██▌       | 103979/400000 [00:12<00:36, 8212.99it/s] 26%|██▌       | 104802/400000 [00:12<00:37, 7887.85it/s] 26%|██▋       | 105611/400000 [00:12<00:37, 7947.14it/s] 27%|██▋       | 106495/400000 [00:12<00:35, 8193.61it/s] 27%|██▋       | 107327/400000 [00:12<00:35, 8228.54it/s] 27%|██▋       | 108163/400000 [00:12<00:35, 8265.72it/s] 27%|██▋       | 108992/400000 [00:13<00:35, 8167.60it/s] 27%|██▋       | 109815/400000 [00:13<00:35, 8186.16it/s] 28%|██▊       | 110662/400000 [00:13<00:35, 8265.51it/s] 28%|██▊       | 111535/400000 [00:13<00:34, 8396.70it/s] 28%|██▊       | 112376/400000 [00:13<00:34, 8231.12it/s] 28%|██▊       | 113201/400000 [00:13<00:35, 8096.95it/s] 29%|██▊       | 114014/400000 [00:13<00:35, 8106.61it/s] 29%|██▊       | 114931/400000 [00:13<00:33, 8398.19it/s] 29%|██▉       | 115781/400000 [00:13<00:33, 8426.89it/s] 29%|██▉       | 116627/400000 [00:13<00:33, 8391.66it/s] 29%|██▉       | 117473/400000 [00:14<00:33, 8411.19it/s] 30%|██▉       | 118316/400000 [00:14<00:33, 8391.26it/s] 30%|██▉       | 119156/400000 [00:14<00:34, 8258.73it/s] 30%|██▉       | 119983/400000 [00:14<00:33, 8252.61it/s] 30%|███       | 120810/400000 [00:14<00:33, 8228.35it/s] 30%|███       | 121634/400000 [00:14<00:33, 8209.40it/s] 31%|███       | 122498/400000 [00:14<00:33, 8333.51it/s] 31%|███       | 123410/400000 [00:14<00:32, 8552.44it/s] 31%|███       | 124332/400000 [00:14<00:31, 8740.85it/s] 31%|███▏      | 125209/400000 [00:15<00:32, 8402.83it/s] 32%|███▏      | 126088/400000 [00:15<00:32, 8514.57it/s] 32%|███▏      | 126943/400000 [00:15<00:32, 8476.13it/s] 32%|███▏      | 127794/400000 [00:15<00:32, 8335.05it/s] 32%|███▏      | 128630/400000 [00:15<00:33, 8183.97it/s] 32%|███▏      | 129451/400000 [00:15<00:33, 7972.02it/s] 33%|███▎      | 130253/400000 [00:15<00:33, 7985.71it/s] 33%|███▎      | 131071/400000 [00:15<00:33, 8042.56it/s] 33%|███▎      | 131884/400000 [00:15<00:33, 8066.18it/s] 33%|███▎      | 132746/400000 [00:15<00:32, 8223.80it/s] 33%|███▎      | 133570/400000 [00:16<00:32, 8159.11it/s] 34%|███▎      | 134428/400000 [00:16<00:32, 8279.09it/s] 34%|███▍      | 135266/400000 [00:16<00:31, 8306.89it/s] 34%|███▍      | 136133/400000 [00:16<00:31, 8411.36it/s] 34%|███▍      | 136998/400000 [00:16<00:31, 8479.04it/s] 34%|███▍      | 137847/400000 [00:16<00:32, 8159.20it/s] 35%|███▍      | 138710/400000 [00:16<00:31, 8291.17it/s] 35%|███▍      | 139542/400000 [00:16<00:31, 8231.99it/s] 35%|███▌      | 140368/400000 [00:16<00:32, 8063.93it/s] 35%|███▌      | 141220/400000 [00:16<00:31, 8192.01it/s] 36%|███▌      | 142042/400000 [00:17<00:31, 8168.47it/s] 36%|███▌      | 142931/400000 [00:17<00:30, 8371.97it/s] 36%|███▌      | 143797/400000 [00:17<00:30, 8455.51it/s] 36%|███▌      | 144645/400000 [00:17<00:30, 8430.02it/s] 36%|███▋      | 145490/400000 [00:17<00:31, 8089.03it/s] 37%|███▋      | 146304/400000 [00:17<00:31, 8101.98it/s] 37%|███▋      | 147191/400000 [00:17<00:30, 8313.27it/s] 37%|███▋      | 148064/400000 [00:17<00:29, 8432.80it/s] 37%|███▋      | 148943/400000 [00:17<00:29, 8536.64it/s] 37%|███▋      | 149799/400000 [00:18<00:30, 8172.93it/s] 38%|███▊      | 150622/400000 [00:18<00:31, 7959.03it/s] 38%|███▊      | 151477/400000 [00:18<00:30, 8127.47it/s] 38%|███▊      | 152361/400000 [00:18<00:29, 8325.66it/s] 38%|███▊      | 153198/400000 [00:18<00:29, 8334.10it/s] 39%|███▊      | 154035/400000 [00:18<00:29, 8293.75it/s] 39%|███▊      | 154867/400000 [00:18<00:30, 8047.86it/s] 39%|███▉      | 155723/400000 [00:18<00:29, 8194.84it/s] 39%|███▉      | 156584/400000 [00:18<00:29, 8313.44it/s] 39%|███▉      | 157462/400000 [00:18<00:28, 8446.72it/s] 40%|███▉      | 158373/400000 [00:19<00:27, 8633.70it/s] 40%|███▉      | 159239/400000 [00:19<00:28, 8502.09it/s] 40%|████      | 160092/400000 [00:19<00:28, 8460.02it/s] 40%|████      | 160940/400000 [00:19<00:28, 8381.96it/s] 40%|████      | 161780/400000 [00:19<00:28, 8272.76it/s] 41%|████      | 162638/400000 [00:19<00:28, 8361.33it/s] 41%|████      | 163492/400000 [00:19<00:28, 8411.84it/s] 41%|████      | 164335/400000 [00:19<00:28, 8407.31it/s] 41%|████▏     | 165177/400000 [00:19<00:27, 8404.43it/s] 42%|████▏     | 166018/400000 [00:19<00:28, 8162.26it/s] 42%|████▏     | 166837/400000 [00:20<00:29, 7890.69it/s] 42%|████▏     | 167635/400000 [00:20<00:29, 7915.62it/s] 42%|████▏     | 168435/400000 [00:20<00:29, 7940.63it/s] 42%|████▏     | 169231/400000 [00:20<00:29, 7894.18it/s] 43%|████▎     | 170022/400000 [00:20<00:29, 7680.87it/s] 43%|████▎     | 170793/400000 [00:20<00:29, 7669.60it/s] 43%|████▎     | 171664/400000 [00:20<00:28, 7953.59it/s] 43%|████▎     | 172532/400000 [00:20<00:27, 8156.07it/s] 43%|████▎     | 173372/400000 [00:20<00:27, 8225.52it/s] 44%|████▎     | 174235/400000 [00:20<00:27, 8341.40it/s] 44%|████▍     | 175072/400000 [00:21<00:27, 8261.07it/s] 44%|████▍     | 175900/400000 [00:21<00:27, 8223.15it/s] 44%|████▍     | 176771/400000 [00:21<00:26, 8362.67it/s] 44%|████▍     | 177633/400000 [00:21<00:26, 8436.41it/s] 45%|████▍     | 178548/400000 [00:21<00:25, 8637.18it/s] 45%|████▍     | 179453/400000 [00:21<00:25, 8756.09it/s] 45%|████▌     | 180331/400000 [00:21<00:25, 8726.29it/s] 45%|████▌     | 181232/400000 [00:21<00:24, 8807.72it/s] 46%|████▌     | 182139/400000 [00:21<00:24, 8883.98it/s] 46%|████▌     | 183032/400000 [00:21<00:24, 8895.56it/s] 46%|████▌     | 183923/400000 [00:22<00:24, 8837.34it/s] 46%|████▌     | 184808/400000 [00:22<00:25, 8507.21it/s] 46%|████▋     | 185663/400000 [00:22<00:25, 8517.26it/s] 47%|████▋     | 186517/400000 [00:22<00:25, 8514.66it/s] 47%|████▋     | 187387/400000 [00:22<00:24, 8569.23it/s] 47%|████▋     | 188246/400000 [00:22<00:24, 8532.44it/s] 47%|████▋     | 189119/400000 [00:22<00:24, 8589.35it/s] 47%|████▋     | 189983/400000 [00:22<00:24, 8603.83it/s] 48%|████▊     | 190900/400000 [00:22<00:23, 8765.37it/s] 48%|████▊     | 191778/400000 [00:23<00:23, 8683.14it/s] 48%|████▊     | 192648/400000 [00:23<00:24, 8596.09it/s] 48%|████▊     | 193509/400000 [00:23<00:24, 8526.98it/s] 49%|████▊     | 194363/400000 [00:23<00:24, 8511.94it/s] 49%|████▉     | 195215/400000 [00:23<00:24, 8430.53it/s] 49%|████▉     | 196065/400000 [00:23<00:24, 8447.26it/s] 49%|████▉     | 196911/400000 [00:23<00:24, 8245.63it/s] 49%|████▉     | 197760/400000 [00:23<00:24, 8315.79it/s] 50%|████▉     | 198618/400000 [00:23<00:24, 8390.65it/s] 50%|████▉     | 199458/400000 [00:23<00:24, 8167.82it/s] 50%|█████     | 200302/400000 [00:24<00:24, 8247.50it/s] 50%|█████     | 201129/400000 [00:24<00:24, 8220.56it/s] 50%|█████     | 201953/400000 [00:24<00:24, 8118.06it/s] 51%|█████     | 202812/400000 [00:24<00:23, 8252.38it/s] 51%|█████     | 203643/400000 [00:24<00:23, 8269.22it/s] 51%|█████     | 204482/400000 [00:24<00:23, 8302.96it/s] 51%|█████▏    | 205313/400000 [00:24<00:23, 8268.65it/s] 52%|█████▏    | 206141/400000 [00:24<00:23, 8226.65it/s] 52%|█████▏    | 207019/400000 [00:24<00:23, 8382.44it/s] 52%|█████▏    | 207899/400000 [00:24<00:22, 8502.76it/s] 52%|█████▏    | 208774/400000 [00:25<00:22, 8572.78it/s] 52%|█████▏    | 209633/400000 [00:25<00:22, 8487.96it/s] 53%|█████▎    | 210509/400000 [00:25<00:22, 8567.05it/s] 53%|█████▎    | 211367/400000 [00:25<00:22, 8569.52it/s] 53%|█████▎    | 212225/400000 [00:25<00:21, 8572.00it/s] 53%|█████▎    | 213100/400000 [00:25<00:21, 8622.09it/s] 53%|█████▎    | 213963/400000 [00:25<00:21, 8569.47it/s] 54%|█████▎    | 214848/400000 [00:25<00:21, 8651.45it/s] 54%|█████▍    | 215744/400000 [00:25<00:21, 8740.88it/s] 54%|█████▍    | 216619/400000 [00:25<00:21, 8483.22it/s] 54%|█████▍    | 217470/400000 [00:26<00:21, 8466.03it/s] 55%|█████▍    | 218327/400000 [00:26<00:21, 8495.42it/s] 55%|█████▍    | 219188/400000 [00:26<00:21, 8527.54it/s] 55%|█████▌    | 220080/400000 [00:26<00:20, 8639.36it/s] 55%|█████▌    | 220945/400000 [00:26<00:20, 8559.38it/s] 55%|█████▌    | 221846/400000 [00:26<00:20, 8686.61it/s] 56%|█████▌    | 222716/400000 [00:26<00:20, 8646.17it/s] 56%|█████▌    | 223582/400000 [00:26<00:20, 8624.70it/s] 56%|█████▌    | 224470/400000 [00:26<00:20, 8699.55it/s] 56%|█████▋    | 225341/400000 [00:26<00:20, 8614.25it/s] 57%|█████▋    | 226203/400000 [00:27<00:20, 8601.71it/s] 57%|█████▋    | 227064/400000 [00:27<00:20, 8468.70it/s] 57%|█████▋    | 227912/400000 [00:27<00:20, 8458.98it/s] 57%|█████▋    | 228759/400000 [00:27<00:20, 8384.13it/s] 57%|█████▋    | 229598/400000 [00:27<00:20, 8356.16it/s] 58%|█████▊    | 230434/400000 [00:27<00:20, 8328.55it/s] 58%|█████▊    | 231268/400000 [00:27<00:20, 8228.10it/s] 58%|█████▊    | 232092/400000 [00:27<00:20, 8212.85it/s] 58%|█████▊    | 233019/400000 [00:27<00:19, 8503.09it/s] 58%|█████▊    | 233884/400000 [00:27<00:19, 8544.27it/s] 59%|█████▊    | 234775/400000 [00:28<00:19, 8648.68it/s] 59%|█████▉    | 235642/400000 [00:28<00:19, 8475.41it/s] 59%|█████▉    | 236551/400000 [00:28<00:18, 8648.60it/s] 59%|█████▉    | 237419/400000 [00:28<00:18, 8645.52it/s] 60%|█████▉    | 238286/400000 [00:28<00:18, 8611.37it/s] 60%|█████▉    | 239213/400000 [00:28<00:18, 8797.02it/s] 60%|██████    | 240095/400000 [00:28<00:18, 8712.32it/s] 60%|██████    | 240968/400000 [00:28<00:18, 8617.50it/s] 60%|██████    | 241886/400000 [00:28<00:18, 8777.37it/s] 61%|██████    | 242784/400000 [00:29<00:17, 8834.84it/s] 61%|██████    | 243669/400000 [00:29<00:18, 8522.01it/s] 61%|██████    | 244525/400000 [00:29<00:19, 8116.96it/s] 61%|██████▏   | 245406/400000 [00:29<00:18, 8311.48it/s] 62%|██████▏   | 246273/400000 [00:29<00:18, 8413.77it/s] 62%|██████▏   | 247156/400000 [00:29<00:17, 8532.20it/s] 62%|██████▏   | 248072/400000 [00:29<00:17, 8709.77it/s] 62%|██████▏   | 248947/400000 [00:29<00:17, 8445.60it/s] 62%|██████▏   | 249861/400000 [00:29<00:17, 8640.15it/s] 63%|██████▎   | 250750/400000 [00:29<00:17, 8712.38it/s] 63%|██████▎   | 251625/400000 [00:30<00:17, 8701.33it/s] 63%|██████▎   | 252498/400000 [00:30<00:17, 8625.96it/s] 63%|██████▎   | 253363/400000 [00:30<00:17, 8489.15it/s] 64%|██████▎   | 254245/400000 [00:30<00:16, 8583.35it/s] 64%|██████▍   | 255118/400000 [00:30<00:16, 8626.61it/s] 64%|██████▍   | 255992/400000 [00:30<00:16, 8657.15it/s] 64%|██████▍   | 256892/400000 [00:30<00:16, 8756.76it/s] 64%|██████▍   | 257769/400000 [00:30<00:16, 8639.32it/s] 65%|██████▍   | 258634/400000 [00:30<00:16, 8619.32it/s] 65%|██████▍   | 259506/400000 [00:30<00:16, 8649.21it/s] 65%|██████▌   | 260384/400000 [00:31<00:16, 8687.78it/s] 65%|██████▌   | 261265/400000 [00:31<00:15, 8723.34it/s] 66%|██████▌   | 262138/400000 [00:31<00:16, 8580.30it/s] 66%|██████▌   | 263018/400000 [00:31<00:15, 8643.48it/s] 66%|██████▌   | 263883/400000 [00:31<00:15, 8514.72it/s] 66%|██████▌   | 264762/400000 [00:31<00:15, 8595.42it/s] 66%|██████▋   | 265665/400000 [00:31<00:15, 8718.81it/s] 67%|██████▋   | 266538/400000 [00:31<00:15, 8541.28it/s] 67%|██████▋   | 267421/400000 [00:31<00:15, 8625.57it/s] 67%|██████▋   | 268304/400000 [00:31<00:15, 8685.60it/s] 67%|██████▋   | 269233/400000 [00:32<00:14, 8855.70it/s] 68%|██████▊   | 270149/400000 [00:32<00:14, 8942.76it/s] 68%|██████▊   | 271045/400000 [00:32<00:14, 8702.61it/s] 68%|██████▊   | 271918/400000 [00:32<00:14, 8653.32it/s] 68%|██████▊   | 272785/400000 [00:32<00:14, 8577.93it/s] 68%|██████▊   | 273645/400000 [00:32<00:14, 8525.44it/s] 69%|██████▊   | 274499/400000 [00:32<00:14, 8525.69it/s] 69%|██████▉   | 275353/400000 [00:32<00:14, 8352.08it/s] 69%|██████▉   | 276205/400000 [00:32<00:14, 8399.99it/s] 69%|██████▉   | 277046/400000 [00:33<00:14, 8369.92it/s] 69%|██████▉   | 277888/400000 [00:33<00:14, 8381.50it/s] 70%|██████▉   | 278740/400000 [00:33<00:14, 8420.38it/s] 70%|██████▉   | 279584/400000 [00:33<00:14, 8423.78it/s] 70%|███████   | 280446/400000 [00:33<00:14, 8480.46it/s] 70%|███████   | 281313/400000 [00:33<00:13, 8535.94it/s] 71%|███████   | 282168/400000 [00:33<00:13, 8537.35it/s] 71%|███████   | 283022/400000 [00:33<00:13, 8479.75it/s] 71%|███████   | 283871/400000 [00:33<00:14, 8281.85it/s] 71%|███████   | 284701/400000 [00:33<00:13, 8246.68it/s] 71%|███████▏  | 285534/400000 [00:34<00:13, 8269.95it/s] 72%|███████▏  | 286362/400000 [00:34<00:13, 8245.13it/s] 72%|███████▏  | 287187/400000 [00:34<00:13, 8189.14it/s] 72%|███████▏  | 288007/400000 [00:34<00:13, 8136.36it/s] 72%|███████▏  | 288892/400000 [00:34<00:13, 8337.72it/s] 72%|███████▏  | 289739/400000 [00:34<00:13, 8376.49it/s] 73%|███████▎  | 290609/400000 [00:34<00:12, 8470.01it/s] 73%|███████▎  | 291493/400000 [00:34<00:12, 8575.99it/s] 73%|███████▎  | 292352/400000 [00:34<00:12, 8394.56it/s] 73%|███████▎  | 293219/400000 [00:34<00:12, 8474.08it/s] 74%|███████▎  | 294071/400000 [00:35<00:12, 8487.79it/s] 74%|███████▎  | 294960/400000 [00:35<00:12, 8604.39it/s] 74%|███████▍  | 295865/400000 [00:35<00:11, 8732.91it/s] 74%|███████▍  | 296740/400000 [00:35<00:11, 8631.67it/s] 74%|███████▍  | 297605/400000 [00:35<00:11, 8582.44it/s] 75%|███████▍  | 298493/400000 [00:35<00:11, 8668.23it/s] 75%|███████▍  | 299361/400000 [00:35<00:11, 8461.14it/s] 75%|███████▌  | 300214/400000 [00:35<00:11, 8481.63it/s] 75%|███████▌  | 301064/400000 [00:35<00:11, 8292.00it/s] 75%|███████▌  | 301940/400000 [00:35<00:11, 8427.01it/s] 76%|███████▌  | 302785/400000 [00:36<00:11, 8424.33it/s] 76%|███████▌  | 303665/400000 [00:36<00:11, 8532.00it/s] 76%|███████▌  | 304520/400000 [00:36<00:11, 8534.86it/s] 76%|███████▋  | 305375/400000 [00:36<00:11, 8432.63it/s] 77%|███████▋  | 306279/400000 [00:36<00:10, 8604.00it/s] 77%|███████▋  | 307141/400000 [00:36<00:10, 8564.94it/s] 77%|███████▋  | 308044/400000 [00:36<00:10, 8699.23it/s] 77%|███████▋  | 308916/400000 [00:36<00:10, 8686.28it/s] 77%|███████▋  | 309786/400000 [00:36<00:10, 8622.27it/s] 78%|███████▊  | 310702/400000 [00:36<00:10, 8775.70it/s] 78%|███████▊  | 311582/400000 [00:37<00:10, 8780.05it/s] 78%|███████▊  | 312475/400000 [00:37<00:09, 8824.40it/s] 78%|███████▊  | 313359/400000 [00:37<00:09, 8795.06it/s] 79%|███████▊  | 314239/400000 [00:37<00:09, 8605.44it/s] 79%|███████▉  | 315101/400000 [00:37<00:10, 8458.27it/s] 79%|███████▉  | 315949/400000 [00:37<00:10, 8333.94it/s] 79%|███████▉  | 316828/400000 [00:37<00:09, 8464.08it/s] 79%|███████▉  | 317721/400000 [00:37<00:09, 8597.94it/s] 80%|███████▉  | 318583/400000 [00:37<00:09, 8503.23it/s] 80%|███████▉  | 319456/400000 [00:37<00:09, 8566.35it/s] 80%|████████  | 320314/400000 [00:38<00:09, 8521.96it/s] 80%|████████  | 321167/400000 [00:38<00:09, 8430.39it/s] 81%|████████  | 322029/400000 [00:38<00:09, 8485.87it/s] 81%|████████  | 322879/400000 [00:38<00:09, 8479.44it/s] 81%|████████  | 323791/400000 [00:38<00:08, 8661.48it/s] 81%|████████  | 324717/400000 [00:38<00:08, 8830.81it/s] 81%|████████▏ | 325636/400000 [00:38<00:08, 8934.72it/s] 82%|████████▏ | 326531/400000 [00:38<00:08, 8804.02it/s] 82%|████████▏ | 327413/400000 [00:38<00:08, 8566.13it/s] 82%|████████▏ | 328273/400000 [00:39<00:08, 8555.76it/s] 82%|████████▏ | 329131/400000 [00:39<00:08, 8518.73it/s] 82%|████████▏ | 329985/400000 [00:39<00:08, 8521.51it/s] 83%|████████▎ | 330850/400000 [00:39<00:08, 8556.12it/s] 83%|████████▎ | 331707/400000 [00:39<00:08, 8305.27it/s] 83%|████████▎ | 332540/400000 [00:39<00:08, 8012.66it/s] 83%|████████▎ | 333346/400000 [00:39<00:08, 8023.92it/s] 84%|████████▎ | 334180/400000 [00:39<00:08, 8113.21it/s] 84%|████████▍ | 335052/400000 [00:39<00:07, 8285.53it/s] 84%|████████▍ | 335883/400000 [00:39<00:07, 8249.73it/s] 84%|████████▍ | 336772/400000 [00:40<00:07, 8430.17it/s] 84%|████████▍ | 337636/400000 [00:40<00:07, 8490.53it/s] 85%|████████▍ | 338487/400000 [00:40<00:07, 8484.88it/s] 85%|████████▍ | 339354/400000 [00:40<00:07, 8539.26it/s] 85%|████████▌ | 340209/400000 [00:40<00:07, 8500.68it/s] 85%|████████▌ | 341103/400000 [00:40<00:06, 8626.50it/s] 85%|████████▌ | 341989/400000 [00:40<00:06, 8693.92it/s] 86%|████████▌ | 342860/400000 [00:40<00:06, 8695.53it/s] 86%|████████▌ | 343731/400000 [00:40<00:06, 8504.54it/s] 86%|████████▌ | 344583/400000 [00:40<00:06, 8369.82it/s] 86%|████████▋ | 345422/400000 [00:41<00:06, 8242.86it/s] 87%|████████▋ | 346277/400000 [00:41<00:06, 8331.25it/s] 87%|████████▋ | 347112/400000 [00:41<00:06, 8197.99it/s] 87%|████████▋ | 347934/400000 [00:41<00:06, 8167.16it/s] 87%|████████▋ | 348762/400000 [00:41<00:06, 8200.57it/s] 87%|████████▋ | 349625/400000 [00:41<00:06, 8323.11it/s] 88%|████████▊ | 350482/400000 [00:41<00:05, 8394.24it/s] 88%|████████▊ | 351323/400000 [00:41<00:05, 8390.14it/s] 88%|████████▊ | 352163/400000 [00:41<00:05, 8390.28it/s] 88%|████████▊ | 353003/400000 [00:41<00:05, 8158.77it/s] 88%|████████▊ | 353847/400000 [00:42<00:05, 8240.13it/s] 89%|████████▊ | 354699/400000 [00:42<00:05, 8319.97it/s] 89%|████████▉ | 355533/400000 [00:42<00:05, 8227.42it/s] 89%|████████▉ | 356357/400000 [00:42<00:05, 8195.16it/s] 89%|████████▉ | 357178/400000 [00:42<00:05, 8132.88it/s] 89%|████████▉ | 357992/400000 [00:42<00:05, 7876.03it/s] 90%|████████▉ | 358792/400000 [00:42<00:05, 7911.59it/s] 90%|████████▉ | 359614/400000 [00:42<00:05, 8000.43it/s] 90%|█████████ | 360510/400000 [00:42<00:04, 8264.55it/s] 90%|█████████ | 361340/400000 [00:42<00:04, 8227.12it/s] 91%|█████████ | 362199/400000 [00:43<00:04, 8330.23it/s] 91%|█████████ | 363048/400000 [00:43<00:04, 8375.90it/s] 91%|█████████ | 363901/400000 [00:43<00:04, 8419.65it/s] 91%|█████████ | 364744/400000 [00:43<00:04, 8379.51it/s] 91%|█████████▏| 365583/400000 [00:43<00:04, 8382.52it/s] 92%|█████████▏| 366450/400000 [00:43<00:03, 8465.61it/s] 92%|█████████▏| 367298/400000 [00:43<00:03, 8418.19it/s] 92%|█████████▏| 368141/400000 [00:43<00:03, 8414.45it/s] 92%|█████████▏| 369017/400000 [00:43<00:03, 8514.29it/s] 92%|█████████▏| 369869/400000 [00:44<00:03, 8299.23it/s] 93%|█████████▎| 370708/400000 [00:44<00:03, 8324.77it/s] 93%|█████████▎| 371542/400000 [00:44<00:03, 8206.66it/s] 93%|█████████▎| 372387/400000 [00:44<00:03, 8276.96it/s] 93%|█████████▎| 373234/400000 [00:44<00:03, 8331.54it/s] 94%|█████████▎| 374069/400000 [00:44<00:03, 8333.17it/s] 94%|█████████▎| 374903/400000 [00:44<00:03, 8314.98it/s] 94%|█████████▍| 375735/400000 [00:44<00:02, 8303.31it/s] 94%|█████████▍| 376605/400000 [00:44<00:02, 8417.88it/s] 94%|█████████▍| 377448/400000 [00:44<00:02, 8344.69it/s] 95%|█████████▍| 378284/400000 [00:45<00:02, 8326.41it/s] 95%|█████████▍| 379127/400000 [00:45<00:02, 8354.33it/s] 95%|█████████▍| 379963/400000 [00:45<00:02, 8239.76it/s] 95%|█████████▌| 380788/400000 [00:45<00:02, 8094.87it/s] 95%|█████████▌| 381630/400000 [00:45<00:02, 8189.47it/s] 96%|█████████▌| 382492/400000 [00:45<00:02, 8311.46it/s] 96%|█████████▌| 383337/400000 [00:45<00:01, 8351.17it/s] 96%|█████████▌| 384210/400000 [00:45<00:01, 8459.63it/s] 96%|█████████▋| 385057/400000 [00:45<00:01, 8233.71it/s] 96%|█████████▋| 385883/400000 [00:45<00:01, 8122.50it/s] 97%|█████████▋| 386710/400000 [00:46<00:01, 8164.90it/s] 97%|█████████▋| 387552/400000 [00:46<00:01, 8238.08it/s] 97%|█████████▋| 388397/400000 [00:46<00:01, 8296.93it/s] 97%|█████████▋| 389228/400000 [00:46<00:01, 8145.67it/s] 98%|█████████▊| 390044/400000 [00:46<00:01, 8068.12it/s] 98%|█████████▊| 390860/400000 [00:46<00:01, 8095.21it/s] 98%|█████████▊| 391705/400000 [00:46<00:01, 8196.65it/s] 98%|█████████▊| 392526/400000 [00:46<00:00, 8082.00it/s] 98%|█████████▊| 393336/400000 [00:46<00:00, 7900.14it/s] 99%|█████████▊| 394130/400000 [00:46<00:00, 7910.95it/s] 99%|█████████▊| 394927/400000 [00:47<00:00, 7926.28it/s] 99%|█████████▉| 395811/400000 [00:47<00:00, 8179.54it/s] 99%|█████████▉| 396669/400000 [00:47<00:00, 8294.18it/s] 99%|█████████▉| 397501/400000 [00:47<00:00, 8022.16it/s]100%|█████████▉| 398307/400000 [00:47<00:00, 7763.09it/s]100%|█████████▉| 399156/400000 [00:47<00:00, 7967.63it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8388.91it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa74811b4e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01134424685408629 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.010967148785607074 	 Accuracy: 66

  model saves at 66% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16056 out of table with 15648 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16056 out of table with 15648 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

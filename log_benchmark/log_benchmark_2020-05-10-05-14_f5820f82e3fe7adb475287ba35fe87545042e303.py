
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6af4c27470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 05:14:43.509169
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 05:14:43.513514
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 05:14:43.517110
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 05:14:43.520878
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6ae11b2b00> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 350167.6875
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 237212.7188
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 147280.5000
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 80760.4531
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 43218.4805
Epoch 6/10

1/1 [==============================] - 0s 106ms/step - loss: 24676.3809
Epoch 7/10

1/1 [==============================] - 0s 107ms/step - loss: 15279.8740
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 10267.3262
Epoch 9/10

1/1 [==============================] - 0s 97ms/step - loss: 7328.8677
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 5621.4155

  #### Inference Need return ypred, ytrue ######################### 
[[ -1.2507825   -0.08618301   1.4734008   -0.62043935   0.9648502
    1.2770655   -0.20923626  -2.612749    -0.8426392   -2.7782059
   -0.39551184   0.1072793   -1.0431038   -0.11388594  -0.0263896
   -0.37427768   0.5446192    0.867325     0.7445152    1.0472983
    0.22956067   0.41748416   0.66991836  -0.09889537   0.38939515
    0.8383574    1.6322753   -0.09960079  -0.55136186  -1.3500364
    0.7284144   -0.9027046    0.32996428  -0.41561317  -1.4705989
   -0.07083343  -1.0145581    1.2929933    0.43685848   1.643563
    0.26133978  -1.091511     1.860431    -1.1195682   -0.17575541
    0.38783288  -0.6029283   -0.83565235  -1.0241326   -0.49325037
   -1.381017     0.52146924  -0.09894687  -0.41535372   1.4522963
   -0.97307426   0.6841029    0.53401446   1.231922    -0.7824
   -0.310517     8.811403     9.762027     7.493021     9.948032
    8.964454     8.219369     8.461655     9.022638     8.89225
    8.913893    10.09722      9.074694     9.675428    10.222708
    8.913066    10.605383     9.196466     9.80625      7.0539145
   10.940913     9.021378     8.419937    10.168107    10.394904
    6.3370876    6.4622498    9.687686     8.068356     9.74694
    7.754702     8.828469     8.085791     8.843394     8.835345
    9.864561     8.718549     8.420471     7.5227013    7.5694475
    8.3915       8.917728     8.6273575    8.787621     8.73328
    9.626071     9.036511     7.6435623   10.215285    10.104313
    8.515919     7.711796    11.115847     9.663591     8.790633
   10.052521    10.354106     7.981193     8.29799      6.673007
    0.39910054   0.85341966  -1.7160776   -0.48865798  -0.40686846
    0.82554257  -1.3424175    0.5000604   -0.66122425   0.11799556
   -0.12508589  -0.23428406   1.2055681    0.2828589    0.6141611
   -0.2013865    0.47093475   0.3897612    0.44879466  -1.4546092
    0.6946752    1.1869824    0.55038184   1.1009437    0.99548566
   -0.3512581   -0.20465422  -1.3282005   -0.6711246    0.1797753
   -1.124246    -0.9632213    1.2916969   -0.6905295   -1.782929
    1.4200065   -0.04822084  -1.5085204   -0.31808537   1.4873072
    0.46965134  -0.6143146    1.5569351   -0.35529134   0.19532692
   -0.582057    -0.4327215    0.6816828   -0.9261925   -0.647773
    0.8550489    2.11008      0.8160342   -1.2040796    0.29946902
   -1.200984     0.31600916  -1.2173959    0.644509     0.27908388
    0.3959058    0.3582934    0.23866951   1.2391694    1.288482
    1.6168625    1.2132388    2.2622004    0.7651043    2.4513092
    2.019383     0.5507981    1.2115018    0.6784669    0.8303338
    0.70199275   0.6647868    1.7757032    0.98460805   1.5183816
    0.28543687   0.69109464   2.1831179    1.094188     0.5315513
    1.413946     0.24067068   2.813556     1.5809975    0.25621277
    0.8062279    0.39586246   0.11097109   1.9139997    0.7404704
    0.33211702   0.61536455   0.28917277   0.64090526   2.2107651
    2.2545512    0.29908884   1.6154026    0.54499567   1.0524828
    0.8542335    0.67932624   0.7358013    1.0982838    0.37991333
    1.9504013    0.28514028   0.60174525   1.9593825    1.8071948
    0.53337216   1.0215993    0.1609875    2.1061754    0.5787572
    0.06986672  10.140118     9.299103    10.325876     8.91191
    9.052881     9.400251    10.255641    10.064175    11.041391
    9.274361     9.57892      8.909894     9.870961     9.057114
    7.936893     7.9768963   10.12051     10.154435     9.05775
    8.428803    10.018209     8.4689865    9.410399     9.24923
    9.505486     8.688212     7.9310026    8.375555     7.0136876
    8.889862     9.954468     7.396354    10.454502     9.503843
    8.665319    10.499744    10.017683    10.346222    10.856859
   10.143924     9.384084     9.027555     9.837173     8.563023
    9.995662    10.024693     9.232164     8.183313    10.692651
    8.343708     8.32704      7.7507977    8.097253     9.394617
    9.572581    10.222309     9.687999     9.239154     8.551203
    2.8055353    1.4123931    0.27280056   0.53849804   0.49229646
    1.7842       0.4817015    0.2745602    2.2698348    0.4906907
    1.2762008    0.9389604    0.56598306   0.37326008   2.9973416
    1.2608337    2.1295748    0.7540703    0.3883294    0.34846532
    1.0835264    0.39774823   1.268216     1.540438     1.654187
    0.5150957    0.4895128    0.35788822   3.7888455    0.59413403
    0.47408354   1.755897     0.79691076   0.47099578   0.2499342
    2.779368     0.79107225   0.68138313   1.1077802    0.6536363
    1.3086932    0.38245952   1.2677099    1.9179695    1.3819783
    0.15963328   0.09865808   1.441322     0.9543823    0.72476757
    1.6583391    0.79708177   3.1303868    1.8122306    2.444475
    1.0904448    0.6134194    1.2962286    0.21140599   0.17447013
  -11.914525     1.6172181   -8.296771  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 05:14:52.700543
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6723
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 05:14:52.704876
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8611.86
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 05:14:52.708752
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7401
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 05:14:52.712541
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -770.253
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140096660791424
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140095450764176
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140095450764680
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140095450765184
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140095450765688
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140095450766192

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6adf3dae80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.510434
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.473212
grad_step = 000002, loss = 0.449933
grad_step = 000003, loss = 0.427456
grad_step = 000004, loss = 0.403386
grad_step = 000005, loss = 0.379521
grad_step = 000006, loss = 0.360540
grad_step = 000007, loss = 0.352318
grad_step = 000008, loss = 0.348121
grad_step = 000009, loss = 0.336177
grad_step = 000010, loss = 0.321295
grad_step = 000011, loss = 0.310086
grad_step = 000012, loss = 0.302663
grad_step = 000013, loss = 0.295958
grad_step = 000014, loss = 0.288186
grad_step = 000015, loss = 0.279006
grad_step = 000016, loss = 0.268855
grad_step = 000017, loss = 0.258555
grad_step = 000018, loss = 0.249005
grad_step = 000019, loss = 0.240652
grad_step = 000020, loss = 0.233170
grad_step = 000021, loss = 0.225287
grad_step = 000022, loss = 0.216424
grad_step = 000023, loss = 0.207478
grad_step = 000024, loss = 0.199539
grad_step = 000025, loss = 0.192453
grad_step = 000026, loss = 0.185377
grad_step = 000027, loss = 0.177812
grad_step = 000028, loss = 0.169908
grad_step = 000029, loss = 0.162215
grad_step = 000030, loss = 0.155282
grad_step = 000031, loss = 0.149147
grad_step = 000032, loss = 0.143079
grad_step = 000033, loss = 0.136450
grad_step = 000034, loss = 0.129779
grad_step = 000035, loss = 0.123734
grad_step = 000036, loss = 0.118325
grad_step = 000037, loss = 0.113101
grad_step = 000038, loss = 0.107732
grad_step = 000039, loss = 0.102346
grad_step = 000040, loss = 0.097371
grad_step = 000041, loss = 0.092921
grad_step = 000042, loss = 0.088623
grad_step = 000043, loss = 0.084211
grad_step = 000044, loss = 0.079891
grad_step = 000045, loss = 0.075951
grad_step = 000046, loss = 0.072332
grad_step = 000047, loss = 0.068798
grad_step = 000048, loss = 0.065281
grad_step = 000049, loss = 0.061928
grad_step = 000050, loss = 0.058832
grad_step = 000051, loss = 0.055887
grad_step = 000052, loss = 0.052997
grad_step = 000053, loss = 0.050190
grad_step = 000054, loss = 0.047559
grad_step = 000055, loss = 0.045084
grad_step = 000056, loss = 0.042700
grad_step = 000057, loss = 0.040390
grad_step = 000058, loss = 0.038177
grad_step = 000059, loss = 0.036080
grad_step = 000060, loss = 0.034072
grad_step = 000061, loss = 0.032182
grad_step = 000062, loss = 0.030364
grad_step = 000063, loss = 0.028609
grad_step = 000064, loss = 0.026949
grad_step = 000065, loss = 0.025396
grad_step = 000066, loss = 0.023912
grad_step = 000067, loss = 0.022481
grad_step = 000068, loss = 0.021138
grad_step = 000069, loss = 0.019895
grad_step = 000070, loss = 0.018730
grad_step = 000071, loss = 0.017610
grad_step = 000072, loss = 0.016528
grad_step = 000073, loss = 0.015471
grad_step = 000074, loss = 0.014509
grad_step = 000075, loss = 0.013646
grad_step = 000076, loss = 0.012807
grad_step = 000077, loss = 0.011984
grad_step = 000078, loss = 0.011230
grad_step = 000079, loss = 0.010559
grad_step = 000080, loss = 0.009919
grad_step = 000081, loss = 0.009287
grad_step = 000082, loss = 0.008702
grad_step = 000083, loss = 0.008190
grad_step = 000084, loss = 0.007710
grad_step = 000085, loss = 0.007236
grad_step = 000086, loss = 0.006793
grad_step = 000087, loss = 0.006406
grad_step = 000088, loss = 0.006054
grad_step = 000089, loss = 0.005715
grad_step = 000090, loss = 0.005389
grad_step = 000091, loss = 0.005094
grad_step = 000092, loss = 0.004831
grad_step = 000093, loss = 0.004597
grad_step = 000094, loss = 0.004378
grad_step = 000095, loss = 0.004167
grad_step = 000096, loss = 0.003969
grad_step = 000097, loss = 0.003794
grad_step = 000098, loss = 0.003640
grad_step = 000099, loss = 0.003503
grad_step = 000100, loss = 0.003379
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003265
grad_step = 000102, loss = 0.003159
grad_step = 000103, loss = 0.003060
grad_step = 000104, loss = 0.002970
grad_step = 000105, loss = 0.002888
grad_step = 000106, loss = 0.002815
grad_step = 000107, loss = 0.002749
grad_step = 000108, loss = 0.002692
grad_step = 000109, loss = 0.002641
grad_step = 000110, loss = 0.002595
grad_step = 000111, loss = 0.002555
grad_step = 000112, loss = 0.002519
grad_step = 000113, loss = 0.002488
grad_step = 000114, loss = 0.002462
grad_step = 000115, loss = 0.002443
grad_step = 000116, loss = 0.002438
grad_step = 000117, loss = 0.002457
grad_step = 000118, loss = 0.002518
grad_step = 000119, loss = 0.002611
grad_step = 000120, loss = 0.002596
grad_step = 000121, loss = 0.002429
grad_step = 000122, loss = 0.002323
grad_step = 000123, loss = 0.002392
grad_step = 000124, loss = 0.002441
grad_step = 000125, loss = 0.002351
grad_step = 000126, loss = 0.002307
grad_step = 000127, loss = 0.002348
grad_step = 000128, loss = 0.002338
grad_step = 000129, loss = 0.002303
grad_step = 000130, loss = 0.002292
grad_step = 000131, loss = 0.002294
grad_step = 000132, loss = 0.002288
grad_step = 000133, loss = 0.002268
grad_step = 000134, loss = 0.002254
grad_step = 000135, loss = 0.002264
grad_step = 000136, loss = 0.002252
grad_step = 000137, loss = 0.002232
grad_step = 000138, loss = 0.002237
grad_step = 000139, loss = 0.002240
grad_step = 000140, loss = 0.002220
grad_step = 000141, loss = 0.002217
grad_step = 000142, loss = 0.002225
grad_step = 000143, loss = 0.002214
grad_step = 000144, loss = 0.002203
grad_step = 000145, loss = 0.002211
grad_step = 000146, loss = 0.002207
grad_step = 000147, loss = 0.002194
grad_step = 000148, loss = 0.002197
grad_step = 000149, loss = 0.002199
grad_step = 000150, loss = 0.002188
grad_step = 000151, loss = 0.002184
grad_step = 000152, loss = 0.002188
grad_step = 000153, loss = 0.002182
grad_step = 000154, loss = 0.002174
grad_step = 000155, loss = 0.002175
grad_step = 000156, loss = 0.002175
grad_step = 000157, loss = 0.002168
grad_step = 000158, loss = 0.002164
grad_step = 000159, loss = 0.002164
grad_step = 000160, loss = 0.002161
grad_step = 000161, loss = 0.002155
grad_step = 000162, loss = 0.002152
grad_step = 000163, loss = 0.002151
grad_step = 000164, loss = 0.002147
grad_step = 000165, loss = 0.002141
grad_step = 000166, loss = 0.002139
grad_step = 000167, loss = 0.002136
grad_step = 000168, loss = 0.002132
grad_step = 000169, loss = 0.002127
grad_step = 000170, loss = 0.002123
grad_step = 000171, loss = 0.002120
grad_step = 000172, loss = 0.002115
grad_step = 000173, loss = 0.002109
grad_step = 000174, loss = 0.002105
grad_step = 000175, loss = 0.002101
grad_step = 000176, loss = 0.002097
grad_step = 000177, loss = 0.002095
grad_step = 000178, loss = 0.002101
grad_step = 000179, loss = 0.002131
grad_step = 000180, loss = 0.002206
grad_step = 000181, loss = 0.002363
grad_step = 000182, loss = 0.002302
grad_step = 000183, loss = 0.002139
grad_step = 000184, loss = 0.002066
grad_step = 000185, loss = 0.002181
grad_step = 000186, loss = 0.002191
grad_step = 000187, loss = 0.002051
grad_step = 000188, loss = 0.002105
grad_step = 000189, loss = 0.002168
grad_step = 000190, loss = 0.002048
grad_step = 000191, loss = 0.002059
grad_step = 000192, loss = 0.002123
grad_step = 000193, loss = 0.002031
grad_step = 000194, loss = 0.002029
grad_step = 000195, loss = 0.002076
grad_step = 000196, loss = 0.002004
grad_step = 000197, loss = 0.002004
grad_step = 000198, loss = 0.002033
grad_step = 000199, loss = 0.001992
grad_step = 000200, loss = 0.001971
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001995
grad_step = 000202, loss = 0.001972
grad_step = 000203, loss = 0.001947
grad_step = 000204, loss = 0.001959
grad_step = 000205, loss = 0.001954
grad_step = 000206, loss = 0.001930
grad_step = 000207, loss = 0.001926
grad_step = 000208, loss = 0.001932
grad_step = 000209, loss = 0.001920
grad_step = 000210, loss = 0.001902
grad_step = 000211, loss = 0.001897
grad_step = 000212, loss = 0.001902
grad_step = 000213, loss = 0.001897
grad_step = 000214, loss = 0.001885
grad_step = 000215, loss = 0.001892
grad_step = 000216, loss = 0.001976
grad_step = 000217, loss = 0.002217
grad_step = 000218, loss = 0.002188
grad_step = 000219, loss = 0.002172
grad_step = 000220, loss = 0.002100
grad_step = 000221, loss = 0.001916
grad_step = 000222, loss = 0.001980
grad_step = 000223, loss = 0.002115
grad_step = 000224, loss = 0.002011
grad_step = 000225, loss = 0.001876
grad_step = 000226, loss = 0.002021
grad_step = 000227, loss = 0.002016
grad_step = 000228, loss = 0.001845
grad_step = 000229, loss = 0.001965
grad_step = 000230, loss = 0.001947
grad_step = 000231, loss = 0.001886
grad_step = 000232, loss = 0.001897
grad_step = 000233, loss = 0.001895
grad_step = 000234, loss = 0.001905
grad_step = 000235, loss = 0.001859
grad_step = 000236, loss = 0.001865
grad_step = 000237, loss = 0.001872
grad_step = 000238, loss = 0.001873
grad_step = 000239, loss = 0.001828
grad_step = 000240, loss = 0.001847
grad_step = 000241, loss = 0.001869
grad_step = 000242, loss = 0.001798
grad_step = 000243, loss = 0.001851
grad_step = 000244, loss = 0.001830
grad_step = 000245, loss = 0.001802
grad_step = 000246, loss = 0.001836
grad_step = 000247, loss = 0.001800
grad_step = 000248, loss = 0.001810
grad_step = 000249, loss = 0.001801
grad_step = 000250, loss = 0.001793
grad_step = 000251, loss = 0.001795
grad_step = 000252, loss = 0.001784
grad_step = 000253, loss = 0.001782
grad_step = 000254, loss = 0.001780
grad_step = 000255, loss = 0.001778
grad_step = 000256, loss = 0.001767
grad_step = 000257, loss = 0.001773
grad_step = 000258, loss = 0.001766
grad_step = 000259, loss = 0.001756
grad_step = 000260, loss = 0.001765
grad_step = 000261, loss = 0.001755
grad_step = 000262, loss = 0.001749
grad_step = 000263, loss = 0.001754
grad_step = 000264, loss = 0.001745
grad_step = 000265, loss = 0.001743
grad_step = 000266, loss = 0.001743
grad_step = 000267, loss = 0.001737
grad_step = 000268, loss = 0.001736
grad_step = 000269, loss = 0.001733
grad_step = 000270, loss = 0.001730
grad_step = 000271, loss = 0.001729
grad_step = 000272, loss = 0.001724
grad_step = 000273, loss = 0.001723
grad_step = 000274, loss = 0.001722
grad_step = 000275, loss = 0.001717
grad_step = 000276, loss = 0.001715
grad_step = 000277, loss = 0.001714
grad_step = 000278, loss = 0.001710
grad_step = 000279, loss = 0.001708
grad_step = 000280, loss = 0.001706
grad_step = 000281, loss = 0.001703
grad_step = 000282, loss = 0.001701
grad_step = 000283, loss = 0.001699
grad_step = 000284, loss = 0.001696
grad_step = 000285, loss = 0.001694
grad_step = 000286, loss = 0.001691
grad_step = 000287, loss = 0.001689
grad_step = 000288, loss = 0.001687
grad_step = 000289, loss = 0.001685
grad_step = 000290, loss = 0.001682
grad_step = 000291, loss = 0.001679
grad_step = 000292, loss = 0.001677
grad_step = 000293, loss = 0.001675
grad_step = 000294, loss = 0.001673
grad_step = 000295, loss = 0.001673
grad_step = 000296, loss = 0.001673
grad_step = 000297, loss = 0.001675
grad_step = 000298, loss = 0.001678
grad_step = 000299, loss = 0.001683
grad_step = 000300, loss = 0.001684
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001682
grad_step = 000302, loss = 0.001673
grad_step = 000303, loss = 0.001661
grad_step = 000304, loss = 0.001653
grad_step = 000305, loss = 0.001652
grad_step = 000306, loss = 0.001657
grad_step = 000307, loss = 0.001660
grad_step = 000308, loss = 0.001653
grad_step = 000309, loss = 0.001642
grad_step = 000310, loss = 0.001636
grad_step = 000311, loss = 0.001637
grad_step = 000312, loss = 0.001639
grad_step = 000313, loss = 0.001637
grad_step = 000314, loss = 0.001631
grad_step = 000315, loss = 0.001625
grad_step = 000316, loss = 0.001622
grad_step = 000317, loss = 0.001623
grad_step = 000318, loss = 0.001624
grad_step = 000319, loss = 0.001621
grad_step = 000320, loss = 0.001617
grad_step = 000321, loss = 0.001612
grad_step = 000322, loss = 0.001609
grad_step = 000323, loss = 0.001607
grad_step = 000324, loss = 0.001606
grad_step = 000325, loss = 0.001605
grad_step = 000326, loss = 0.001603
grad_step = 000327, loss = 0.001601
grad_step = 000328, loss = 0.001597
grad_step = 000329, loss = 0.001594
grad_step = 000330, loss = 0.001593
grad_step = 000331, loss = 0.001595
grad_step = 000332, loss = 0.001605
grad_step = 000333, loss = 0.001628
grad_step = 000334, loss = 0.001676
grad_step = 000335, loss = 0.001710
grad_step = 000336, loss = 0.001751
grad_step = 000337, loss = 0.001676
grad_step = 000338, loss = 0.001618
grad_step = 000339, loss = 0.001643
grad_step = 000340, loss = 0.001664
grad_step = 000341, loss = 0.001605
grad_step = 000342, loss = 0.001567
grad_step = 000343, loss = 0.001608
grad_step = 000344, loss = 0.001610
grad_step = 000345, loss = 0.001572
grad_step = 000346, loss = 0.001583
grad_step = 000347, loss = 0.001593
grad_step = 000348, loss = 0.001563
grad_step = 000349, loss = 0.001557
grad_step = 000350, loss = 0.001568
grad_step = 000351, loss = 0.001560
grad_step = 000352, loss = 0.001552
grad_step = 000353, loss = 0.001551
grad_step = 000354, loss = 0.001548
grad_step = 000355, loss = 0.001540
grad_step = 000356, loss = 0.001535
grad_step = 000357, loss = 0.001535
grad_step = 000358, loss = 0.001533
grad_step = 000359, loss = 0.001526
grad_step = 000360, loss = 0.001525
grad_step = 000361, loss = 0.001527
grad_step = 000362, loss = 0.001522
grad_step = 000363, loss = 0.001514
grad_step = 000364, loss = 0.001510
grad_step = 000365, loss = 0.001512
grad_step = 000366, loss = 0.001512
grad_step = 000367, loss = 0.001507
grad_step = 000368, loss = 0.001503
grad_step = 000369, loss = 0.001501
grad_step = 000370, loss = 0.001503
grad_step = 000371, loss = 0.001508
grad_step = 000372, loss = 0.001518
grad_step = 000373, loss = 0.001537
grad_step = 000374, loss = 0.001562
grad_step = 000375, loss = 0.001593
grad_step = 000376, loss = 0.001611
grad_step = 000377, loss = 0.001615
grad_step = 000378, loss = 0.001629
grad_step = 000379, loss = 0.001637
grad_step = 000380, loss = 0.001636
grad_step = 000381, loss = 0.001571
grad_step = 000382, loss = 0.001490
grad_step = 000383, loss = 0.001472
grad_step = 000384, loss = 0.001523
grad_step = 000385, loss = 0.001557
grad_step = 000386, loss = 0.001523
grad_step = 000387, loss = 0.001469
grad_step = 000388, loss = 0.001460
grad_step = 000389, loss = 0.001489
grad_step = 000390, loss = 0.001497
grad_step = 000391, loss = 0.001471
grad_step = 000392, loss = 0.001449
grad_step = 000393, loss = 0.001457
grad_step = 000394, loss = 0.001470
grad_step = 000395, loss = 0.001464
grad_step = 000396, loss = 0.001446
grad_step = 000397, loss = 0.001439
grad_step = 000398, loss = 0.001441
grad_step = 000399, loss = 0.001438
grad_step = 000400, loss = 0.001431
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001427
grad_step = 000402, loss = 0.001431
grad_step = 000403, loss = 0.001431
grad_step = 000404, loss = 0.001425
grad_step = 000405, loss = 0.001418
grad_step = 000406, loss = 0.001418
grad_step = 000407, loss = 0.001421
grad_step = 000408, loss = 0.001419
grad_step = 000409, loss = 0.001413
grad_step = 000410, loss = 0.001409
grad_step = 000411, loss = 0.001411
grad_step = 000412, loss = 0.001416
grad_step = 000413, loss = 0.001419
grad_step = 000414, loss = 0.001422
grad_step = 000415, loss = 0.001427
grad_step = 000416, loss = 0.001434
grad_step = 000417, loss = 0.001435
grad_step = 000418, loss = 0.001428
grad_step = 000419, loss = 0.001407
grad_step = 000420, loss = 0.001387
grad_step = 000421, loss = 0.001378
grad_step = 000422, loss = 0.001380
grad_step = 000423, loss = 0.001385
grad_step = 000424, loss = 0.001382
grad_step = 000425, loss = 0.001374
grad_step = 000426, loss = 0.001366
grad_step = 000427, loss = 0.001361
grad_step = 000428, loss = 0.001362
grad_step = 000429, loss = 0.001364
grad_step = 000430, loss = 0.001364
grad_step = 000431, loss = 0.001361
grad_step = 000432, loss = 0.001356
grad_step = 000433, loss = 0.001351
grad_step = 000434, loss = 0.001347
grad_step = 000435, loss = 0.001344
grad_step = 000436, loss = 0.001342
grad_step = 000437, loss = 0.001340
grad_step = 000438, loss = 0.001339
grad_step = 000439, loss = 0.001338
grad_step = 000440, loss = 0.001338
grad_step = 000441, loss = 0.001337
grad_step = 000442, loss = 0.001338
grad_step = 000443, loss = 0.001339
grad_step = 000444, loss = 0.001343
grad_step = 000445, loss = 0.001349
grad_step = 000446, loss = 0.001363
grad_step = 000447, loss = 0.001385
grad_step = 000448, loss = 0.001428
grad_step = 000449, loss = 0.001455
grad_step = 000450, loss = 0.001498
grad_step = 000451, loss = 0.001428
grad_step = 000452, loss = 0.001348
grad_step = 000453, loss = 0.001308
grad_step = 000454, loss = 0.001352
grad_step = 000455, loss = 0.001393
grad_step = 000456, loss = 0.001364
grad_step = 000457, loss = 0.001336
grad_step = 000458, loss = 0.001383
grad_step = 000459, loss = 0.001426
grad_step = 000460, loss = 0.001429
grad_step = 000461, loss = 0.001386
grad_step = 000462, loss = 0.001361
grad_step = 000463, loss = 0.001346
grad_step = 000464, loss = 0.001318
grad_step = 000465, loss = 0.001295
grad_step = 000466, loss = 0.001278
grad_step = 000467, loss = 0.001292
grad_step = 000468, loss = 0.001312
grad_step = 000469, loss = 0.001312
grad_step = 000470, loss = 0.001283
grad_step = 000471, loss = 0.001253
grad_step = 000472, loss = 0.001247
grad_step = 000473, loss = 0.001261
grad_step = 000474, loss = 0.001275
grad_step = 000475, loss = 0.001272
grad_step = 000476, loss = 0.001253
grad_step = 000477, loss = 0.001238
grad_step = 000478, loss = 0.001235
grad_step = 000479, loss = 0.001241
grad_step = 000480, loss = 0.001247
grad_step = 000481, loss = 0.001244
grad_step = 000482, loss = 0.001235
grad_step = 000483, loss = 0.001222
grad_step = 000484, loss = 0.001214
grad_step = 000485, loss = 0.001215
grad_step = 000486, loss = 0.001217
grad_step = 000487, loss = 0.001218
grad_step = 000488, loss = 0.001213
grad_step = 000489, loss = 0.001206
grad_step = 000490, loss = 0.001198
grad_step = 000491, loss = 0.001195
grad_step = 000492, loss = 0.001196
grad_step = 000493, loss = 0.001196
grad_step = 000494, loss = 0.001194
grad_step = 000495, loss = 0.001190
grad_step = 000496, loss = 0.001185
grad_step = 000497, loss = 0.001180
grad_step = 000498, loss = 0.001177
grad_step = 000499, loss = 0.001176
grad_step = 000500, loss = 0.001174
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001173
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

  date_run                              2020-05-10 05:15:14.680999
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.256988
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 05:15:14.687085
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.162643
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 05:15:14.694508
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.148028
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 05:15:14.700100
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.47142
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
0   2020-05-10 05:14:43.509169  ...    mean_absolute_error
1   2020-05-10 05:14:43.513514  ...     mean_squared_error
2   2020-05-10 05:14:43.517110  ...  median_absolute_error
3   2020-05-10 05:14:43.520878  ...               r2_score
4   2020-05-10 05:14:52.700543  ...    mean_absolute_error
5   2020-05-10 05:14:52.704876  ...     mean_squared_error
6   2020-05-10 05:14:52.708752  ...  median_absolute_error
7   2020-05-10 05:14:52.712541  ...               r2_score
8   2020-05-10 05:15:14.680999  ...    mean_absolute_error
9   2020-05-10 05:15:14.687085  ...     mean_squared_error
10  2020-05-10 05:15:14.694508  ...  median_absolute_error
11  2020-05-10 05:15:14.700100  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3571712/9912422 [00:00<00:00, 35587169.22it/s]9920512it [00:00, 32348168.45it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1117771.54it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 459126.66it/s]1654784it [00:00, 11489002.32it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185291.63it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1cba149780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1c5788cc18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1cba149e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1cba100e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1c5788e080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1c60faf4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1c5788cc18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1cba149f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1c6cafbcc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1cba149f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1c6cafbcc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ffbe3e661d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=78fd23bff6c004d149cb049cd0345caade1fd27383b4251c3329257d21177b5b
  Stored in directory: /tmp/pip-ephem-wheel-cache-tp_qyu2v/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ffb7ba4b0b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3547136/17464789 [=====>........................] - ETA: 0s
10240000/17464789 [================>.............] - ETA: 0s
13893632/17464789 [======================>.......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 05:16:42.519941: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 05:16:42.524363: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-10 05:16:42.524560: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cd35ba0ef0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 05:16:42.524579: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8813 - accuracy: 0.4860 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7893 - accuracy: 0.4920
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7490 - accuracy: 0.4946
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7757 - accuracy: 0.4929
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7663 - accuracy: 0.4935
11000/25000 [============>.................] - ETA: 4s - loss: 7.7419 - accuracy: 0.4951
12000/25000 [=============>................] - ETA: 3s - loss: 7.7548 - accuracy: 0.4942
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7398 - accuracy: 0.4952
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7367 - accuracy: 0.4954
15000/25000 [=================>............] - ETA: 2s - loss: 7.7331 - accuracy: 0.4957
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7088 - accuracy: 0.4972
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6705 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6688 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 9s 350us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 05:16:58.601978
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 05:16:58.601978  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 05:17:05.248843: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 05:17:05.254356: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-10 05:17:05.254820: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5597594216b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 05:17:05.255129: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f209dc08d30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5377 - crf_viterbi_accuracy: 0.3467 - val_loss: 1.4848 - val_crf_viterbi_accuracy: 0.3200

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2092fb0f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6881 - accuracy: 0.4986
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6532 - accuracy: 0.5009
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6104 - accuracy: 0.5037
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5915 - accuracy: 0.5049
11000/25000 [============>.................] - ETA: 4s - loss: 7.5830 - accuracy: 0.5055
12000/25000 [=============>................] - ETA: 3s - loss: 7.6027 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6018 - accuracy: 0.5042
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5921 - accuracy: 0.5049
15000/25000 [=================>............] - ETA: 2s - loss: 7.6288 - accuracy: 0.5025
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6177 - accuracy: 0.5032
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6197 - accuracy: 0.5031
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6240 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6260 - accuracy: 0.5027
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6308 - accuracy: 0.5023
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6443 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 9s 355us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f20906c9438> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:20:35, 9.84kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:16:18, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:08:36, 19.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:30:28, 28.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:56:23, 40.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.59M/862M [00:01<4:08:07, 57.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:52:53, 81.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<2:00:36, 117kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<1:23:58, 167kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:01<58:40, 238kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:01<40:54, 339kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:02<28:38, 482kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:02<20:00, 686kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.8M/862M [00:02<14:03, 972kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.7M/862M [00:02<09:51, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<07:06, 1.90MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<06:52, 1.95MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<08:27, 1.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:42, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:04<04:53, 2.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<07:42, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<07:00, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<05:13, 2.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<06:24, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<05:52, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<04:23, 3.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<07:02, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<05:35, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<06:02, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<05:34, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:12<04:13, 3.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<06:00, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<05:32, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<04:08, 3.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<06:00, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<06:51, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<05:22, 2.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:16<03:53, 3.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<16:40, 777kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<12:59, 997kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<09:24, 1.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<09:35, 1.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<09:20, 1.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<07:10, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<07:05, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<06:17, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:40, 2.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<06:16, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:00, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<05:33, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:55, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:13, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<03:58, 3.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:45, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:18, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<03:58, 3.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:44, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:28, 1.94MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:10, 2.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:37, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:14, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<03:59, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:41, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:16, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<03:57, 3.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:40, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:13, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:57, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:41, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:11, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:56, 3.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:38, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:27, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:03, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:40, 3.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<12:30, 970kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<09:59, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<07:17, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:55, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:59, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:06, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:24, 2.72MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<10:03, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<08:15, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:04, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:01, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:24, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:42, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:10, 2.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:17, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:20, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:43, 2.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:04, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:38, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:14, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:47, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:27:31, 134kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:02:25, 188kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<43:53, 266kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<33:21, 349kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<24:31, 475kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<17:25, 667kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<14:54, 777kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<12:46, 906kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<09:26, 1.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<06:44, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<10:29, 1.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<08:21, 1.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<06:07, 1.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:59, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:12, 1.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:37, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<04:02, 2.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<10:58:53, 17.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<7:41:57, 24.6kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<5:22:58, 35.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<3:47:55, 49.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<2:40:36, 70.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<1:52:27, 100kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<1:21:07, 139kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<59:03, 190kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<41:51, 268kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<30:59, 361kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<22:49, 489kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<16:12, 688kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<13:54, 798kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<10:51, 1.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:49, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:05, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:54, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:00, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:20, 2.53MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:59, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:13, 1.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:19, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:51, 2.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<29:06, 375kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<22:34, 483kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<16:20, 666kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<11:29, 943kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<1:27:07, 124kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<1:02:03, 174kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<43:36, 248kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<32:57, 327kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<24:08, 446kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<17:05, 628kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<14:26, 740kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<12:16, 871kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:03, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<06:25, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<16:07, 659kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<12:21, 859kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<08:51, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:40, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:12, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:16, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:04, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:20, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:59, 2.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:13, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:48, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:35, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:18, 3.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<10:00:04, 17.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<7:00:52, 24.6kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<4:54:05, 35.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<3:27:34, 49.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<2:27:20, 69.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:43:26, 99.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<1:12:19, 142kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<55:05, 185kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<39:35, 258kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<27:54, 365kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<21:50, 465kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<17:20, 585kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<12:38, 802kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<10:26, 965kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<09:04, 1.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:52, 1.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:57, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<07:23, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:10, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<04:32, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:30, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:41, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:29, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<02:34, 3.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<39:07, 252kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<28:22, 348kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<20:01, 491kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<16:18, 601kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<13:42, 715kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<10:09, 964kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<07:11, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<12:23, 786kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<09:48, 992kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<07:05, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<06:58, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:08, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:33, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<03:59, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<10:02, 956kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:57, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<05:48, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<04:11, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<09:48, 971kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<09:04, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:54, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<04:55, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<11:16, 839kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:59, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<06:32, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:32, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:40, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:12, 2.23MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:54, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:32, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:24, 2.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<02:55, 3.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<7:02:20, 21.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<4:55:49, 31.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<3:26:11, 44.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<2:30:37, 61.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<1:47:33, 85.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<1:15:43, 121kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<52:51, 173kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<44:24, 205kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<32:08, 283kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<22:42, 400kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<17:43, 510kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<13:27, 672kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<09:37, 936kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:36, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:13, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:10, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:27, 2.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:57, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:14, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:52, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:16, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:12, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<03:02, 2.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<08:37, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:02, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<05:08, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:25, 1.60MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:50, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:36, 2.41MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:19, 2.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:02, 2.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:56, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:08, 935kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:23, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:24, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:33, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:51, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:31, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:18, 2.55MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:46, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:19, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:14, 2.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:01, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:47, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:45, 2.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:45, 3.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:45, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:18, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:13, 2.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:58, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:44, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:51, 2.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:42, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:30, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:37, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:37, 3.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:56, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:30, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:46, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:00, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:25, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:18, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:00, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:39, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:39, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:38, 2.98MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<06:49, 1.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:41, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:12, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:33, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:01, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:53, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:49, 2.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<05:39, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:50, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:35, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:09, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:48, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:51, 2.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:35, 2.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:25, 2.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:35, 2.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:23, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:08, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:20, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:26, 3.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<07:52, 945kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:23, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:38, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:47, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:05, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:55, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:51, 2.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:25, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:56, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:55, 2.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:34, 2.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:21, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:31, 2.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<01:50, 3.89MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<1:49:12, 65.7kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<1:17:12, 92.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<54:04, 132kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<39:08, 181kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<29:02, 244kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<20:42, 342kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<14:29, 486kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<15:56, 441kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<11:55, 589kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<08:30, 823kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<07:27, 933kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:51, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:07, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:41, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:44, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:06, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:02, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:33, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:17, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:29, 2.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:09, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:47, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:58, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:14, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:10, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<02:52, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:12, 3.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<01:37, 4.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<45:19, 146kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<33:15, 199kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<23:37, 279kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<16:31, 397kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<16:59, 385kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<12:39, 517kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:01, 722kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<07:38, 848kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<06:05, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:24, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<03:09, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<17:01, 377kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<12:31, 511kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<08:53, 718kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<06:18, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<09:13, 687kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<07:47, 813kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:44, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:06, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:45, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:03, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:59, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:22, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:48, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:58, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:11, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:18, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:01, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:15, 2.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:51, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:25, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:41, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:57, 3.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:50, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<02:32, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:00, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:40, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:00, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<01:29, 3.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:11, 1.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:59, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:50, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:44, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:43, 860kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<05:17, 1.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<03:51, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:52, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:13, 1.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:23, 2.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:45, 3.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<07:56, 711kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<06:52, 821kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:04, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:36, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:49, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:54, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:53, 1.92MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<02:05, 2.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:03, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:49, 1.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:39, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:37, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:18, 861kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:00, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<03:36, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:41, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:50, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:56, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:07, 2.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:35, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:08, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:19, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:43, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:07, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:29, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<01:47, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:24, 953kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:23, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:11, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:17, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:25, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:37, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:53, 2.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:29, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:56, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:11, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:33, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:57, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:18, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:40, 2.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:17, 1.48MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:51, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:06, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:09<01:31, 3.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<45:22, 106kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<32:52, 146kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<23:11, 207kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<16:11, 294kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<13:03, 363kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<09:40, 489kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<06:51, 686kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<05:45, 811kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<05:07, 911kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:48, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:32, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:58, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:10, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<02:30, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:18, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:42, 2.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<01:15, 3.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:59, 893kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:58, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:53, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:00, 1.46MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:35, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:55, 2.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:13, 1.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:01, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:30, 2.84MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:59, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:22, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:52, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:22, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:11, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:02, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:31, 2.72MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:54, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:48, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:22, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:48, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:40, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:15, 3.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<00:56, 4.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:20, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<3:06:39, 21.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<2:10:28, 30.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<1:30:15, 43.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:07:38, 57.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<48:15, 80.9kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<33:54, 115kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<23:30, 164kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<19:34, 196kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<14:06, 272kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<09:53, 385kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<07:41, 490kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<06:14, 603kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:32, 825kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<03:12, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:36, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:57, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:15, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:25, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:52, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:22, 2.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:53, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:42, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:16, 2.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:39, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:33, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:11, 2.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:32, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:52, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:28, 2.32MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:06, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:33, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:28, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:06, 2.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:27, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:43, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:21, 2.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<00:58, 3.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:07, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:46, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:19, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:36, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:51, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:28, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:03, 2.92MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:16, 937kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:38, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:55, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:57, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:03, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:35, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:08, 2.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:55, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:40, 1.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:14, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:29, 1.92MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:42, 1.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:19, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:57, 2.92MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:35, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:23, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:01, 2.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:45, 3.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:38, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:28, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:51, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:19, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:54, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:38, 1.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:12, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:22, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:33, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:13, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:53, 2.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:24, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:17, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:57, 2.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:10, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:22, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:06, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:47, 3.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:33, 932kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:03, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:29, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:30, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:18, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:57, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:09, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:19, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:44, 2.95MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:34, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:20, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:59, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:07, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:13, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:57, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:40, 3.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:07, 957kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:42, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:14, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:16, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:20, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:03, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:44, 2.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:04, 910kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:40, 1.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:12, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:12, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:01, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:45, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:53, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:01, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:49, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:34, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:48, 933kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:25, 1.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:01, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:02, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:54, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:39, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:47, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:54, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:43, 2.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:30, 2.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:35, 922kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:17, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:55, 1.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:55, 1.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:57, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:43, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:31, 2.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:50, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:43, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:32, 2.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:38, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:44, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:34, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:24, 2.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:41, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:26, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:33, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:30, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:23, 2.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:35, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:28, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:19, 3.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:03, 941kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:50, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:36, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:36, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:31, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:23, 2.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:26, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:30, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:23, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:16, 2.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:49, 943kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:39, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:27, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:27, 1.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:28, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:22, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:15, 2.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:26, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:22, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:20, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:10, 2.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:52, 574kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:39, 759kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:27, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:22, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:18, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:11, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:04, 3.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:08, 1.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 3.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.41MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 757/400000 [00:00<00:52, 7567.73it/s]  0%|          | 1510/400000 [00:00<00:52, 7553.84it/s]  1%|          | 2242/400000 [00:00<00:53, 7479.74it/s]  1%|          | 2943/400000 [00:00<00:54, 7329.50it/s]  1%|          | 3663/400000 [00:00<00:54, 7289.19it/s]  1%|          | 4393/400000 [00:00<00:54, 7292.26it/s]  1%|▏         | 5133/400000 [00:00<00:53, 7322.60it/s]  1%|▏         | 5888/400000 [00:00<00:53, 7386.04it/s]  2%|▏         | 6654/400000 [00:00<00:52, 7464.96it/s]  2%|▏         | 7418/400000 [00:01<00:52, 7515.77it/s]  2%|▏         | 8180/400000 [00:01<00:51, 7545.72it/s]  2%|▏         | 8920/400000 [00:01<00:52, 7484.41it/s]  2%|▏         | 9659/400000 [00:01<00:52, 7392.28it/s]  3%|▎         | 10392/400000 [00:01<00:53, 7343.67it/s]  3%|▎         | 11133/400000 [00:01<00:52, 7362.10it/s]  3%|▎         | 11879/400000 [00:01<00:52, 7389.01it/s]  3%|▎         | 12631/400000 [00:01<00:52, 7425.95it/s]  3%|▎         | 13373/400000 [00:01<00:52, 7340.61it/s]  4%|▎         | 14118/400000 [00:01<00:52, 7372.43it/s]  4%|▎         | 14867/400000 [00:02<00:52, 7406.31it/s]  4%|▍         | 15624/400000 [00:02<00:51, 7454.66it/s]  4%|▍         | 16370/400000 [00:02<00:52, 7340.63it/s]  4%|▍         | 17105/400000 [00:02<00:52, 7322.54it/s]  4%|▍         | 17838/400000 [00:02<00:52, 7311.99it/s]  5%|▍         | 18589/400000 [00:02<00:51, 7368.57it/s]  5%|▍         | 19327/400000 [00:02<00:52, 7311.88it/s]  5%|▌         | 20059/400000 [00:02<00:51, 7307.38it/s]  5%|▌         | 20790/400000 [00:02<00:51, 7305.33it/s]  5%|▌         | 21559/400000 [00:02<00:51, 7416.08it/s]  6%|▌         | 22316/400000 [00:03<00:50, 7461.49it/s]  6%|▌         | 23063/400000 [00:03<00:50, 7394.75it/s]  6%|▌         | 23803/400000 [00:03<00:51, 7317.54it/s]  6%|▌         | 24571/400000 [00:03<00:50, 7420.35it/s]  6%|▋         | 25343/400000 [00:03<00:49, 7507.28it/s]  7%|▋         | 26106/400000 [00:03<00:49, 7541.31it/s]  7%|▋         | 26864/400000 [00:03<00:49, 7550.99it/s]  7%|▋         | 27620/400000 [00:03<00:49, 7500.41it/s]  7%|▋         | 28371/400000 [00:03<00:49, 7433.24it/s]  7%|▋         | 29115/400000 [00:03<00:49, 7425.78it/s]  7%|▋         | 29869/400000 [00:04<00:49, 7459.31it/s]  8%|▊         | 30637/400000 [00:04<00:49, 7521.61it/s]  8%|▊         | 31390/400000 [00:04<00:49, 7429.23it/s]  8%|▊         | 32134/400000 [00:04<00:49, 7427.85it/s]  8%|▊         | 32884/400000 [00:04<00:49, 7447.23it/s]  8%|▊         | 33635/400000 [00:04<00:49, 7463.88it/s]  9%|▊         | 34382/400000 [00:04<00:49, 7394.14it/s]  9%|▉         | 35133/400000 [00:04<00:49, 7425.88it/s]  9%|▉         | 35888/400000 [00:04<00:48, 7461.70it/s]  9%|▉         | 36655/400000 [00:04<00:48, 7522.09it/s]  9%|▉         | 37418/400000 [00:05<00:48, 7551.73it/s] 10%|▉         | 38183/400000 [00:05<00:47, 7580.07it/s] 10%|▉         | 38942/400000 [00:05<00:47, 7531.05it/s] 10%|▉         | 39696/400000 [00:05<00:48, 7463.11it/s] 10%|█         | 40443/400000 [00:05<00:48, 7358.23it/s] 10%|█         | 41180/400000 [00:05<00:49, 7216.92it/s] 10%|█         | 41903/400000 [00:05<00:49, 7208.72it/s] 11%|█         | 42625/400000 [00:05<00:49, 7210.87it/s] 11%|█         | 43364/400000 [00:05<00:49, 7261.19it/s] 11%|█         | 44091/400000 [00:05<00:49, 7205.56it/s] 11%|█         | 44812/400000 [00:06<00:49, 7170.44it/s] 11%|█▏        | 45545/400000 [00:06<00:49, 7217.45it/s] 12%|█▏        | 46308/400000 [00:06<00:48, 7334.89it/s] 12%|█▏        | 47057/400000 [00:06<00:47, 7377.90it/s] 12%|█▏        | 47823/400000 [00:06<00:47, 7457.88it/s] 12%|█▏        | 48587/400000 [00:06<00:46, 7511.27it/s] 12%|█▏        | 49352/400000 [00:06<00:46, 7550.13it/s] 13%|█▎        | 50108/400000 [00:06<00:46, 7510.20it/s] 13%|█▎        | 50868/400000 [00:06<00:46, 7535.31it/s] 13%|█▎        | 51639/400000 [00:06<00:45, 7584.67it/s] 13%|█▎        | 52398/400000 [00:07<00:46, 7545.48it/s] 13%|█▎        | 53153/400000 [00:07<00:46, 7502.70it/s] 13%|█▎        | 53904/400000 [00:07<00:47, 7300.43it/s] 14%|█▎        | 54661/400000 [00:07<00:46, 7376.82it/s] 14%|█▍        | 55417/400000 [00:07<00:46, 7429.38it/s] 14%|█▍        | 56191/400000 [00:07<00:45, 7519.58it/s] 14%|█▍        | 56966/400000 [00:07<00:45, 7586.41it/s] 14%|█▍        | 57726/400000 [00:07<00:45, 7532.75it/s] 15%|█▍        | 58480/400000 [00:07<00:45, 7532.94it/s] 15%|█▍        | 59239/400000 [00:07<00:45, 7548.27it/s] 15%|█▌        | 60018/400000 [00:08<00:44, 7617.27it/s] 15%|█▌        | 60797/400000 [00:08<00:44, 7665.96it/s] 15%|█▌        | 61564/400000 [00:08<00:44, 7644.90it/s] 16%|█▌        | 62329/400000 [00:08<00:44, 7637.66it/s] 16%|█▌        | 63101/400000 [00:08<00:43, 7659.68it/s] 16%|█▌        | 63882/400000 [00:08<00:43, 7702.58it/s] 16%|█▌        | 64659/400000 [00:08<00:43, 7720.52it/s] 16%|█▋        | 65432/400000 [00:08<00:43, 7702.17it/s] 17%|█▋        | 66203/400000 [00:08<00:43, 7598.92it/s] 17%|█▋        | 66980/400000 [00:08<00:43, 7647.87it/s] 17%|█▋        | 67759/400000 [00:09<00:43, 7689.93it/s] 17%|█▋        | 68538/400000 [00:09<00:42, 7719.12it/s] 17%|█▋        | 69311/400000 [00:09<00:43, 7686.58it/s] 18%|█▊        | 70080/400000 [00:09<00:43, 7627.63it/s] 18%|█▊        | 70844/400000 [00:09<00:43, 7591.72it/s] 18%|█▊        | 71604/400000 [00:09<00:43, 7558.35it/s] 18%|█▊        | 72366/400000 [00:09<00:43, 7575.24it/s] 18%|█▊        | 73124/400000 [00:09<00:43, 7455.13it/s] 18%|█▊        | 73889/400000 [00:09<00:43, 7509.81it/s] 19%|█▊        | 74660/400000 [00:10<00:42, 7567.89it/s] 19%|█▉        | 75418/400000 [00:10<00:43, 7542.61it/s] 19%|█▉        | 76196/400000 [00:10<00:42, 7609.75it/s] 19%|█▉        | 76958/400000 [00:10<00:42, 7581.07it/s] 19%|█▉        | 77735/400000 [00:10<00:42, 7636.08it/s] 20%|█▉        | 78512/400000 [00:10<00:41, 7673.99it/s] 20%|█▉        | 79280/400000 [00:10<00:41, 7669.32it/s] 20%|██        | 80052/400000 [00:10<00:41, 7683.90it/s] 20%|██        | 80821/400000 [00:10<00:42, 7566.78it/s] 20%|██        | 81579/400000 [00:10<00:42, 7521.48it/s] 21%|██        | 82332/400000 [00:11<00:42, 7470.71it/s] 21%|██        | 83106/400000 [00:11<00:41, 7549.15it/s] 21%|██        | 83877/400000 [00:11<00:41, 7595.41it/s] 21%|██        | 84637/400000 [00:11<00:42, 7457.63it/s] 21%|██▏       | 85388/400000 [00:11<00:42, 7472.03it/s] 22%|██▏       | 86136/400000 [00:11<00:43, 7209.04it/s] 22%|██▏       | 86901/400000 [00:11<00:42, 7334.74it/s] 22%|██▏       | 87638/400000 [00:11<00:42, 7343.88it/s] 22%|██▏       | 88374/400000 [00:11<00:42, 7298.68it/s] 22%|██▏       | 89106/400000 [00:11<00:42, 7282.15it/s] 22%|██▏       | 89873/400000 [00:12<00:41, 7391.56it/s] 23%|██▎       | 90643/400000 [00:12<00:41, 7480.32it/s] 23%|██▎       | 91392/400000 [00:12<00:41, 7478.03it/s] 23%|██▎       | 92141/400000 [00:12<00:41, 7458.18it/s] 23%|██▎       | 92888/400000 [00:12<00:41, 7368.12it/s] 23%|██▎       | 93633/400000 [00:12<00:41, 7390.97it/s] 24%|██▎       | 94375/400000 [00:12<00:41, 7398.16it/s] 24%|██▍       | 95116/400000 [00:12<00:41, 7337.71it/s] 24%|██▍       | 95851/400000 [00:12<00:41, 7321.79it/s] 24%|██▍       | 96614/400000 [00:12<00:40, 7410.74it/s] 24%|██▍       | 97359/400000 [00:13<00:40, 7420.36it/s] 25%|██▍       | 98102/400000 [00:13<00:40, 7418.18it/s] 25%|██▍       | 98864/400000 [00:13<00:40, 7477.24it/s] 25%|██▍       | 99612/400000 [00:13<00:40, 7461.70it/s] 25%|██▌       | 100371/400000 [00:13<00:39, 7497.65it/s] 25%|██▌       | 101140/400000 [00:13<00:39, 7552.28it/s] 25%|██▌       | 101905/400000 [00:13<00:39, 7579.14it/s] 26%|██▌       | 102675/400000 [00:13<00:39, 7613.82it/s] 26%|██▌       | 103437/400000 [00:13<00:39, 7466.50it/s] 26%|██▌       | 104185/400000 [00:13<00:39, 7405.02it/s] 26%|██▌       | 104944/400000 [00:14<00:39, 7456.67it/s] 26%|██▋       | 105691/400000 [00:14<00:40, 7297.02it/s] 27%|██▋       | 106422/400000 [00:14<00:40, 7272.98it/s] 27%|██▋       | 107174/400000 [00:14<00:39, 7344.36it/s] 27%|██▋       | 107914/400000 [00:14<00:39, 7359.62it/s] 27%|██▋       | 108651/400000 [00:14<00:39, 7361.27it/s] 27%|██▋       | 109407/400000 [00:14<00:39, 7417.21it/s] 28%|██▊       | 110150/400000 [00:14<00:39, 7322.45it/s] 28%|██▊       | 110890/400000 [00:14<00:39, 7344.21it/s] 28%|██▊       | 111652/400000 [00:14<00:38, 7422.42it/s] 28%|██▊       | 112396/400000 [00:15<00:38, 7426.14it/s] 28%|██▊       | 113168/400000 [00:15<00:38, 7511.43it/s] 28%|██▊       | 113935/400000 [00:15<00:37, 7557.80it/s] 29%|██▊       | 114700/400000 [00:15<00:37, 7583.75it/s] 29%|██▉       | 115459/400000 [00:15<00:37, 7529.04it/s] 29%|██▉       | 116227/400000 [00:15<00:37, 7571.48it/s] 29%|██▉       | 116985/400000 [00:15<00:37, 7509.96it/s] 29%|██▉       | 117737/400000 [00:15<00:37, 7482.01it/s] 30%|██▉       | 118486/400000 [00:15<00:38, 7392.88it/s] 30%|██▉       | 119226/400000 [00:15<00:38, 7322.05it/s] 30%|███       | 120002/400000 [00:16<00:37, 7446.20it/s] 30%|███       | 120751/400000 [00:16<00:37, 7456.87it/s] 30%|███       | 121521/400000 [00:16<00:37, 7525.93it/s] 31%|███       | 122275/400000 [00:16<00:36, 7521.41it/s] 31%|███       | 123028/400000 [00:16<00:37, 7477.98it/s] 31%|███       | 123777/400000 [00:16<00:36, 7478.54it/s] 31%|███       | 124526/400000 [00:16<00:36, 7471.52it/s] 31%|███▏      | 125274/400000 [00:16<00:36, 7437.01it/s] 32%|███▏      | 126018/400000 [00:16<00:37, 7354.22it/s] 32%|███▏      | 126754/400000 [00:17<00:37, 7195.94it/s] 32%|███▏      | 127487/400000 [00:17<00:37, 7235.35it/s] 32%|███▏      | 128247/400000 [00:17<00:37, 7338.80it/s] 32%|███▏      | 129010/400000 [00:17<00:36, 7421.50it/s] 32%|███▏      | 129764/400000 [00:17<00:36, 7455.30it/s] 33%|███▎      | 130511/400000 [00:17<00:36, 7340.92it/s] 33%|███▎      | 131246/400000 [00:17<00:36, 7327.10it/s] 33%|███▎      | 131993/400000 [00:17<00:36, 7368.51it/s] 33%|███▎      | 132744/400000 [00:17<00:36, 7408.50it/s] 33%|███▎      | 133495/400000 [00:17<00:35, 7438.19it/s] 34%|███▎      | 134240/400000 [00:18<00:35, 7399.54it/s] 34%|███▎      | 134986/400000 [00:18<00:35, 7414.68it/s] 34%|███▍      | 135753/400000 [00:18<00:35, 7487.78it/s] 34%|███▍      | 136521/400000 [00:18<00:34, 7543.18it/s] 34%|███▍      | 137287/400000 [00:18<00:34, 7577.33it/s] 35%|███▍      | 138045/400000 [00:18<00:35, 7394.38it/s] 35%|███▍      | 138786/400000 [00:18<00:35, 7361.73it/s] 35%|███▍      | 139540/400000 [00:18<00:35, 7413.12it/s] 35%|███▌      | 140313/400000 [00:18<00:34, 7505.25it/s] 35%|███▌      | 141065/400000 [00:18<00:34, 7503.75it/s] 35%|███▌      | 141816/400000 [00:19<00:34, 7422.70it/s] 36%|███▌      | 142567/400000 [00:19<00:34, 7446.07it/s] 36%|███▌      | 143333/400000 [00:19<00:34, 7507.77it/s] 36%|███▌      | 144106/400000 [00:19<00:33, 7570.45it/s] 36%|███▌      | 144864/400000 [00:19<00:33, 7562.88it/s] 36%|███▋      | 145621/400000 [00:19<00:33, 7517.04it/s] 37%|███▋      | 146373/400000 [00:19<00:33, 7509.25it/s] 37%|███▋      | 147125/400000 [00:19<00:33, 7469.29it/s] 37%|███▋      | 147873/400000 [00:19<00:34, 7401.44it/s] 37%|███▋      | 148638/400000 [00:19<00:33, 7472.07it/s] 37%|███▋      | 149398/400000 [00:20<00:33, 7509.07it/s] 38%|███▊      | 150162/400000 [00:20<00:33, 7545.82it/s] 38%|███▊      | 150935/400000 [00:20<00:32, 7597.97it/s] 38%|███▊      | 151710/400000 [00:20<00:32, 7642.84it/s] 38%|███▊      | 152488/400000 [00:20<00:32, 7683.09it/s] 38%|███▊      | 153257/400000 [00:20<00:32, 7632.97it/s] 39%|███▊      | 154021/400000 [00:20<00:32, 7584.80it/s] 39%|███▊      | 154801/400000 [00:20<00:32, 7647.64it/s] 39%|███▉      | 155567/400000 [00:20<00:31, 7643.37it/s] 39%|███▉      | 156332/400000 [00:20<00:32, 7591.85it/s] 39%|███▉      | 157092/400000 [00:21<00:32, 7568.19it/s] 39%|███▉      | 157849/400000 [00:21<00:32, 7553.86it/s] 40%|███▉      | 158624/400000 [00:21<00:31, 7609.48it/s] 40%|███▉      | 159401/400000 [00:21<00:31, 7655.80it/s] 40%|████      | 160174/400000 [00:21<00:31, 7677.80it/s] 40%|████      | 160942/400000 [00:21<00:31, 7541.47it/s] 40%|████      | 161697/400000 [00:21<00:31, 7517.12it/s] 41%|████      | 162450/400000 [00:21<00:31, 7470.78it/s] 41%|████      | 163227/400000 [00:21<00:31, 7557.04it/s] 41%|████      | 164004/400000 [00:21<00:30, 7618.51it/s] 41%|████      | 164767/400000 [00:22<00:30, 7593.60it/s] 41%|████▏     | 165540/400000 [00:22<00:30, 7631.50it/s] 42%|████▏     | 166304/400000 [00:22<00:30, 7590.50it/s] 42%|████▏     | 167085/400000 [00:22<00:30, 7654.85it/s] 42%|████▏     | 167866/400000 [00:22<00:30, 7698.49it/s] 42%|████▏     | 168637/400000 [00:22<00:30, 7670.29it/s] 42%|████▏     | 169414/400000 [00:22<00:29, 7697.57it/s] 43%|████▎     | 170185/400000 [00:22<00:29, 7699.16it/s] 43%|████▎     | 170964/400000 [00:22<00:29, 7725.71it/s] 43%|████▎     | 171737/400000 [00:22<00:29, 7726.93it/s] 43%|████▎     | 172510/400000 [00:23<00:29, 7668.08it/s] 43%|████▎     | 173289/400000 [00:23<00:29, 7702.20it/s] 44%|████▎     | 174070/400000 [00:23<00:29, 7733.09it/s] 44%|████▎     | 174844/400000 [00:23<00:29, 7732.97it/s] 44%|████▍     | 175618/400000 [00:23<00:29, 7732.42it/s] 44%|████▍     | 176392/400000 [00:23<00:29, 7630.37it/s] 44%|████▍     | 177156/400000 [00:23<00:29, 7583.77it/s] 44%|████▍     | 177920/400000 [00:23<00:29, 7598.16it/s] 45%|████▍     | 178681/400000 [00:23<00:29, 7425.55it/s] 45%|████▍     | 179438/400000 [00:23<00:29, 7467.27it/s] 45%|████▌     | 180186/400000 [00:24<00:29, 7416.00it/s] 45%|████▌     | 180929/400000 [00:24<00:29, 7405.30it/s] 45%|████▌     | 181697/400000 [00:24<00:29, 7485.54it/s] 46%|████▌     | 182470/400000 [00:24<00:28, 7554.85it/s] 46%|████▌     | 183252/400000 [00:24<00:28, 7630.52it/s] 46%|████▌     | 184016/400000 [00:24<00:28, 7593.03it/s] 46%|████▌     | 184787/400000 [00:24<00:28, 7626.59it/s] 46%|████▋     | 185561/400000 [00:24<00:27, 7659.96it/s] 47%|████▋     | 186336/400000 [00:24<00:27, 7683.80it/s] 47%|████▋     | 187112/400000 [00:24<00:27, 7704.28it/s] 47%|████▋     | 187883/400000 [00:25<00:27, 7591.72it/s] 47%|████▋     | 188660/400000 [00:25<00:27, 7642.28it/s] 47%|████▋     | 189436/400000 [00:25<00:27, 7676.10it/s] 48%|████▊     | 190214/400000 [00:25<00:27, 7706.97it/s] 48%|████▊     | 190986/400000 [00:25<00:27, 7709.80it/s] 48%|████▊     | 191758/400000 [00:25<00:27, 7629.36it/s] 48%|████▊     | 192530/400000 [00:25<00:27, 7655.42it/s] 48%|████▊     | 193306/400000 [00:25<00:26, 7686.26it/s] 49%|████▊     | 194075/400000 [00:25<00:26, 7681.88it/s] 49%|████▊     | 194850/400000 [00:25<00:26, 7699.57it/s] 49%|████▉     | 195621/400000 [00:26<00:27, 7568.71it/s] 49%|████▉     | 196396/400000 [00:26<00:26, 7620.40it/s] 49%|████▉     | 197167/400000 [00:26<00:26, 7645.80it/s] 49%|████▉     | 197938/400000 [00:26<00:26, 7663.65it/s] 50%|████▉     | 198711/400000 [00:26<00:26, 7682.52it/s] 50%|████▉     | 199480/400000 [00:26<00:26, 7623.49it/s] 50%|█████     | 200243/400000 [00:26<00:26, 7589.04it/s] 50%|█████     | 201003/400000 [00:26<00:26, 7556.29it/s] 50%|█████     | 201759/400000 [00:26<00:26, 7463.28it/s] 51%|█████     | 202520/400000 [00:27<00:26, 7502.25it/s] 51%|█████     | 203271/400000 [00:27<00:26, 7379.41it/s] 51%|█████     | 204010/400000 [00:27<00:26, 7294.01it/s] 51%|█████     | 204770/400000 [00:27<00:26, 7382.27it/s] 51%|█████▏    | 205537/400000 [00:27<00:26, 7464.30it/s] 52%|█████▏    | 206285/400000 [00:27<00:26, 7402.20it/s] 52%|█████▏    | 207026/400000 [00:27<00:26, 7389.81it/s] 52%|█████▏    | 207766/400000 [00:27<00:26, 7333.42it/s] 52%|█████▏    | 208500/400000 [00:27<00:26, 7324.19it/s] 52%|█████▏    | 209233/400000 [00:27<00:26, 7302.78it/s] 52%|█████▏    | 209983/400000 [00:28<00:25, 7358.51it/s] 53%|█████▎    | 210720/400000 [00:28<00:25, 7353.48it/s] 53%|█████▎    | 211493/400000 [00:28<00:25, 7461.64it/s] 53%|█████▎    | 212262/400000 [00:28<00:24, 7527.11it/s] 53%|█████▎    | 213029/400000 [00:28<00:24, 7567.46it/s] 53%|█████▎    | 213805/400000 [00:28<00:24, 7621.77it/s] 54%|█████▎    | 214568/400000 [00:28<00:24, 7596.50it/s] 54%|█████▍    | 215337/400000 [00:28<00:24, 7623.55it/s] 54%|█████▍    | 216100/400000 [00:28<00:24, 7538.18it/s] 54%|█████▍    | 216855/400000 [00:28<00:24, 7525.94it/s] 54%|█████▍    | 217608/400000 [00:29<00:24, 7516.88it/s] 55%|█████▍    | 218360/400000 [00:29<00:24, 7504.50it/s] 55%|█████▍    | 219114/400000 [00:29<00:24, 7513.14it/s] 55%|█████▍    | 219889/400000 [00:29<00:23, 7581.78it/s] 55%|█████▌    | 220648/400000 [00:29<00:23, 7578.71it/s] 55%|█████▌    | 221407/400000 [00:29<00:23, 7568.24it/s] 56%|█████▌    | 222164/400000 [00:29<00:23, 7544.35it/s] 56%|█████▌    | 222934/400000 [00:29<00:23, 7589.51it/s] 56%|█████▌    | 223694/400000 [00:29<00:23, 7524.37it/s] 56%|█████▌    | 224450/400000 [00:29<00:23, 7534.98it/s] 56%|█████▋    | 225215/400000 [00:30<00:23, 7567.07it/s] 56%|█████▋    | 225972/400000 [00:30<00:23, 7394.60it/s] 57%|█████▋    | 226713/400000 [00:30<00:23, 7369.29it/s] 57%|█████▋    | 227451/400000 [00:30<00:23, 7317.24it/s] 57%|█████▋    | 228186/400000 [00:30<00:23, 7325.53it/s] 57%|█████▋    | 228936/400000 [00:30<00:23, 7375.32it/s] 57%|█████▋    | 229691/400000 [00:30<00:22, 7425.11it/s] 58%|█████▊    | 230464/400000 [00:30<00:22, 7514.00it/s] 58%|█████▊    | 231238/400000 [00:30<00:22, 7580.35it/s] 58%|█████▊    | 231997/400000 [00:30<00:22, 7491.58it/s] 58%|█████▊    | 232762/400000 [00:31<00:22, 7537.88it/s] 58%|█████▊    | 233517/400000 [00:31<00:22, 7455.66it/s] 59%|█████▊    | 234289/400000 [00:31<00:22, 7531.01it/s] 59%|█████▉    | 235051/400000 [00:31<00:21, 7556.02it/s] 59%|█████▉    | 235818/400000 [00:31<00:21, 7589.52it/s] 59%|█████▉    | 236578/400000 [00:31<00:21, 7560.57it/s] 59%|█████▉    | 237335/400000 [00:31<00:21, 7495.10it/s] 60%|█████▉    | 238085/400000 [00:31<00:21, 7485.45it/s] 60%|█████▉    | 238834/400000 [00:31<00:21, 7474.14it/s] 60%|█████▉    | 239582/400000 [00:31<00:21, 7473.66it/s] 60%|██████    | 240330/400000 [00:32<00:21, 7444.71it/s] 60%|██████    | 241075/400000 [00:32<00:21, 7400.15it/s] 60%|██████    | 241816/400000 [00:32<00:21, 7331.15it/s] 61%|██████    | 242550/400000 [00:32<00:21, 7177.45it/s] 61%|██████    | 243304/400000 [00:32<00:21, 7279.90it/s] 61%|██████    | 244063/400000 [00:32<00:21, 7367.83it/s] 61%|██████    | 244801/400000 [00:32<00:21, 7323.12it/s] 61%|██████▏   | 245563/400000 [00:32<00:20, 7409.38it/s] 62%|██████▏   | 246328/400000 [00:32<00:20, 7478.52it/s] 62%|██████▏   | 247100/400000 [00:32<00:20, 7548.60it/s] 62%|██████▏   | 247878/400000 [00:33<00:19, 7616.38it/s] 62%|██████▏   | 248641/400000 [00:33<00:20, 7443.94it/s] 62%|██████▏   | 249387/400000 [00:33<00:20, 7418.47it/s] 63%|██████▎   | 250130/400000 [00:33<00:20, 7263.31it/s] 63%|██████▎   | 250883/400000 [00:33<00:20, 7338.95it/s] 63%|██████▎   | 251643/400000 [00:33<00:20, 7413.42it/s] 63%|██████▎   | 252386/400000 [00:33<00:19, 7414.13it/s] 63%|██████▎   | 253129/400000 [00:33<00:20, 7315.09it/s] 63%|██████▎   | 253876/400000 [00:33<00:19, 7359.76it/s] 64%|██████▎   | 254615/400000 [00:34<00:19, 7368.54it/s] 64%|██████▍   | 255394/400000 [00:34<00:19, 7488.00it/s] 64%|██████▍   | 256144/400000 [00:34<00:19, 7322.87it/s] 64%|██████▍   | 256924/400000 [00:34<00:19, 7457.63it/s] 64%|██████▍   | 257702/400000 [00:34<00:18, 7549.69it/s] 65%|██████▍   | 258473/400000 [00:34<00:18, 7596.59it/s] 65%|██████▍   | 259242/400000 [00:34<00:18, 7623.23it/s] 65%|██████▌   | 260006/400000 [00:34<00:18, 7598.12it/s] 65%|██████▌   | 260779/400000 [00:34<00:18, 7635.82it/s] 65%|██████▌   | 261561/400000 [00:34<00:18, 7687.96it/s] 66%|██████▌   | 262331/400000 [00:35<00:18, 7608.57it/s] 66%|██████▌   | 263093/400000 [00:35<00:18, 7564.45it/s] 66%|██████▌   | 263850/400000 [00:35<00:18, 7547.90it/s] 66%|██████▌   | 264606/400000 [00:35<00:18, 7489.39it/s] 66%|██████▋   | 265356/400000 [00:35<00:18, 7475.67it/s] 67%|██████▋   | 266130/400000 [00:35<00:17, 7552.78it/s] 67%|██████▋   | 266895/400000 [00:35<00:17, 7580.14it/s] 67%|██████▋   | 267654/400000 [00:35<00:17, 7422.19it/s] 67%|██████▋   | 268407/400000 [00:35<00:17, 7451.31it/s] 67%|██████▋   | 269155/400000 [00:35<00:17, 7457.79it/s] 67%|██████▋   | 269912/400000 [00:36<00:17, 7488.47it/s] 68%|██████▊   | 270662/400000 [00:36<00:17, 7468.42it/s] 68%|██████▊   | 271410/400000 [00:36<00:17, 7385.39it/s] 68%|██████▊   | 272162/400000 [00:36<00:17, 7423.10it/s] 68%|██████▊   | 272938/400000 [00:36<00:16, 7519.27it/s] 68%|██████▊   | 273712/400000 [00:36<00:16, 7584.07it/s] 69%|██████▊   | 274471/400000 [00:36<00:16, 7538.74it/s] 69%|██████▉   | 275226/400000 [00:36<00:16, 7469.39it/s] 69%|██████▉   | 275974/400000 [00:36<00:16, 7455.59it/s] 69%|██████▉   | 276738/400000 [00:36<00:16, 7507.89it/s] 69%|██████▉   | 277505/400000 [00:37<00:16, 7553.94it/s] 70%|██████▉   | 278276/400000 [00:37<00:16, 7599.56it/s] 70%|██████▉   | 279037/400000 [00:37<00:16, 7462.02it/s] 70%|██████▉   | 279787/400000 [00:37<00:16, 7472.03it/s] 70%|███████   | 280535/400000 [00:37<00:16, 7449.23it/s] 70%|███████   | 281281/400000 [00:37<00:16, 7415.58it/s] 71%|███████   | 282057/400000 [00:37<00:15, 7515.56it/s] 71%|███████   | 282810/400000 [00:37<00:15, 7514.85it/s] 71%|███████   | 283577/400000 [00:37<00:15, 7560.33it/s] 71%|███████   | 284355/400000 [00:37<00:15, 7624.32it/s] 71%|███████▏  | 285131/400000 [00:38<00:14, 7661.91it/s] 71%|███████▏  | 285905/400000 [00:38<00:14, 7684.21it/s] 72%|███████▏  | 286674/400000 [00:38<00:14, 7613.35it/s] 72%|███████▏  | 287437/400000 [00:38<00:14, 7616.36it/s] 72%|███████▏  | 288199/400000 [00:38<00:14, 7562.56it/s] 72%|███████▏  | 288956/400000 [00:38<00:14, 7525.37it/s] 72%|███████▏  | 289727/400000 [00:38<00:14, 7576.81it/s] 73%|███████▎  | 290485/400000 [00:38<00:14, 7536.54it/s] 73%|███████▎  | 291254/400000 [00:38<00:14, 7581.64it/s] 73%|███████▎  | 292033/400000 [00:38<00:14, 7641.83it/s] 73%|███████▎  | 292812/400000 [00:39<00:13, 7683.37it/s] 73%|███████▎  | 293589/400000 [00:39<00:13, 7708.00it/s] 74%|███████▎  | 294360/400000 [00:39<00:13, 7704.64it/s] 74%|███████▍  | 295131/400000 [00:39<00:13, 7636.56it/s] 74%|███████▍  | 295895/400000 [00:39<00:13, 7628.63it/s] 74%|███████▍  | 296659/400000 [00:39<00:13, 7586.96it/s] 74%|███████▍  | 297418/400000 [00:39<00:13, 7563.93it/s] 75%|███████▍  | 298175/400000 [00:39<00:13, 7530.24it/s] 75%|███████▍  | 298929/400000 [00:39<00:13, 7472.82it/s] 75%|███████▍  | 299677/400000 [00:39<00:13, 7427.21it/s] 75%|███████▌  | 300435/400000 [00:40<00:13, 7470.57it/s] 75%|███████▌  | 301183/400000 [00:40<00:13, 7439.57it/s] 75%|███████▌  | 301942/400000 [00:40<00:13, 7481.35it/s] 76%|███████▌  | 302691/400000 [00:40<00:13, 7443.63it/s] 76%|███████▌  | 303436/400000 [00:40<00:13, 7409.70it/s] 76%|███████▌  | 304190/400000 [00:40<00:12, 7447.65it/s] 76%|███████▌  | 304935/400000 [00:40<00:12, 7439.85it/s] 76%|███████▋  | 305684/400000 [00:40<00:12, 7454.40it/s] 77%|███████▋  | 306430/400000 [00:40<00:12, 7421.00it/s] 77%|███████▋  | 307173/400000 [00:40<00:12, 7409.50it/s] 77%|███████▋  | 307922/400000 [00:41<00:12, 7431.91it/s] 77%|███████▋  | 308666/400000 [00:41<00:12, 7397.28it/s] 77%|███████▋  | 309406/400000 [00:41<00:12, 7329.37it/s] 78%|███████▊  | 310140/400000 [00:41<00:12, 7255.25it/s] 78%|███████▊  | 310914/400000 [00:41<00:12, 7393.25it/s] 78%|███████▊  | 311688/400000 [00:41<00:11, 7491.49it/s] 78%|███████▊  | 312458/400000 [00:41<00:11, 7551.91it/s] 78%|███████▊  | 313226/400000 [00:41<00:11, 7589.65it/s] 78%|███████▊  | 313986/400000 [00:41<00:11, 7528.20it/s] 79%|███████▊  | 314764/400000 [00:41<00:11, 7600.97it/s] 79%|███████▉  | 315525/400000 [00:42<00:11, 7588.55it/s] 79%|███████▉  | 316292/400000 [00:42<00:10, 7611.67it/s] 79%|███████▉  | 317054/400000 [00:42<00:10, 7570.12it/s] 79%|███████▉  | 317812/400000 [00:42<00:10, 7521.81it/s] 80%|███████▉  | 318565/400000 [00:42<00:10, 7492.58it/s] 80%|███████▉  | 319333/400000 [00:42<00:10, 7547.78it/s] 80%|████████  | 320089/400000 [00:42<00:10, 7518.72it/s] 80%|████████  | 320842/400000 [00:42<00:10, 7456.09it/s] 80%|████████  | 321588/400000 [00:42<00:10, 7414.22it/s] 81%|████████  | 322341/400000 [00:43<00:10, 7448.52it/s] 81%|████████  | 323109/400000 [00:43<00:10, 7513.93it/s] 81%|████████  | 323879/400000 [00:43<00:10, 7568.71it/s] 81%|████████  | 324642/400000 [00:43<00:09, 7586.76it/s] 81%|████████▏ | 325401/400000 [00:43<00:10, 7458.13it/s] 82%|████████▏ | 326153/400000 [00:43<00:09, 7476.35it/s] 82%|████████▏ | 326927/400000 [00:43<00:09, 7553.02it/s] 82%|████████▏ | 327703/400000 [00:43<00:09, 7613.08it/s] 82%|████████▏ | 328465/400000 [00:43<00:09, 7593.20it/s] 82%|████████▏ | 329225/400000 [00:43<00:09, 7536.45it/s] 82%|████████▏ | 329980/400000 [00:44<00:09, 7540.18it/s] 83%|████████▎ | 330742/400000 [00:44<00:09, 7561.06it/s] 83%|████████▎ | 331499/400000 [00:44<00:09, 7518.11it/s] 83%|████████▎ | 332251/400000 [00:44<00:09, 7436.82it/s] 83%|████████▎ | 333015/400000 [00:44<00:08, 7495.33it/s] 83%|████████▎ | 333765/400000 [00:44<00:08, 7393.50it/s] 84%|████████▎ | 334513/400000 [00:44<00:08, 7418.72it/s] 84%|████████▍ | 335256/400000 [00:44<00:08, 7406.58it/s] 84%|████████▍ | 335997/400000 [00:44<00:08, 7390.16it/s] 84%|████████▍ | 336764/400000 [00:44<00:08, 7471.29it/s] 84%|████████▍ | 337537/400000 [00:45<00:08, 7544.39it/s] 85%|████████▍ | 338313/400000 [00:45<00:08, 7607.08it/s] 85%|████████▍ | 339086/400000 [00:45<00:07, 7641.43it/s] 85%|████████▍ | 339851/400000 [00:45<00:07, 7564.44it/s] 85%|████████▌ | 340625/400000 [00:45<00:07, 7605.05it/s] 85%|████████▌ | 341398/400000 [00:45<00:07, 7639.72it/s] 86%|████████▌ | 342174/400000 [00:45<00:07, 7672.83it/s] 86%|████████▌ | 342951/400000 [00:45<00:07, 7699.04it/s] 86%|████████▌ | 343722/400000 [00:45<00:07, 7576.12it/s] 86%|████████▌ | 344481/400000 [00:45<00:07, 7507.93it/s] 86%|████████▋ | 345244/400000 [00:46<00:07, 7543.48it/s] 87%|████████▋ | 346015/400000 [00:46<00:07, 7591.64it/s] 87%|████████▋ | 346789/400000 [00:46<00:06, 7635.48it/s] 87%|████████▋ | 347553/400000 [00:46<00:06, 7593.40it/s] 87%|████████▋ | 348329/400000 [00:46<00:06, 7641.20it/s] 87%|████████▋ | 349094/400000 [00:46<00:06, 7636.37it/s] 87%|████████▋ | 349858/400000 [00:46<00:06, 7547.26it/s] 88%|████████▊ | 350614/400000 [00:46<00:06, 7512.80it/s] 88%|████████▊ | 351366/400000 [00:46<00:06, 7419.74it/s] 88%|████████▊ | 352117/400000 [00:46<00:06, 7446.38it/s] 88%|████████▊ | 352871/400000 [00:47<00:06, 7472.13it/s] 88%|████████▊ | 353644/400000 [00:47<00:06, 7544.80it/s] 89%|████████▊ | 354405/400000 [00:47<00:06, 7563.48it/s] 89%|████████▉ | 355162/400000 [00:47<00:06, 7404.26it/s] 89%|████████▉ | 355910/400000 [00:47<00:05, 7426.28it/s] 89%|████████▉ | 356654/400000 [00:47<00:05, 7379.83it/s] 89%|████████▉ | 357406/400000 [00:47<00:05, 7419.04it/s] 90%|████████▉ | 358179/400000 [00:47<00:05, 7509.42it/s] 90%|████████▉ | 358937/400000 [00:47<00:05, 7529.46it/s] 90%|████████▉ | 359699/400000 [00:47<00:05, 7555.55it/s] 90%|█████████ | 360455/400000 [00:48<00:05, 7484.19it/s] 90%|█████████ | 361232/400000 [00:48<00:05, 7566.02it/s] 90%|█████████ | 361990/400000 [00:48<00:05, 7378.11it/s] 91%|█████████ | 362730/400000 [00:48<00:05, 7356.45it/s] 91%|█████████ | 363472/400000 [00:48<00:04, 7372.77it/s] 91%|█████████ | 364231/400000 [00:48<00:04, 7434.09it/s] 91%|█████████ | 364999/400000 [00:48<00:04, 7504.80it/s] 91%|█████████▏| 365751/400000 [00:48<00:04, 7396.61it/s] 92%|█████████▏| 366492/400000 [00:48<00:04, 7378.96it/s] 92%|█████████▏| 367231/400000 [00:48<00:04, 7235.96it/s] 92%|█████████▏| 367956/400000 [00:49<00:04, 7152.49it/s] 92%|█████████▏| 368673/400000 [00:49<00:04, 7113.21it/s] 92%|█████████▏| 369386/400000 [00:49<00:04, 7109.10it/s] 93%|█████████▎| 370112/400000 [00:49<00:04, 7152.01it/s] 93%|█████████▎| 370884/400000 [00:49<00:03, 7312.45it/s] 93%|█████████▎| 371654/400000 [00:49<00:03, 7422.05it/s] 93%|█████████▎| 372425/400000 [00:49<00:03, 7505.72it/s] 93%|█████████▎| 373199/400000 [00:49<00:03, 7571.73it/s] 93%|█████████▎| 373958/400000 [00:49<00:03, 7537.78it/s] 94%|█████████▎| 374734/400000 [00:49<00:03, 7602.16it/s] 94%|█████████▍| 375507/400000 [00:50<00:03, 7637.90it/s] 94%|█████████▍| 376272/400000 [00:50<00:03, 7444.98it/s] 94%|█████████▍| 377018/400000 [00:50<00:03, 7305.89it/s] 94%|█████████▍| 377751/400000 [00:50<00:03, 7215.40it/s] 95%|█████████▍| 378474/400000 [00:50<00:03, 7014.27it/s] 95%|█████████▍| 379238/400000 [00:50<00:02, 7189.62it/s] 95%|█████████▌| 380008/400000 [00:50<00:02, 7335.41it/s] 95%|█████████▌| 380775/400000 [00:50<00:02, 7432.36it/s] 95%|█████████▌| 381529/400000 [00:50<00:02, 7462.52it/s] 96%|█████████▌| 382301/400000 [00:51<00:02, 7537.09it/s] 96%|█████████▌| 383070/400000 [00:51<00:02, 7582.25it/s] 96%|█████████▌| 383847/400000 [00:51<00:02, 7635.65it/s] 96%|█████████▌| 384625/400000 [00:51<00:02, 7677.63it/s] 96%|█████████▋| 385394/400000 [00:51<00:01, 7450.07it/s] 97%|█████████▋| 386164/400000 [00:51<00:01, 7523.21it/s] 97%|█████████▋| 386927/400000 [00:51<00:01, 7554.57it/s] 97%|█████████▋| 387684/400000 [00:51<00:01, 7460.43it/s] 97%|█████████▋| 388432/400000 [00:51<00:01, 7358.83it/s] 97%|█████████▋| 389185/400000 [00:51<00:01, 7408.11it/s] 97%|█████████▋| 389953/400000 [00:52<00:01, 7486.91it/s] 98%|█████████▊| 390722/400000 [00:52<00:01, 7546.14it/s] 98%|█████████▊| 391478/400000 [00:52<00:01, 7542.08it/s] 98%|█████████▊| 392253/400000 [00:52<00:01, 7602.37it/s] 98%|█████████▊| 393014/400000 [00:52<00:00, 7566.34it/s] 98%|█████████▊| 393771/400000 [00:52<00:00, 7300.12it/s] 99%|█████████▊| 394504/400000 [00:52<00:00, 7129.09it/s] 99%|█████████▉| 395220/400000 [00:52<00:00, 7045.40it/s] 99%|█████████▉| 395990/400000 [00:52<00:00, 7227.78it/s] 99%|█████████▉| 396716/400000 [00:52<00:00, 7148.84it/s] 99%|█████████▉| 397461/400000 [00:53<00:00, 7236.18it/s]100%|█████████▉| 398233/400000 [00:53<00:00, 7374.41it/s]100%|█████████▉| 399004/400000 [00:53<00:00, 7470.06it/s]100%|█████████▉| 399771/400000 [00:53<00:00, 7526.41it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7489.02it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f20c62de400> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011255841398443787 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.012016323497861523 	 Accuracy: 53

  model saves at 53% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15868 out of table with 15567 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15868 out of table with 15567 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 


  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/85948faac40e5bea0f7d8209fd6131b3e186f819', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '85948faac40e5bea0f7d8209fd6131b3e186f819', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/85948faac40e5bea0f7d8209fd6131b3e186f819

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/85948faac40e5bea0f7d8209fd6131b3e186f819

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f0e1a3de4a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 13:13:30.160454
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 13:13:30.165085
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 13:13:30.168931
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 13:13:30.173007
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0e06969b00> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356127.6562
Epoch 2/10

1/1 [==============================] - 0s 91ms/step - loss: 298700.6562
Epoch 3/10

1/1 [==============================] - 0s 87ms/step - loss: 211650.9688
Epoch 4/10

1/1 [==============================] - 0s 87ms/step - loss: 138661.0156
Epoch 5/10

1/1 [==============================] - 0s 91ms/step - loss: 83245.1953
Epoch 6/10

1/1 [==============================] - 0s 109ms/step - loss: 48336.0391
Epoch 7/10

1/1 [==============================] - 0s 89ms/step - loss: 28732.3477
Epoch 8/10

1/1 [==============================] - 0s 86ms/step - loss: 17839.0742
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 11826.9336
Epoch 10/10

1/1 [==============================] - 0s 89ms/step - loss: 8415.3467

  #### Inference Need return ypred, ytrue ######################### 
[[-0.9417415   1.2336869   0.360273   -1.2467016   0.740266   -1.3968873
   0.07607523 -1.2476275   0.4623546   1.1827929  -0.45122182  0.9037042
  -1.2210447   0.6542914  -1.2959149  -0.9312385   0.6073779  -0.09648003
   1.2160234   1.5459166  -0.77106917 -0.06089854 -2.3690248   0.68654233
  -0.07298636 -0.27249902  0.21494597 -0.618662    0.9825044  -0.16777311
   0.49692345  1.713992   -0.25765464 -0.62391764  0.8772434  -0.51424503
  -0.12123722 -0.01268778  0.6686556  -1.063288   -0.29637194 -0.31422687
  -0.70525044  0.45747033  1.8969573  -0.07944763 -1.0693063   0.9275412
  -0.13174963  1.3220341   0.24845397 -0.45229828 -1.2577984   0.7115478
  -1.4304135  -0.14185724  1.0431023  -1.0492723   0.3332464  -0.5248809
   0.11605164  5.695339    8.512617    6.3869934   7.789658    6.9742923
   6.2734447   6.736082    7.0411615   7.6009936   8.155515    8.23652
   5.9447503   6.7556615   7.027528    6.285378    6.3754816   5.792196
   6.672266    7.194158    7.5189323   5.3697433   7.512327    6.55418
   7.7964544   7.458905    7.181938    8.290998    6.911885    6.4184475
   7.3677006   8.807277    5.671239    7.3847685   8.39709     6.6179996
   7.534445    5.3174124   7.9812827   6.0568194   6.582349    8.541655
   7.199807    5.800968    6.8666935   6.863675    7.4035664   5.8477693
   7.931637    8.7964115   6.753642    6.7630415   6.4521456   6.710967
   7.2863812   7.567624    7.569809    7.882673    7.2920084   6.3315415
  -1.516415   -0.57165575 -0.471186    0.78145194 -0.56479156  1.0382348
  -0.4415277  -0.48926336  0.6002952   0.48664027  1.4879255  -0.2674908
  -0.53542244  0.04180645 -0.6843142  -0.9362745   1.5609232   0.6349495
   2.7211537  -0.3018453  -0.41221157 -0.863777   -1.296078   -0.12260509
   0.45079708 -0.3038825  -0.43547237 -0.48149467 -0.08275727 -0.922431
  -0.5463759  -0.8177146   1.3285289   0.7695676  -0.3444406   0.06750653
   0.7969482   1.0537152   0.39898512  0.5052742  -1.188734    0.18950272
   0.94779676 -1.0984684   1.7276185  -1.0397086   0.4379525  -0.5173456
  -0.29249102 -0.64082503 -0.9484465   0.01492339 -1.4450691  -1.2922441
   0.7280368   0.64635456 -0.3988852   0.43224686  0.22819385  0.05949073
   1.7497123   0.5550617   2.547831    0.8192519   0.33413815  0.38029045
   0.59107935  0.93109167  1.8578689   0.8873077   0.45777047  1.933088
   0.7427223   0.7769687   0.5496807   1.5367253   0.9501423   1.7844585
   0.3422464   1.2953489   0.69832456  0.2442522   0.7255439   1.7349256
   0.18599474  1.1513059   0.22616106  1.4909046   0.1861676   0.29724228
   0.473001    2.3344207   0.34574825  0.14826983  0.57782155  0.93354195
   2.8747673   1.2807989   2.1072283   1.9107735   0.6391821   1.5093067
   2.3797555   1.5246797   1.3974712   0.44399953  0.68563664  1.1351352
   0.4752183   2.0926814   1.3874707   0.31753033  2.1990705   0.53805965
   3.1429315   0.21454918  0.31583846  0.26880044  0.9835582   0.56248796
   0.09206879  8.456297    6.513373    8.03491     7.8391438   7.6334186
   6.3608527   8.452136    8.6846285   7.8817596   7.5723076   7.7374206
   8.269344    7.668618    6.1807404   6.789882    8.674355    7.1665173
   8.849858    7.039187    7.609876    9.171259    7.75705     8.389384
   7.635347    7.2282186   7.799526    7.4028206   6.2858925   7.7445865
   8.399667    7.498035    6.2739143   8.735185    7.319923    7.35954
   7.112777    6.3389387   7.7536607   6.6319013   8.804264    8.996022
   7.706805    7.584623    7.1045694   6.9213934   6.901916    7.104947
   7.2997904   6.8032446   7.8283424   8.299626    8.4655285   6.7241287
   8.319781    8.07359     6.9976373   6.9087706   7.271629    8.4615755
   1.4569871   0.96221644  1.363678    0.2607212   0.71962464  1.959107
   1.3675268   1.1024925   0.70961845  0.32486862  1.3596735   0.96038526
   1.4986042   0.89086837  1.0366513   0.47811186  1.7561721   0.89879894
   0.5477338   0.5122799   1.6306577   2.4607677   2.70195     0.55693835
   2.0703015   0.45119143  1.4761953   0.07815027  1.3995008   1.7852938
   0.26831007  2.9567719   1.8312644   1.0442443   0.66930103  1.8359876
   2.1455345   0.68049467  2.2948985   1.5113822   0.99675405  1.3457484
   2.3045707   1.278039    0.16430897  1.801246    0.35995495  1.6513807
   0.7056713   0.12509906  1.783509    1.057457    0.3086077   1.4874734
   0.2505057   1.8537397   0.3144102   1.0505288   1.852695    0.80703115
  -4.216258   12.672932   -6.896558  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 13:13:38.857966
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4377
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 13:13:38.862189
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9126.66
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 13:13:38.865839
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4075
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 13:13:38.869519
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -816.357
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139697857655304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139696916271904
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139696916272408
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139696916272912
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139696916273416
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139696916273920

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f0df345e940> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.623838
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.594192
grad_step = 000002, loss = 0.576994
grad_step = 000003, loss = 0.557406
grad_step = 000004, loss = 0.535152
grad_step = 000005, loss = 0.513277
grad_step = 000006, loss = 0.497425
grad_step = 000007, loss = 0.490136
grad_step = 000008, loss = 0.480749
grad_step = 000009, loss = 0.464197
grad_step = 000010, loss = 0.448383
grad_step = 000011, loss = 0.435414
grad_step = 000012, loss = 0.423903
grad_step = 000013, loss = 0.412230
grad_step = 000014, loss = 0.398959
grad_step = 000015, loss = 0.385343
grad_step = 000016, loss = 0.372495
grad_step = 000017, loss = 0.361340
grad_step = 000018, loss = 0.351298
grad_step = 000019, loss = 0.340139
grad_step = 000020, loss = 0.328422
grad_step = 000021, loss = 0.317810
grad_step = 000022, loss = 0.308126
grad_step = 000023, loss = 0.298342
grad_step = 000024, loss = 0.288154
grad_step = 000025, loss = 0.278189
grad_step = 000026, loss = 0.269073
grad_step = 000027, loss = 0.260220
grad_step = 000028, loss = 0.250819
grad_step = 000029, loss = 0.241363
grad_step = 000030, loss = 0.232539
grad_step = 000031, loss = 0.224162
grad_step = 000032, loss = 0.215767
grad_step = 000033, loss = 0.207399
grad_step = 000034, loss = 0.199507
grad_step = 000035, loss = 0.192161
grad_step = 000036, loss = 0.184846
grad_step = 000037, loss = 0.177426
grad_step = 000038, loss = 0.170303
grad_step = 000039, loss = 0.163538
grad_step = 000040, loss = 0.156841
grad_step = 000041, loss = 0.150209
grad_step = 000042, loss = 0.143944
grad_step = 000043, loss = 0.138024
grad_step = 000044, loss = 0.132121
grad_step = 000045, loss = 0.126352
grad_step = 000046, loss = 0.120953
grad_step = 000047, loss = 0.115747
grad_step = 000048, loss = 0.110595
grad_step = 000049, loss = 0.105662
grad_step = 000050, loss = 0.100984
grad_step = 000051, loss = 0.096379
grad_step = 000052, loss = 0.091918
grad_step = 000053, loss = 0.087724
grad_step = 000054, loss = 0.083678
grad_step = 000055, loss = 0.079732
grad_step = 000056, loss = 0.075992
grad_step = 000057, loss = 0.072421
grad_step = 000058, loss = 0.068936
grad_step = 000059, loss = 0.065626
grad_step = 000060, loss = 0.062479
grad_step = 000061, loss = 0.059425
grad_step = 000062, loss = 0.056514
grad_step = 000063, loss = 0.053756
grad_step = 000064, loss = 0.051092
grad_step = 000065, loss = 0.048550
grad_step = 000066, loss = 0.046140
grad_step = 000067, loss = 0.043822
grad_step = 000068, loss = 0.041613
grad_step = 000069, loss = 0.039522
grad_step = 000070, loss = 0.037517
grad_step = 000071, loss = 0.035607
grad_step = 000072, loss = 0.033798
grad_step = 000073, loss = 0.032069
grad_step = 000074, loss = 0.030425
grad_step = 000075, loss = 0.028864
grad_step = 000076, loss = 0.027371
grad_step = 000077, loss = 0.025953
grad_step = 000078, loss = 0.024612
grad_step = 000079, loss = 0.023335
grad_step = 000080, loss = 0.022127
grad_step = 000081, loss = 0.020982
grad_step = 000082, loss = 0.019888
grad_step = 000083, loss = 0.018856
grad_step = 000084, loss = 0.017874
grad_step = 000085, loss = 0.016944
grad_step = 000086, loss = 0.016066
grad_step = 000087, loss = 0.015231
grad_step = 000088, loss = 0.014448
grad_step = 000089, loss = 0.013710
grad_step = 000090, loss = 0.013014
grad_step = 000091, loss = 0.012360
grad_step = 000092, loss = 0.011743
grad_step = 000093, loss = 0.011163
grad_step = 000094, loss = 0.010617
grad_step = 000095, loss = 0.010101
grad_step = 000096, loss = 0.009615
grad_step = 000097, loss = 0.009155
grad_step = 000098, loss = 0.008722
grad_step = 000099, loss = 0.008314
grad_step = 000100, loss = 0.007929
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.007566
grad_step = 000102, loss = 0.007222
grad_step = 000103, loss = 0.006897
grad_step = 000104, loss = 0.006591
grad_step = 000105, loss = 0.006303
grad_step = 000106, loss = 0.006032
grad_step = 000107, loss = 0.005777
grad_step = 000108, loss = 0.005537
grad_step = 000109, loss = 0.005312
grad_step = 000110, loss = 0.005101
grad_step = 000111, loss = 0.004901
grad_step = 000112, loss = 0.004709
grad_step = 000113, loss = 0.004526
grad_step = 000114, loss = 0.004354
grad_step = 000115, loss = 0.004194
grad_step = 000116, loss = 0.004047
grad_step = 000117, loss = 0.003910
grad_step = 000118, loss = 0.003781
grad_step = 000119, loss = 0.003662
grad_step = 000120, loss = 0.003549
grad_step = 000121, loss = 0.003442
grad_step = 000122, loss = 0.003337
grad_step = 000123, loss = 0.003236
grad_step = 000124, loss = 0.003139
grad_step = 000125, loss = 0.003049
grad_step = 000126, loss = 0.002965
grad_step = 000127, loss = 0.002888
grad_step = 000128, loss = 0.002818
grad_step = 000129, loss = 0.002753
grad_step = 000130, loss = 0.002695
grad_step = 000131, loss = 0.002644
grad_step = 000132, loss = 0.002608
grad_step = 000133, loss = 0.002587
grad_step = 000134, loss = 0.002591
grad_step = 000135, loss = 0.002559
grad_step = 000136, loss = 0.002483
grad_step = 000137, loss = 0.002364
grad_step = 000138, loss = 0.002311
grad_step = 000139, loss = 0.002325
grad_step = 000140, loss = 0.002315
grad_step = 000141, loss = 0.002253
grad_step = 000142, loss = 0.002180
grad_step = 000143, loss = 0.002163
grad_step = 000144, loss = 0.002173
grad_step = 000145, loss = 0.002142
grad_step = 000146, loss = 0.002086
grad_step = 000147, loss = 0.002054
grad_step = 000148, loss = 0.002054
grad_step = 000149, loss = 0.002050
grad_step = 000150, loss = 0.002015
grad_step = 000151, loss = 0.001978
grad_step = 000152, loss = 0.001964
grad_step = 000153, loss = 0.001963
grad_step = 000154, loss = 0.001954
grad_step = 000155, loss = 0.001928
grad_step = 000156, loss = 0.001902
grad_step = 000157, loss = 0.001889
grad_step = 000158, loss = 0.001885
grad_step = 000159, loss = 0.001881
grad_step = 000160, loss = 0.001868
grad_step = 000161, loss = 0.001850
grad_step = 000162, loss = 0.001832
grad_step = 000163, loss = 0.001820
grad_step = 000164, loss = 0.001813
grad_step = 000165, loss = 0.001808
grad_step = 000166, loss = 0.001805
grad_step = 000167, loss = 0.001800
grad_step = 000168, loss = 0.001794
grad_step = 000169, loss = 0.001785
grad_step = 000170, loss = 0.001776
grad_step = 000171, loss = 0.001766
grad_step = 000172, loss = 0.001756
grad_step = 000173, loss = 0.001746
grad_step = 000174, loss = 0.001738
grad_step = 000175, loss = 0.001730
grad_step = 000176, loss = 0.001725
grad_step = 000177, loss = 0.001720
grad_step = 000178, loss = 0.001720
grad_step = 000179, loss = 0.001723
grad_step = 000180, loss = 0.001738
grad_step = 000181, loss = 0.001763
grad_step = 000182, loss = 0.001815
grad_step = 000183, loss = 0.001852
grad_step = 000184, loss = 0.001881
grad_step = 000185, loss = 0.001800
grad_step = 000186, loss = 0.001703
grad_step = 000187, loss = 0.001652
grad_step = 000188, loss = 0.001684
grad_step = 000189, loss = 0.001741
grad_step = 000190, loss = 0.001734
grad_step = 000191, loss = 0.001680
grad_step = 000192, loss = 0.001634
grad_step = 000193, loss = 0.001644
grad_step = 000194, loss = 0.001681
grad_step = 000195, loss = 0.001682
grad_step = 000196, loss = 0.001650
grad_step = 000197, loss = 0.001618
grad_step = 000198, loss = 0.001619
grad_step = 000199, loss = 0.001640
grad_step = 000200, loss = 0.001647
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001630
grad_step = 000202, loss = 0.001606
grad_step = 000203, loss = 0.001598
grad_step = 000204, loss = 0.001607
grad_step = 000205, loss = 0.001617
grad_step = 000206, loss = 0.001616
grad_step = 000207, loss = 0.001604
grad_step = 000208, loss = 0.001590
grad_step = 000209, loss = 0.001583
grad_step = 000210, loss = 0.001585
grad_step = 000211, loss = 0.001591
grad_step = 000212, loss = 0.001595
grad_step = 000213, loss = 0.001595
grad_step = 000214, loss = 0.001589
grad_step = 000215, loss = 0.001582
grad_step = 000216, loss = 0.001575
grad_step = 000217, loss = 0.001570
grad_step = 000218, loss = 0.001568
grad_step = 000219, loss = 0.001568
grad_step = 000220, loss = 0.001569
grad_step = 000221, loss = 0.001571
grad_step = 000222, loss = 0.001574
grad_step = 000223, loss = 0.001577
grad_step = 000224, loss = 0.001582
grad_step = 000225, loss = 0.001588
grad_step = 000226, loss = 0.001599
grad_step = 000227, loss = 0.001613
grad_step = 000228, loss = 0.001631
grad_step = 000229, loss = 0.001642
grad_step = 000230, loss = 0.001639
grad_step = 000231, loss = 0.001619
grad_step = 000232, loss = 0.001600
grad_step = 000233, loss = 0.001585
grad_step = 000234, loss = 0.001574
grad_step = 000235, loss = 0.001558
grad_step = 000236, loss = 0.001551
grad_step = 000237, loss = 0.001559
grad_step = 000238, loss = 0.001574
grad_step = 000239, loss = 0.001587
grad_step = 000240, loss = 0.001585
grad_step = 000241, loss = 0.001590
grad_step = 000242, loss = 0.001598
grad_step = 000243, loss = 0.001606
grad_step = 000244, loss = 0.001604
grad_step = 000245, loss = 0.001586
grad_step = 000246, loss = 0.001568
grad_step = 000247, loss = 0.001559
grad_step = 000248, loss = 0.001550
grad_step = 000249, loss = 0.001542
grad_step = 000250, loss = 0.001538
grad_step = 000251, loss = 0.001545
grad_step = 000252, loss = 0.001556
grad_step = 000253, loss = 0.001561
grad_step = 000254, loss = 0.001564
grad_step = 000255, loss = 0.001562
grad_step = 000256, loss = 0.001563
grad_step = 000257, loss = 0.001559
grad_step = 000258, loss = 0.001552
grad_step = 000259, loss = 0.001542
grad_step = 000260, loss = 0.001536
grad_step = 000261, loss = 0.001531
grad_step = 000262, loss = 0.001528
grad_step = 000263, loss = 0.001526
grad_step = 000264, loss = 0.001525
grad_step = 000265, loss = 0.001526
grad_step = 000266, loss = 0.001529
grad_step = 000267, loss = 0.001533
grad_step = 000268, loss = 0.001537
grad_step = 000269, loss = 0.001545
grad_step = 000270, loss = 0.001555
grad_step = 000271, loss = 0.001573
grad_step = 000272, loss = 0.001592
grad_step = 000273, loss = 0.001616
grad_step = 000274, loss = 0.001619
grad_step = 000275, loss = 0.001614
grad_step = 000276, loss = 0.001576
grad_step = 000277, loss = 0.001537
grad_step = 000278, loss = 0.001514
grad_step = 000279, loss = 0.001519
grad_step = 000280, loss = 0.001539
grad_step = 000281, loss = 0.001555
grad_step = 000282, loss = 0.001562
grad_step = 000283, loss = 0.001549
grad_step = 000284, loss = 0.001531
grad_step = 000285, loss = 0.001513
grad_step = 000286, loss = 0.001505
grad_step = 000287, loss = 0.001506
grad_step = 000288, loss = 0.001513
grad_step = 000289, loss = 0.001521
grad_step = 000290, loss = 0.001526
grad_step = 000291, loss = 0.001526
grad_step = 000292, loss = 0.001520
grad_step = 000293, loss = 0.001513
grad_step = 000294, loss = 0.001504
grad_step = 000295, loss = 0.001498
grad_step = 000296, loss = 0.001494
grad_step = 000297, loss = 0.001494
grad_step = 000298, loss = 0.001495
grad_step = 000299, loss = 0.001498
grad_step = 000300, loss = 0.001500
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001502
grad_step = 000302, loss = 0.001503
grad_step = 000303, loss = 0.001502
grad_step = 000304, loss = 0.001502
grad_step = 000305, loss = 0.001499
grad_step = 000306, loss = 0.001497
grad_step = 000307, loss = 0.001494
grad_step = 000308, loss = 0.001491
grad_step = 000309, loss = 0.001488
grad_step = 000310, loss = 0.001485
grad_step = 000311, loss = 0.001482
grad_step = 000312, loss = 0.001480
grad_step = 000313, loss = 0.001478
grad_step = 000314, loss = 0.001476
grad_step = 000315, loss = 0.001474
grad_step = 000316, loss = 0.001473
grad_step = 000317, loss = 0.001471
grad_step = 000318, loss = 0.001470
grad_step = 000319, loss = 0.001469
grad_step = 000320, loss = 0.001468
grad_step = 000321, loss = 0.001466
grad_step = 000322, loss = 0.001465
grad_step = 000323, loss = 0.001464
grad_step = 000324, loss = 0.001463
grad_step = 000325, loss = 0.001463
grad_step = 000326, loss = 0.001464
grad_step = 000327, loss = 0.001467
grad_step = 000328, loss = 0.001474
grad_step = 000329, loss = 0.001488
grad_step = 000330, loss = 0.001520
grad_step = 000331, loss = 0.001561
grad_step = 000332, loss = 0.001633
grad_step = 000333, loss = 0.001640
grad_step = 000334, loss = 0.001629
grad_step = 000335, loss = 0.001520
grad_step = 000336, loss = 0.001475
grad_step = 000337, loss = 0.001519
grad_step = 000338, loss = 0.001547
grad_step = 000339, loss = 0.001504
grad_step = 000340, loss = 0.001450
grad_step = 000341, loss = 0.001470
grad_step = 000342, loss = 0.001510
grad_step = 000343, loss = 0.001492
grad_step = 000344, loss = 0.001451
grad_step = 000345, loss = 0.001444
grad_step = 000346, loss = 0.001470
grad_step = 000347, loss = 0.001484
grad_step = 000348, loss = 0.001462
grad_step = 000349, loss = 0.001438
grad_step = 000350, loss = 0.001436
grad_step = 000351, loss = 0.001451
grad_step = 000352, loss = 0.001461
grad_step = 000353, loss = 0.001451
grad_step = 000354, loss = 0.001435
grad_step = 000355, loss = 0.001428
grad_step = 000356, loss = 0.001432
grad_step = 000357, loss = 0.001441
grad_step = 000358, loss = 0.001442
grad_step = 000359, loss = 0.001437
grad_step = 000360, loss = 0.001427
grad_step = 000361, loss = 0.001421
grad_step = 000362, loss = 0.001421
grad_step = 000363, loss = 0.001424
grad_step = 000364, loss = 0.001427
grad_step = 000365, loss = 0.001427
grad_step = 000366, loss = 0.001424
grad_step = 000367, loss = 0.001419
grad_step = 000368, loss = 0.001414
grad_step = 000369, loss = 0.001411
grad_step = 000370, loss = 0.001410
grad_step = 000371, loss = 0.001411
grad_step = 000372, loss = 0.001412
grad_step = 000373, loss = 0.001413
grad_step = 000374, loss = 0.001413
grad_step = 000375, loss = 0.001412
grad_step = 000376, loss = 0.001410
grad_step = 000377, loss = 0.001408
grad_step = 000378, loss = 0.001405
grad_step = 000379, loss = 0.001402
grad_step = 000380, loss = 0.001400
grad_step = 000381, loss = 0.001398
grad_step = 000382, loss = 0.001396
grad_step = 000383, loss = 0.001395
grad_step = 000384, loss = 0.001394
grad_step = 000385, loss = 0.001393
grad_step = 000386, loss = 0.001392
grad_step = 000387, loss = 0.001392
grad_step = 000388, loss = 0.001392
grad_step = 000389, loss = 0.001392
grad_step = 000390, loss = 0.001394
grad_step = 000391, loss = 0.001399
grad_step = 000392, loss = 0.001409
grad_step = 000393, loss = 0.001424
grad_step = 000394, loss = 0.001451
grad_step = 000395, loss = 0.001476
grad_step = 000396, loss = 0.001501
grad_step = 000397, loss = 0.001484
grad_step = 000398, loss = 0.001445
grad_step = 000399, loss = 0.001403
grad_step = 000400, loss = 0.001402
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001435
grad_step = 000402, loss = 0.001459
grad_step = 000403, loss = 0.001459
grad_step = 000404, loss = 0.001429
grad_step = 000405, loss = 0.001416
grad_step = 000406, loss = 0.001433
grad_step = 000407, loss = 0.001443
grad_step = 000408, loss = 0.001440
grad_step = 000409, loss = 0.001408
grad_step = 000410, loss = 0.001382
grad_step = 000411, loss = 0.001376
grad_step = 000412, loss = 0.001384
grad_step = 000413, loss = 0.001393
grad_step = 000414, loss = 0.001392
grad_step = 000415, loss = 0.001383
grad_step = 000416, loss = 0.001371
grad_step = 000417, loss = 0.001368
grad_step = 000418, loss = 0.001374
grad_step = 000419, loss = 0.001380
grad_step = 000420, loss = 0.001381
grad_step = 000421, loss = 0.001373
grad_step = 000422, loss = 0.001364
grad_step = 000423, loss = 0.001358
grad_step = 000424, loss = 0.001355
grad_step = 000425, loss = 0.001358
grad_step = 000426, loss = 0.001363
grad_step = 000427, loss = 0.001367
grad_step = 000428, loss = 0.001371
grad_step = 000429, loss = 0.001379
grad_step = 000430, loss = 0.001386
grad_step = 000431, loss = 0.001389
grad_step = 000432, loss = 0.001369
grad_step = 000433, loss = 0.001363
grad_step = 000434, loss = 0.001358
grad_step = 000435, loss = 0.001350
grad_step = 000436, loss = 0.001349
grad_step = 000437, loss = 0.001358
grad_step = 000438, loss = 0.001364
grad_step = 000439, loss = 0.001359
grad_step = 000440, loss = 0.001352
grad_step = 000441, loss = 0.001350
grad_step = 000442, loss = 0.001351
grad_step = 000443, loss = 0.001344
grad_step = 000444, loss = 0.001338
grad_step = 000445, loss = 0.001338
grad_step = 000446, loss = 0.001343
grad_step = 000447, loss = 0.001348
grad_step = 000448, loss = 0.001343
grad_step = 000449, loss = 0.001345
grad_step = 000450, loss = 0.001355
grad_step = 000451, loss = 0.001371
grad_step = 000452, loss = 0.001407
grad_step = 000453, loss = 0.001474
grad_step = 000454, loss = 0.001610
grad_step = 000455, loss = 0.001718
grad_step = 000456, loss = 0.001811
grad_step = 000457, loss = 0.001607
grad_step = 000458, loss = 0.001398
grad_step = 000459, loss = 0.001394
grad_step = 000460, loss = 0.001536
grad_step = 000461, loss = 0.001534
grad_step = 000462, loss = 0.001371
grad_step = 000463, loss = 0.001399
grad_step = 000464, loss = 0.001495
grad_step = 000465, loss = 0.001417
grad_step = 000466, loss = 0.001350
grad_step = 000467, loss = 0.001427
grad_step = 000468, loss = 0.001424
grad_step = 000469, loss = 0.001347
grad_step = 000470, loss = 0.001362
grad_step = 000471, loss = 0.001409
grad_step = 000472, loss = 0.001355
grad_step = 000473, loss = 0.001359
grad_step = 000474, loss = 0.001370
grad_step = 000475, loss = 0.001383
grad_step = 000476, loss = 0.001343
grad_step = 000477, loss = 0.001331
grad_step = 000478, loss = 0.001365
grad_step = 000479, loss = 0.001336
grad_step = 000480, loss = 0.001329
grad_step = 000481, loss = 0.001345
grad_step = 000482, loss = 0.001344
grad_step = 000483, loss = 0.001343
grad_step = 000484, loss = 0.001325
grad_step = 000485, loss = 0.001346
grad_step = 000486, loss = 0.001337
grad_step = 000487, loss = 0.001323
grad_step = 000488, loss = 0.001322
grad_step = 000489, loss = 0.001323
grad_step = 000490, loss = 0.001332
grad_step = 000491, loss = 0.001319
grad_step = 000492, loss = 0.001310
grad_step = 000493, loss = 0.001322
grad_step = 000494, loss = 0.001321
grad_step = 000495, loss = 0.001308
grad_step = 000496, loss = 0.001328
grad_step = 000497, loss = 0.001316
grad_step = 000498, loss = 0.001320
grad_step = 000499, loss = 0.001324
grad_step = 000500, loss = 0.001310
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001308
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

  date_run                              2020-05-10 13:13:59.294954
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.27733
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 13:13:59.301043
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.197235
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 13:13:59.308379
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.148252
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 13:13:59.314338
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.99705
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
0   2020-05-10 13:13:30.160454  ...    mean_absolute_error
1   2020-05-10 13:13:30.165085  ...     mean_squared_error
2   2020-05-10 13:13:30.168931  ...  median_absolute_error
3   2020-05-10 13:13:30.173007  ...               r2_score
4   2020-05-10 13:13:38.857966  ...    mean_absolute_error
5   2020-05-10 13:13:38.862189  ...     mean_squared_error
6   2020-05-10 13:13:38.865839  ...  median_absolute_error
7   2020-05-10 13:13:38.869519  ...               r2_score
8   2020-05-10 13:13:59.294954  ...    mean_absolute_error
9   2020-05-10 13:13:59.301043  ...     mean_squared_error
10  2020-05-10 13:13:59.308379  ...  median_absolute_error
11  2020-05-10 13:13:59.314338  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3776512/9912422 [00:00<00:00, 37705374.71it/s]9920512it [00:00, 34593120.53it/s]                             
0it [00:00, ?it/s]32768it [00:00, 499799.82it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 481917.28it/s]1654784it [00:00, 9224127.61it/s]                          
0it [00:00, ?it/s]8192it [00:00, 205727.23it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c7ad1b780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c1845dc18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c7acd2e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c1845dda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c7acd2e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c7ad1be80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c7ad1b780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c2d6cecc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c1845dda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c2d6cecc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5c7acd2e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd7c7d2b1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e7e8013a2ec1af70d24f3d6a935fffd7f6054f6b18b3f61819226cfa90431405
  Stored in directory: /tmp/pip-ephem-wheel-cache-5s18o5r_/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd7be0b5048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3072000/17464789 [====>.........................] - ETA: 0s
 9666560/17464789 [===============>..............] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 13:15:26.818771: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 13:15:26.823223: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-10 13:15:26.823417: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557ce6389f30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 13:15:26.823437: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7510 - accuracy: 0.4945 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4008 - accuracy: 0.5173
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5808 - accuracy: 0.5056
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5670 - accuracy: 0.5065
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6649 - accuracy: 0.5001
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6498 - accuracy: 0.5011
11000/25000 [============>.................] - ETA: 4s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 3s - loss: 7.6385 - accuracy: 0.5018
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6525 - accuracy: 0.5009
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6273 - accuracy: 0.5026
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6143 - accuracy: 0.5034
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6326 - accuracy: 0.5022
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6448 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6283 - accuracy: 0.5025
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6257 - accuracy: 0.5027
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6255 - accuracy: 0.5027
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6253 - accuracy: 0.5027
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6455 - accuracy: 0.5014
25000/25000 [==============================] - 9s 343us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 13:15:42.520932
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 13:15:42.520932  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 13:15:48.997790: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 13:15:49.002744: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-10 13:15:49.002935: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560ce188e6a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 13:15:49.002954: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f8197a68be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 0.9018 - crf_viterbi_accuracy: 0.6533 - val_loss: 0.8925 - val_crf_viterbi_accuracy: 0.6800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f81729aef60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4826 - accuracy: 0.5120
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6743 - accuracy: 0.4995
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6381 - accuracy: 0.5019
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6649 - accuracy: 0.5001
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6866 - accuracy: 0.4987
11000/25000 [============>.................] - ETA: 4s - loss: 7.6736 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6761 - accuracy: 0.4994
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6852 - accuracy: 0.4988
15000/25000 [=================>............] - ETA: 2s - loss: 7.7085 - accuracy: 0.4973
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7308 - accuracy: 0.4958
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7314 - accuracy: 0.4958
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6889 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6834 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6933 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6922 - accuracy: 0.4983
25000/25000 [==============================] - 9s 346us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f812e70b048> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:45:02, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:45:22, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:22:59, 23.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:16:35, 32.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<5:04:50, 46.9kB/s].vector_cache/glove.6B.zip:   1%|          | 8.88M/862M [00:01<3:32:09, 67.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:28:03, 95.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:43:03, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:01<1:11:45, 195kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.8M/862M [00:01<49:59, 278kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<34:57, 396kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<24:24, 563kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<17:07, 799kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.7M/862M [00:02<12:15, 1.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:02<08:36, 1.58MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:42, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:35, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:25, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<04:54, 2.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:59, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:17, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<05:44, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:07<04:13, 3.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<07:39, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<06:51, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<05:07, 2.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<06:25, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<05:51, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<04:23, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<06:08, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<06:58, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<05:28, 2.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:13<03:57, 3.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<26:51, 487kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<20:07, 650kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<14:23, 906kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<13:06, 993kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<11:50, 1.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<08:56, 1.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<08:20, 1.55MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<07:08, 1.81MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<05:19, 2.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:43, 1.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:19, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<05:42, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<04:37, 2.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<7:59:08, 26.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<5:35:20, 38.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<3:56:11, 54.0kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<2:48:33, 75.6kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<1:58:42, 107kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<1:23:00, 153kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<1:05:16, 194kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<47:04, 269kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<33:11, 381kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<25:54, 487kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<20:48, 606kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<15:11, 829kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<10:45, 1.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:37:10, 129kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:09:17, 181kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<48:41, 257kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<36:54, 338kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<28:21, 440kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<20:28, 609kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<16:16, 762kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<12:38, 980kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<09:09, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:17, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:43, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:42, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:53, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:03, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:32, 2.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:03, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:43, 1.81MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:13, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<03:47, 3.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<11:24, 1.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:13, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:45, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:32, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:43, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:54, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:16, 2.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<10:34, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:38, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:18, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:10, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:15, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:40, 2.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:03, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:38, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:14, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:34, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:17, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:55, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:34, 3.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<10:46, 1.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:45, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:21, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:11, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:23, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:39, 2.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:05, 2.83MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<11:13, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<09:02, 1.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:34, 1.76MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:17, 1.58MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:27, 1.54MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:48, 1.98MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:54, 1.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:28, 1.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:06, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:10, 2.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<6:51:40, 27.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<4:48:03, 39.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<3:22:54, 55.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<2:24:12, 78.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:41:23, 111kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<1:12:29, 155kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<51:54, 217kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<36:33, 307kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<28:01, 399kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<21:54, 510kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<15:49, 706kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<11:12, 992kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<12:09, 914kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:38, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:00, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:28, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:20, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:40, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:52, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:20, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:48, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:35, 3.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:37, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<03:27, 3.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<04:57, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:33, 2.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:27, 3.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<04:56, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:21, 2.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:16, 3.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<02:27, 4.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<1:20:55, 131kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<57:30, 185kB/s]  .vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<40:25, 262kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<30:42, 344kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<23:37, 447kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<17:02, 619kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<11:59, 875kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<25:25, 413kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<18:56, 553kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<13:31, 774kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<11:43, 889kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<09:21, 1.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:46, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<07:00, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:02, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:30, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:24, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:54, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:42, 2.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:50, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:38, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:30, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:16, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<14:41, 691kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<11:12, 905kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:02, 1.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<05:46, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<12:52, 782kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<10:07, 995kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:18, 1.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:16, 1.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<07:18, 1.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:35, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:03, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:12, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:25, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:03, 2.44MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:00, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:34, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:25, 2.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:33, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:20, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:11, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:05, 3.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:32, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:57, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<03:43, 2.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:42, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:25, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:20, 2.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:08, 3.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<11:12, 856kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<08:54, 1.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:28, 1.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:36, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:44, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:14, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:46, 2.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<11:36, 814kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<09:08, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:36, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:40, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<06:45, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:15, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:46, 2.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<12:36, 739kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<09:50, 945kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:07, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:59, 1.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:56, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:51, 2.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<13:48, 664kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<10:38, 861kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<07:39, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:20, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:08, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:30, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:56, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<12:37, 716kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<09:48, 920kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<07:05, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:55, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:49, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:03, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<5:37:51, 26.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<3:56:11, 37.6kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<2:46:10, 53.2kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<1:58:19, 74.6kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<1:23:14, 106kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<58:05, 151kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<47:54, 183kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<34:28, 254kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<24:16, 360kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<18:47, 463kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<15:04, 577kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<11:01, 788kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<07:46, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<15:10, 568kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<11:32, 746kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<08:17, 1.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:39, 1.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:10, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:26, 1.57MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<13:08, 646kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<10:07, 837kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<07:18, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:56, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:42, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:07, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<03:40, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<11:11, 746kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:43, 955kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<06:16, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:12, 1.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:05, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<04:39, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<03:21, 2.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<11:08, 737kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:41, 943kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<06:15, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:08, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:06, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:39, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<03:20, 2.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:02, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:49, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:14, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:43, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:11, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:08, 2.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:31, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:36, 2.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<02:36, 3.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:40, 736kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<08:18, 946kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<05:58, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:55, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:47, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:23, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:11, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:15, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:29, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:18, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:02, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:27, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:31, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:32, 2.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<20:50, 364kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<15:14, 497kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<10:50, 696kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<09:12, 816kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:16, 1.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:14, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:18, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:31, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:19, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:56, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:22, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:24, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:28, 2.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:18, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:22, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:14, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:52, 1.87MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:28, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:36, 2.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:29, 2.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:59, 1.80MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:09, 2.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:17, 3.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<24:05, 295kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<17:36, 403kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<12:26, 568kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<10:15, 685kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<08:40, 809kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:23, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<04:32, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<07:02, 989kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:41, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:07, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:24, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:36, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:35, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:35, 2.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<09:05, 751kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<07:06, 958kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:08, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:04, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<05:02, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:53, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:47, 2.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<10:03, 665kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<07:44, 863kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:32, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:21, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:09, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:57, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:49, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<17:45, 369kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<13:08, 497kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:19, 699kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<07:54, 819kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<06:57, 929kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:13, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:43, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<10:22, 618kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<07:57, 804kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:41, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:21, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:05, 1.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:53, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:46, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<15:22, 407kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<11:26, 547kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<08:09, 765kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<07:02, 881kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<06:17, 985kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:41, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:20, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:05, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:13, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:06, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:30, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:47, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:59, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:08, 2.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<07:56, 754kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:12, 963kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:29, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:25, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:24, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:24, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:26, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:48, 664kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<06:48, 858kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:54, 1.19MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:40, 1.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:33, 1.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:27, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:29, 2.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:58, 1.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:25, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:32, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:00, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:21, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:36, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:53, 2.95MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:40, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:11, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:22, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:50, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:28, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:49, 2.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:22, 3.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<14:27, 376kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<11:09, 487kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<08:06, 669kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<05:43, 941kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<04:37, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<3:27:21, 25.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<2:24:40, 37.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<1:41:27, 52.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<1:12:09, 73.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<50:39, 104kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<35:18, 149kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<26:39, 196kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<19:12, 272kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<13:31, 385kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<10:31, 490kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<08:27, 610kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<06:10, 833kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:01<04:21, 1.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<18:22, 277kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<13:23, 380kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<09:28, 535kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:41, 654kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<05:54, 848kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:14, 1.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:02, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:55, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:00, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<02:09, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<07:26, 657kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:43, 852kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<04:07, 1.18MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:55, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:45, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:50, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<02:03, 2.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:59, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:36, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:55, 2.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:22, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:42, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:06, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:31, 3.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:14, 1.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:46, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:03, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:25, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:42, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:08, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<01:32, 2.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<06:29, 688kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:02, 886kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:36, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:45, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:34, 963kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:41, 1.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:41, 1.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:46, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:25, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:48, 2.39MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:18, 3.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<09:29, 449kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<07:05, 600kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<05:03, 837kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:33, 1.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<11:23, 368kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<08:54, 471kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<06:24, 651kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<04:30, 920kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<05:15, 784kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:08, 996kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:58, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:57, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:58, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:18, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:38, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<05:58, 667kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:35, 865kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:18, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:10, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:05, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:22, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:41, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<04:47, 803kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:45, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:42, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:43, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:18, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:41, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:01, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:49, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:22, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:45, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:04, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:37, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:10, 3.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:27, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:06, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:33, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:50, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:39, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:13, 2.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:38, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:54, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:31, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:05, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<04:20, 773kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:23, 988kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:26, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:26, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:26, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:51, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:20, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:04, 1.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:48, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:20, 2.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:36, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:48, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:25, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:01, 3.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<08:05, 381kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:58, 515kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<04:13, 723kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<03:36, 837kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<03:11, 943kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:23, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:41, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<04:20, 679kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:21, 876kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:23, 1.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<02:17, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:55, 1.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:24, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:35, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:44, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:20, 2.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:58, 2.84MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:44, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:30, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:06, 2.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:33, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:13, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:52, 2.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:22, 768kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:39, 977kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:54, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:52, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:35, 1.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:09, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:20, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:28, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:10, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:49, 2.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:08, 760kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:26, 973kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:45, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:44, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:42, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:17, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:55, 2.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:59, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:38, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:11, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:18, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:25, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:06, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:47, 2.70MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<02:48, 753kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:11, 961kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:34, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:31, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:30, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:08, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:48, 2.46MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:28, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:14, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<00:54, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:03, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:08, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:54, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:38, 2.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<05:57, 308kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<04:20, 421kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<03:02, 593kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:29, 711kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:07, 827kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:34, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:05, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<02:49, 601kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<02:09, 784kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:31, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:24, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:09, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:50, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:59, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:45, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:32, 2.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:53, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:47, 1.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:34, 2.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:42, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:48, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:37, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:26, 3.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<01:09, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:57, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:41, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:45, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:47, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:37, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:25, 2.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<03:56, 307kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<02:52, 418kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<02:00, 588kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:36, 709kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:22, 827kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:00, 1.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:41, 1.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:56, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:46, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:33, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:29, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:20, 2.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<03:02, 307kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<02:13, 418kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:32, 588kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:13, 709kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<01:02, 832kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:45, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:30, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<02:45, 289kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<02:00, 395kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:22, 555kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:04, 676kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:54, 800kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:39, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:26, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:44, 896kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:34, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:24, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:24, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:24, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:18, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:12, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:41, 743kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:32, 951kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:22, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:20, 1.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:19, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:14, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:08, 2.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:07, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:04, 3.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:47, 309kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:33, 421kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:21, 592kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:14, 713kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:12, 838kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:08, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.93MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.98MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 727/400000 [00:00<00:54, 7266.11it/s]  0%|          | 1502/400000 [00:00<00:53, 7404.54it/s]  1%|          | 2273/400000 [00:00<00:53, 7493.56it/s]  1%|          | 3038/400000 [00:00<00:52, 7537.19it/s]  1%|          | 3807/400000 [00:00<00:52, 7581.70it/s]  1%|          | 4555/400000 [00:00<00:52, 7549.32it/s]  1%|▏         | 5328/400000 [00:00<00:51, 7600.92it/s]  2%|▏         | 6105/400000 [00:00<00:51, 7649.52it/s]  2%|▏         | 6879/400000 [00:00<00:51, 7673.85it/s]  2%|▏         | 7618/400000 [00:01<00:51, 7571.84it/s]  2%|▏         | 8364/400000 [00:01<00:51, 7535.46it/s]  2%|▏         | 9125/400000 [00:01<00:51, 7555.28it/s]  2%|▏         | 9897/400000 [00:01<00:51, 7602.61it/s]  3%|▎         | 10656/400000 [00:01<00:51, 7596.09it/s]  3%|▎         | 11428/400000 [00:01<00:50, 7631.62it/s]  3%|▎         | 12188/400000 [00:01<00:51, 7545.45it/s]  3%|▎         | 12964/400000 [00:01<00:50, 7608.52it/s]  3%|▎         | 13741/400000 [00:01<00:50, 7654.44it/s]  4%|▎         | 14514/400000 [00:01<00:50, 7674.78it/s]  4%|▍         | 15295/400000 [00:02<00:49, 7713.11it/s]  4%|▍         | 16066/400000 [00:02<00:50, 7638.94it/s]  4%|▍         | 16832/400000 [00:02<00:50, 7643.76it/s]  4%|▍         | 17597/400000 [00:02<00:50, 7645.25it/s]  5%|▍         | 18379/400000 [00:02<00:49, 7694.55it/s]  5%|▍         | 19158/400000 [00:02<00:49, 7721.11it/s]  5%|▍         | 19931/400000 [00:02<00:49, 7673.63it/s]  5%|▌         | 20710/400000 [00:02<00:49, 7706.47it/s]  5%|▌         | 21481/400000 [00:02<00:49, 7683.41it/s]  6%|▌         | 22251/400000 [00:02<00:49, 7687.88it/s]  6%|▌         | 23027/400000 [00:03<00:48, 7706.40it/s]  6%|▌         | 23798/400000 [00:03<00:49, 7626.40it/s]  6%|▌         | 24564/400000 [00:03<00:49, 7635.53it/s]  6%|▋         | 25342/400000 [00:03<00:48, 7676.41it/s]  7%|▋         | 26110/400000 [00:03<00:48, 7674.09it/s]  7%|▋         | 26886/400000 [00:03<00:48, 7699.27it/s]  7%|▋         | 27657/400000 [00:03<00:48, 7667.14it/s]  7%|▋         | 28433/400000 [00:03<00:48, 7694.45it/s]  7%|▋         | 29212/400000 [00:03<00:48, 7721.71it/s]  7%|▋         | 29985/400000 [00:03<00:48, 7704.47it/s]  8%|▊         | 30756/400000 [00:04<00:48, 7632.56it/s]  8%|▊         | 31520/400000 [00:04<00:49, 7514.68it/s]  8%|▊         | 32282/400000 [00:04<00:48, 7544.76it/s]  8%|▊         | 33062/400000 [00:04<00:48, 7618.21it/s]  8%|▊         | 33837/400000 [00:04<00:47, 7655.62it/s]  9%|▊         | 34610/400000 [00:04<00:47, 7676.05it/s]  9%|▉         | 35378/400000 [00:04<00:47, 7672.86it/s]  9%|▉         | 36146/400000 [00:04<00:47, 7592.19it/s]  9%|▉         | 36921/400000 [00:04<00:47, 7636.82it/s]  9%|▉         | 37701/400000 [00:04<00:47, 7683.34it/s] 10%|▉         | 38482/400000 [00:05<00:46, 7718.22it/s] 10%|▉         | 39255/400000 [00:05<00:46, 7676.60it/s] 10%|█         | 40023/400000 [00:05<00:46, 7663.14it/s] 10%|█         | 40801/400000 [00:05<00:46, 7696.20it/s] 10%|█         | 41576/400000 [00:05<00:46, 7710.41it/s] 11%|█         | 42352/400000 [00:05<00:46, 7723.95it/s] 11%|█         | 43125/400000 [00:05<00:46, 7677.02it/s] 11%|█         | 43904/400000 [00:05<00:46, 7709.41it/s] 11%|█         | 44680/400000 [00:05<00:46, 7721.48it/s] 11%|█▏        | 45453/400000 [00:05<00:46, 7685.76it/s] 12%|█▏        | 46222/400000 [00:06<00:46, 7686.14it/s] 12%|█▏        | 46991/400000 [00:06<00:46, 7672.24it/s] 12%|█▏        | 47760/400000 [00:06<00:45, 7675.04it/s] 12%|█▏        | 48540/400000 [00:06<00:45, 7711.82it/s] 12%|█▏        | 49321/400000 [00:06<00:45, 7740.68it/s] 13%|█▎        | 50099/400000 [00:06<00:45, 7751.59it/s] 13%|█▎        | 50875/400000 [00:06<00:45, 7719.13it/s] 13%|█▎        | 51647/400000 [00:06<00:45, 7697.50it/s] 13%|█▎        | 52417/400000 [00:06<00:46, 7522.37it/s] 13%|█▎        | 53194/400000 [00:06<00:45, 7594.63it/s] 13%|█▎        | 53969/400000 [00:07<00:45, 7639.06it/s] 14%|█▎        | 54734/400000 [00:07<00:45, 7574.90it/s] 14%|█▍        | 55493/400000 [00:07<00:45, 7563.46it/s] 14%|█▍        | 56264/400000 [00:07<00:45, 7604.59it/s] 14%|█▍        | 57035/400000 [00:07<00:44, 7634.46it/s] 14%|█▍        | 57807/400000 [00:07<00:44, 7659.79it/s] 15%|█▍        | 58574/400000 [00:07<00:44, 7652.46it/s] 15%|█▍        | 59340/400000 [00:07<00:44, 7651.63it/s] 15%|█▌        | 60117/400000 [00:07<00:44, 7684.94it/s] 15%|█▌        | 60886/400000 [00:07<00:44, 7579.11it/s] 15%|█▌        | 61658/400000 [00:08<00:44, 7618.02it/s] 16%|█▌        | 62423/400000 [00:08<00:44, 7626.90it/s] 16%|█▌        | 63186/400000 [00:08<00:44, 7585.08it/s] 16%|█▌        | 63966/400000 [00:08<00:43, 7647.28it/s] 16%|█▌        | 64736/400000 [00:08<00:43, 7662.25it/s] 16%|█▋        | 65508/400000 [00:08<00:43, 7679.11it/s] 17%|█▋        | 66287/400000 [00:08<00:43, 7709.22it/s] 17%|█▋        | 67059/400000 [00:08<00:43, 7668.92it/s] 17%|█▋        | 67827/400000 [00:08<00:43, 7613.47it/s] 17%|█▋        | 68600/400000 [00:08<00:43, 7647.28it/s] 17%|█▋        | 69365/400000 [00:09<00:43, 7637.57it/s] 18%|█▊        | 70139/400000 [00:09<00:43, 7667.46it/s] 18%|█▊        | 70906/400000 [00:09<00:43, 7634.70it/s] 18%|█▊        | 71683/400000 [00:09<00:42, 7673.36it/s] 18%|█▊        | 72465/400000 [00:09<00:42, 7715.63it/s] 18%|█▊        | 73247/400000 [00:09<00:42, 7745.13it/s] 19%|█▊        | 74027/400000 [00:09<00:42, 7761.05it/s] 19%|█▊        | 74804/400000 [00:09<00:42, 7677.57it/s] 19%|█▉        | 75578/400000 [00:09<00:42, 7694.09it/s] 19%|█▉        | 76348/400000 [00:09<00:42, 7586.08it/s] 19%|█▉        | 77113/400000 [00:10<00:42, 7602.37it/s] 19%|█▉        | 77874/400000 [00:10<00:42, 7591.95it/s] 20%|█▉        | 78634/400000 [00:10<00:42, 7518.88it/s] 20%|█▉        | 79413/400000 [00:10<00:42, 7597.38it/s] 20%|██        | 80193/400000 [00:10<00:41, 7655.89it/s] 20%|██        | 80971/400000 [00:10<00:41, 7690.74it/s] 20%|██        | 81751/400000 [00:10<00:41, 7721.76it/s] 21%|██        | 82524/400000 [00:10<00:41, 7671.17it/s] 21%|██        | 83295/400000 [00:10<00:41, 7679.82it/s] 21%|██        | 84069/400000 [00:10<00:41, 7696.45it/s] 21%|██        | 84839/400000 [00:11<00:40, 7695.55it/s] 21%|██▏       | 85609/400000 [00:11<00:40, 7690.82it/s] 22%|██▏       | 86379/400000 [00:11<00:41, 7618.48it/s] 22%|██▏       | 87150/400000 [00:11<00:40, 7645.49it/s] 22%|██▏       | 87915/400000 [00:11<00:41, 7580.63it/s] 22%|██▏       | 88682/400000 [00:11<00:40, 7605.60it/s] 22%|██▏       | 89446/400000 [00:11<00:40, 7614.78it/s] 23%|██▎       | 90208/400000 [00:11<00:40, 7578.34it/s] 23%|██▎       | 90984/400000 [00:11<00:40, 7630.54it/s] 23%|██▎       | 91748/400000 [00:11<00:40, 7615.43it/s] 23%|██▎       | 92518/400000 [00:12<00:40, 7639.26it/s] 23%|██▎       | 93289/400000 [00:12<00:40, 7660.14it/s] 24%|██▎       | 94056/400000 [00:12<00:40, 7586.31it/s] 24%|██▎       | 94827/400000 [00:12<00:40, 7621.97it/s] 24%|██▍       | 95594/400000 [00:12<00:39, 7634.82it/s] 24%|██▍       | 96363/400000 [00:12<00:39, 7650.74it/s] 24%|██▍       | 97130/400000 [00:12<00:39, 7655.12it/s] 24%|██▍       | 97896/400000 [00:12<00:39, 7581.57it/s] 25%|██▍       | 98665/400000 [00:12<00:39, 7612.95it/s] 25%|██▍       | 99427/400000 [00:13<00:39, 7547.13it/s] 25%|██▌       | 100183/400000 [00:13<00:39, 7549.77it/s] 25%|██▌       | 100939/400000 [00:13<00:39, 7550.57it/s] 25%|██▌       | 101695/400000 [00:13<00:39, 7532.43it/s] 26%|██▌       | 102461/400000 [00:13<00:39, 7569.68it/s] 26%|██▌       | 103230/400000 [00:13<00:39, 7602.76it/s] 26%|██▌       | 103991/400000 [00:13<00:39, 7587.56it/s] 26%|██▌       | 104760/400000 [00:13<00:38, 7615.74it/s] 26%|██▋       | 105522/400000 [00:13<00:38, 7578.59it/s] 27%|██▋       | 106294/400000 [00:13<00:38, 7619.60it/s] 27%|██▋       | 107057/400000 [00:14<00:38, 7601.12it/s] 27%|██▋       | 107828/400000 [00:14<00:38, 7630.69it/s] 27%|██▋       | 108600/400000 [00:14<00:38, 7655.44it/s] 27%|██▋       | 109366/400000 [00:14<00:38, 7516.79it/s] 28%|██▊       | 110121/400000 [00:14<00:38, 7525.22it/s] 28%|██▊       | 110898/400000 [00:14<00:38, 7595.07it/s] 28%|██▊       | 111663/400000 [00:14<00:37, 7610.21it/s] 28%|██▊       | 112425/400000 [00:14<00:38, 7424.58it/s] 28%|██▊       | 113169/400000 [00:14<00:38, 7366.42it/s] 28%|██▊       | 113907/400000 [00:14<00:38, 7356.72it/s] 29%|██▊       | 114677/400000 [00:15<00:38, 7456.01it/s] 29%|██▉       | 115424/400000 [00:15<00:38, 7357.57it/s] 29%|██▉       | 116181/400000 [00:15<00:38, 7418.83it/s] 29%|██▉       | 116924/400000 [00:15<00:38, 7320.26it/s] 29%|██▉       | 117685/400000 [00:15<00:38, 7402.30it/s] 30%|██▉       | 118448/400000 [00:15<00:37, 7468.36it/s] 30%|██▉       | 119200/400000 [00:15<00:37, 7482.64it/s] 30%|██▉       | 119969/400000 [00:15<00:37, 7541.63it/s] 30%|███       | 120724/400000 [00:15<00:37, 7520.84it/s] 30%|███       | 121497/400000 [00:15<00:36, 7580.98it/s] 31%|███       | 122257/400000 [00:16<00:36, 7585.67it/s] 31%|███       | 123035/400000 [00:16<00:36, 7639.67it/s] 31%|███       | 123811/400000 [00:16<00:35, 7675.10it/s] 31%|███       | 124579/400000 [00:16<00:37, 7401.85it/s] 31%|███▏      | 125322/400000 [00:16<00:37, 7281.62it/s] 32%|███▏      | 126092/400000 [00:16<00:37, 7402.01it/s] 32%|███▏      | 126867/400000 [00:16<00:36, 7501.59it/s] 32%|███▏      | 127623/400000 [00:16<00:36, 7518.20it/s] 32%|███▏      | 128376/400000 [00:16<00:36, 7466.11it/s] 32%|███▏      | 129131/400000 [00:16<00:36, 7480.80it/s] 32%|███▏      | 129887/400000 [00:17<00:35, 7503.45it/s] 33%|███▎      | 130645/400000 [00:17<00:35, 7525.85it/s] 33%|███▎      | 131420/400000 [00:17<00:35, 7590.29it/s] 33%|███▎      | 132180/400000 [00:17<00:35, 7584.75it/s] 33%|███▎      | 132958/400000 [00:17<00:34, 7641.55it/s] 33%|███▎      | 133723/400000 [00:17<00:34, 7640.06it/s] 34%|███▎      | 134495/400000 [00:17<00:34, 7661.64it/s] 34%|███▍      | 135264/400000 [00:17<00:34, 7667.39it/s] 34%|███▍      | 136031/400000 [00:17<00:34, 7618.03it/s] 34%|███▍      | 136802/400000 [00:17<00:34, 7642.62it/s] 34%|███▍      | 137581/400000 [00:18<00:34, 7683.40it/s] 35%|███▍      | 138350/400000 [00:18<00:34, 7642.59it/s] 35%|███▍      | 139129/400000 [00:18<00:33, 7684.08it/s] 35%|███▍      | 139898/400000 [00:18<00:34, 7620.72it/s] 35%|███▌      | 140670/400000 [00:18<00:33, 7650.06it/s] 35%|███▌      | 141444/400000 [00:18<00:33, 7672.97it/s] 36%|███▌      | 142212/400000 [00:18<00:33, 7671.46it/s] 36%|███▌      | 142986/400000 [00:18<00:33, 7691.38it/s] 36%|███▌      | 143756/400000 [00:18<00:33, 7692.50it/s] 36%|███▌      | 144526/400000 [00:18<00:33, 7657.79it/s] 36%|███▋      | 145292/400000 [00:19<00:33, 7594.28it/s] 37%|███▋      | 146055/400000 [00:19<00:33, 7603.40it/s] 37%|███▋      | 146820/400000 [00:19<00:33, 7614.99it/s] 37%|███▋      | 147596/400000 [00:19<00:32, 7656.93it/s] 37%|███▋      | 148362/400000 [00:19<00:33, 7557.29it/s] 37%|███▋      | 149124/400000 [00:19<00:33, 7574.53it/s] 37%|███▋      | 149882/400000 [00:19<00:33, 7453.06it/s] 38%|███▊      | 150665/400000 [00:19<00:32, 7560.06it/s] 38%|███▊      | 151422/400000 [00:19<00:33, 7466.76it/s] 38%|███▊      | 152183/400000 [00:19<00:33, 7507.51it/s] 38%|███▊      | 152941/400000 [00:20<00:32, 7529.10it/s] 38%|███▊      | 153718/400000 [00:20<00:32, 7597.87it/s] 39%|███▊      | 154497/400000 [00:20<00:32, 7653.67it/s] 39%|███▉      | 155267/400000 [00:20<00:31, 7664.79it/s] 39%|███▉      | 156034/400000 [00:20<00:31, 7624.16it/s] 39%|███▉      | 156818/400000 [00:20<00:31, 7686.17it/s] 39%|███▉      | 157603/400000 [00:20<00:31, 7731.62it/s] 40%|███▉      | 158377/400000 [00:20<00:31, 7701.73it/s] 40%|███▉      | 159151/400000 [00:20<00:31, 7712.50it/s] 40%|███▉      | 159923/400000 [00:20<00:31, 7621.69it/s] 40%|████      | 160701/400000 [00:21<00:31, 7666.72it/s] 40%|████      | 161479/400000 [00:21<00:30, 7697.59it/s] 41%|████      | 162255/400000 [00:21<00:30, 7713.32it/s] 41%|████      | 163030/400000 [00:21<00:30, 7723.89it/s] 41%|████      | 163803/400000 [00:21<00:31, 7570.53it/s] 41%|████      | 164577/400000 [00:21<00:30, 7620.46it/s] 41%|████▏     | 165340/400000 [00:21<00:30, 7622.27it/s] 42%|████▏     | 166103/400000 [00:21<00:30, 7550.20it/s] 42%|████▏     | 166859/400000 [00:21<00:30, 7544.33it/s] 42%|████▏     | 167619/400000 [00:22<00:30, 7559.77it/s] 42%|████▏     | 168396/400000 [00:22<00:30, 7621.22it/s] 42%|████▏     | 169173/400000 [00:22<00:30, 7664.87it/s] 42%|████▏     | 169943/400000 [00:22<00:29, 7675.06it/s] 43%|████▎     | 170720/400000 [00:22<00:29, 7700.83it/s] 43%|████▎     | 171491/400000 [00:22<00:29, 7639.51it/s] 43%|████▎     | 172275/400000 [00:22<00:29, 7696.53it/s] 43%|████▎     | 173057/400000 [00:22<00:29, 7731.36it/s] 43%|████▎     | 173834/400000 [00:22<00:29, 7741.95it/s] 44%|████▎     | 174609/400000 [00:22<00:29, 7743.84it/s] 44%|████▍     | 175384/400000 [00:23<00:29, 7629.92it/s] 44%|████▍     | 176160/400000 [00:23<00:29, 7666.90it/s] 44%|████▍     | 176937/400000 [00:23<00:28, 7694.90it/s] 44%|████▍     | 177715/400000 [00:23<00:28, 7719.78it/s] 45%|████▍     | 178488/400000 [00:23<00:28, 7709.57it/s] 45%|████▍     | 179260/400000 [00:23<00:28, 7641.92it/s] 45%|████▌     | 180025/400000 [00:23<00:28, 7622.54it/s] 45%|████▌     | 180789/400000 [00:23<00:28, 7626.12it/s] 45%|████▌     | 181552/400000 [00:23<00:29, 7523.10it/s] 46%|████▌     | 182305/400000 [00:23<00:29, 7489.02it/s] 46%|████▌     | 183055/400000 [00:24<00:28, 7489.33it/s] 46%|████▌     | 183827/400000 [00:24<00:28, 7554.29it/s] 46%|████▌     | 184583/400000 [00:24<00:28, 7488.86it/s] 46%|████▋     | 185361/400000 [00:24<00:28, 7573.06it/s] 47%|████▋     | 186132/400000 [00:24<00:28, 7613.33it/s] 47%|████▋     | 186894/400000 [00:24<00:28, 7377.56it/s] 47%|████▋     | 187675/400000 [00:24<00:28, 7501.62it/s] 47%|████▋     | 188455/400000 [00:24<00:27, 7587.80it/s] 47%|████▋     | 189225/400000 [00:24<00:27, 7618.91it/s] 47%|████▋     | 189996/400000 [00:24<00:27, 7643.81it/s] 48%|████▊     | 190762/400000 [00:25<00:27, 7609.73it/s] 48%|████▊     | 191541/400000 [00:25<00:27, 7660.87it/s] 48%|████▊     | 192324/400000 [00:25<00:26, 7709.39it/s] 48%|████▊     | 193100/400000 [00:25<00:26, 7723.59it/s] 48%|████▊     | 193873/400000 [00:25<00:26, 7691.75it/s] 49%|████▊     | 194643/400000 [00:25<00:27, 7547.63it/s] 49%|████▉     | 195423/400000 [00:25<00:26, 7619.79it/s] 49%|████▉     | 196205/400000 [00:25<00:26, 7677.57it/s] 49%|████▉     | 196975/400000 [00:25<00:26, 7684.09it/s] 49%|████▉     | 197751/400000 [00:25<00:26, 7704.82it/s] 50%|████▉     | 198522/400000 [00:26<00:26, 7619.91it/s] 50%|████▉     | 199303/400000 [00:26<00:26, 7675.10it/s] 50%|█████     | 200083/400000 [00:26<00:25, 7710.05it/s] 50%|█████     | 200855/400000 [00:26<00:25, 7683.03it/s] 50%|█████     | 201624/400000 [00:26<00:26, 7620.64it/s] 51%|█████     | 202387/400000 [00:26<00:26, 7585.93it/s] 51%|█████     | 203159/400000 [00:26<00:25, 7624.22it/s] 51%|█████     | 203922/400000 [00:26<00:25, 7611.73it/s] 51%|█████     | 204684/400000 [00:26<00:25, 7592.74it/s] 51%|█████▏    | 205444/400000 [00:26<00:25, 7533.18it/s] 52%|█████▏    | 206206/400000 [00:27<00:25, 7556.89it/s] 52%|█████▏    | 206978/400000 [00:27<00:25, 7603.27it/s] 52%|█████▏    | 207755/400000 [00:27<00:25, 7649.95it/s] 52%|█████▏    | 208521/400000 [00:27<00:25, 7638.47it/s] 52%|█████▏    | 209285/400000 [00:27<00:25, 7608.73it/s] 53%|█████▎    | 210046/400000 [00:27<00:25, 7453.11it/s] 53%|█████▎    | 210820/400000 [00:27<00:25, 7535.63it/s] 53%|█████▎    | 211589/400000 [00:27<00:24, 7579.37it/s] 53%|█████▎    | 212348/400000 [00:27<00:24, 7578.19it/s] 53%|█████▎    | 213117/400000 [00:27<00:24, 7610.40it/s] 53%|█████▎    | 213879/400000 [00:28<00:24, 7562.91it/s] 54%|█████▎    | 214657/400000 [00:28<00:24, 7625.13it/s] 54%|█████▍    | 215436/400000 [00:28<00:24, 7670.99it/s] 54%|█████▍    | 216205/400000 [00:28<00:23, 7675.30it/s] 54%|█████▍    | 216975/400000 [00:28<00:23, 7680.17it/s] 54%|█████▍    | 217744/400000 [00:28<00:23, 7626.61it/s] 55%|█████▍    | 218522/400000 [00:28<00:23, 7671.90it/s] 55%|█████▍    | 219296/400000 [00:28<00:23, 7690.20it/s] 55%|█████▌    | 220066/400000 [00:28<00:23, 7640.90it/s] 55%|█████▌    | 220838/400000 [00:28<00:23, 7662.07it/s] 55%|█████▌    | 221605/400000 [00:29<00:23, 7631.51it/s] 56%|█████▌    | 222369/400000 [00:29<00:23, 7623.62it/s] 56%|█████▌    | 223140/400000 [00:29<00:23, 7648.07it/s] 56%|█████▌    | 223905/400000 [00:29<00:23, 7623.25it/s] 56%|█████▌    | 224669/400000 [00:29<00:22, 7626.78it/s] 56%|█████▋    | 225432/400000 [00:29<00:22, 7621.27it/s] 57%|█████▋    | 226196/400000 [00:29<00:22, 7625.30it/s] 57%|█████▋    | 226959/400000 [00:29<00:22, 7576.86it/s] 57%|█████▋    | 227717/400000 [00:29<00:22, 7533.65it/s] 57%|█████▋    | 228495/400000 [00:29<00:22, 7603.91it/s] 57%|█████▋    | 229256/400000 [00:30<00:22, 7567.01it/s] 58%|█████▊    | 230037/400000 [00:30<00:22, 7636.85it/s] 58%|█████▊    | 230812/400000 [00:30<00:22, 7668.28it/s] 58%|█████▊    | 231585/400000 [00:30<00:21, 7684.22it/s] 58%|█████▊    | 232357/400000 [00:30<00:21, 7694.47it/s] 58%|█████▊    | 233127/400000 [00:30<00:21, 7642.01it/s] 58%|█████▊    | 233910/400000 [00:30<00:21, 7697.22it/s] 59%|█████▊    | 234692/400000 [00:30<00:21, 7733.11it/s] 59%|█████▉    | 235466/400000 [00:30<00:21, 7722.03it/s] 59%|█████▉    | 236239/400000 [00:30<00:21, 7710.72it/s] 59%|█████▉    | 237011/400000 [00:31<00:21, 7687.63it/s] 59%|█████▉    | 237781/400000 [00:31<00:21, 7688.94it/s] 60%|█████▉    | 238550/400000 [00:31<00:21, 7666.94it/s] 60%|█████▉    | 239317/400000 [00:31<00:21, 7650.80it/s] 60%|██████    | 240083/400000 [00:31<00:21, 7574.35it/s] 60%|██████    | 240859/400000 [00:31<00:20, 7624.36it/s] 60%|██████    | 241622/400000 [00:31<00:20, 7603.78it/s] 61%|██████    | 242400/400000 [00:31<00:20, 7654.08it/s] 61%|██████    | 243181/400000 [00:31<00:20, 7698.48it/s] 61%|██████    | 243962/400000 [00:32<00:20, 7730.19it/s] 61%|██████    | 244736/400000 [00:32<00:20, 7730.44it/s] 61%|██████▏   | 245510/400000 [00:32<00:20, 7633.87it/s] 62%|██████▏   | 246287/400000 [00:32<00:20, 7673.59it/s] 62%|██████▏   | 247061/400000 [00:32<00:19, 7690.94it/s] 62%|██████▏   | 247840/400000 [00:32<00:19, 7717.69it/s] 62%|██████▏   | 248620/400000 [00:32<00:19, 7740.47it/s] 62%|██████▏   | 249395/400000 [00:32<00:19, 7671.06it/s] 63%|██████▎   | 250168/400000 [00:32<00:19, 7686.70it/s] 63%|██████▎   | 250948/400000 [00:32<00:19, 7718.57it/s] 63%|██████▎   | 251721/400000 [00:33<00:19, 7708.80it/s] 63%|██████▎   | 252492/400000 [00:33<00:19, 7679.44it/s] 63%|██████▎   | 253261/400000 [00:33<00:19, 7613.72it/s] 64%|██████▎   | 254037/400000 [00:33<00:19, 7655.35it/s] 64%|██████▎   | 254804/400000 [00:33<00:18, 7657.96it/s] 64%|██████▍   | 255575/400000 [00:33<00:18, 7671.59it/s] 64%|██████▍   | 256358/400000 [00:33<00:18, 7718.09it/s] 64%|██████▍   | 257130/400000 [00:33<00:18, 7657.24it/s] 64%|██████▍   | 257912/400000 [00:33<00:18, 7704.20it/s] 65%|██████▍   | 258687/400000 [00:33<00:18, 7717.80it/s] 65%|██████▍   | 259459/400000 [00:34<00:18, 7717.62it/s] 65%|██████▌   | 260244/400000 [00:34<00:18, 7754.86it/s] 65%|██████▌   | 261020/400000 [00:34<00:18, 7675.33it/s] 65%|██████▌   | 261800/400000 [00:34<00:17, 7709.52it/s] 66%|██████▌   | 262576/400000 [00:34<00:17, 7722.33it/s] 66%|██████▌   | 263357/400000 [00:34<00:17, 7747.04it/s] 66%|██████▌   | 264136/400000 [00:34<00:17, 7757.63it/s] 66%|██████▌   | 264912/400000 [00:34<00:17, 7666.25it/s] 66%|██████▋   | 265679/400000 [00:34<00:17, 7646.35it/s] 67%|██████▋   | 266459/400000 [00:34<00:17, 7691.03it/s] 67%|██████▋   | 267235/400000 [00:35<00:17, 7711.22it/s] 67%|██████▋   | 268007/400000 [00:35<00:17, 7701.85it/s] 67%|██████▋   | 268778/400000 [00:35<00:17, 7661.06it/s] 67%|██████▋   | 269561/400000 [00:35<00:16, 7708.57it/s] 68%|██████▊   | 270339/400000 [00:35<00:16, 7727.92it/s] 68%|██████▊   | 271113/400000 [00:35<00:16, 7729.09it/s] 68%|██████▊   | 271887/400000 [00:35<00:16, 7731.54it/s] 68%|██████▊   | 272661/400000 [00:35<00:16, 7524.82it/s] 68%|██████▊   | 273442/400000 [00:35<00:16, 7605.56it/s] 69%|██████▊   | 274223/400000 [00:35<00:16, 7665.28it/s] 69%|██████▉   | 275003/400000 [00:36<00:16, 7703.58it/s] 69%|██████▉   | 275783/400000 [00:36<00:16, 7731.80it/s] 69%|██████▉   | 276557/400000 [00:36<00:16, 7689.45it/s] 69%|██████▉   | 277338/400000 [00:36<00:15, 7723.42it/s] 70%|██████▉   | 278117/400000 [00:36<00:15, 7741.97it/s] 70%|██████▉   | 278893/400000 [00:36<00:15, 7746.03it/s] 70%|██████▉   | 279670/400000 [00:36<00:15, 7751.66it/s] 70%|███████   | 280446/400000 [00:36<00:15, 7702.70it/s] 70%|███████   | 281229/400000 [00:36<00:15, 7739.76it/s] 71%|███████   | 282004/400000 [00:36<00:15, 7741.88it/s] 71%|███████   | 282779/400000 [00:37<00:15, 7739.01it/s] 71%|███████   | 283559/400000 [00:37<00:15, 7754.40it/s] 71%|███████   | 284335/400000 [00:37<00:15, 7668.65it/s] 71%|███████▏  | 285110/400000 [00:37<00:14, 7692.14it/s] 71%|███████▏  | 285885/400000 [00:37<00:14, 7708.09it/s] 72%|███████▏  | 286667/400000 [00:37<00:14, 7739.34it/s] 72%|███████▏  | 287445/400000 [00:37<00:14, 7749.51it/s] 72%|███████▏  | 288221/400000 [00:37<00:14, 7663.11it/s] 72%|███████▏  | 288996/400000 [00:37<00:14, 7687.54it/s] 72%|███████▏  | 289769/400000 [00:37<00:14, 7699.56it/s] 73%|███████▎  | 290541/400000 [00:38<00:14, 7702.74it/s] 73%|███████▎  | 291313/400000 [00:38<00:14, 7706.71it/s] 73%|███████▎  | 292084/400000 [00:38<00:14, 7550.14it/s] 73%|███████▎  | 292860/400000 [00:38<00:14, 7611.06it/s] 73%|███████▎  | 293637/400000 [00:38<00:13, 7656.03it/s] 74%|███████▎  | 294422/400000 [00:38<00:13, 7711.17it/s] 74%|███████▍  | 295202/400000 [00:38<00:13, 7737.44it/s] 74%|███████▍  | 295977/400000 [00:38<00:13, 7678.44it/s] 74%|███████▍  | 296751/400000 [00:38<00:13, 7695.41it/s] 74%|███████▍  | 297527/400000 [00:38<00:13, 7712.24it/s] 75%|███████▍  | 298301/400000 [00:39<00:13, 7718.56it/s] 75%|███████▍  | 299082/400000 [00:39<00:13, 7743.95it/s] 75%|███████▍  | 299857/400000 [00:39<00:13, 7680.21it/s] 75%|███████▌  | 300635/400000 [00:39<00:12, 7708.97it/s] 75%|███████▌  | 301412/400000 [00:39<00:12, 7726.99it/s] 76%|███████▌  | 302187/400000 [00:39<00:12, 7731.64it/s] 76%|███████▌  | 302963/400000 [00:39<00:12, 7739.36it/s] 76%|███████▌  | 303737/400000 [00:39<00:12, 7566.21it/s] 76%|███████▌  | 304509/400000 [00:39<00:12, 7610.82it/s] 76%|███████▋  | 305286/400000 [00:39<00:12, 7655.45it/s] 77%|███████▋  | 306053/400000 [00:40<00:12, 7525.06it/s] 77%|███████▋  | 306831/400000 [00:40<00:12, 7599.68it/s] 77%|███████▋  | 307592/400000 [00:40<00:12, 7589.16it/s] 77%|███████▋  | 308374/400000 [00:40<00:11, 7654.88it/s] 77%|███████▋  | 309156/400000 [00:40<00:11, 7702.47it/s] 77%|███████▋  | 309940/400000 [00:40<00:11, 7741.70it/s] 78%|███████▊  | 310715/400000 [00:40<00:11, 7727.16it/s] 78%|███████▊  | 311488/400000 [00:40<00:11, 7683.76it/s] 78%|███████▊  | 312264/400000 [00:40<00:11, 7703.54it/s] 78%|███████▊  | 313043/400000 [00:40<00:11, 7729.23it/s] 78%|███████▊  | 313826/400000 [00:41<00:11, 7757.57it/s] 79%|███████▊  | 314602/400000 [00:41<00:11, 7719.98it/s] 79%|███████▉  | 315375/400000 [00:41<00:11, 7686.62it/s] 79%|███████▉  | 316144/400000 [00:41<00:11, 7519.65it/s] 79%|███████▉  | 316919/400000 [00:41<00:10, 7586.18it/s] 79%|███████▉  | 317698/400000 [00:41<00:10, 7645.02it/s] 80%|███████▉  | 318477/400000 [00:41<00:10, 7687.88it/s] 80%|███████▉  | 319247/400000 [00:41<00:10, 7597.82it/s] 80%|████████  | 320022/400000 [00:41<00:10, 7640.41it/s] 80%|████████  | 320799/400000 [00:41<00:10, 7676.68it/s] 80%|████████  | 321578/400000 [00:42<00:10, 7710.21it/s] 81%|████████  | 322350/400000 [00:42<00:10, 7545.56it/s] 81%|████████  | 323106/400000 [00:42<00:10, 7545.54it/s] 81%|████████  | 323881/400000 [00:42<00:10, 7604.46it/s] 81%|████████  | 324659/400000 [00:42<00:09, 7655.62it/s] 81%|████████▏ | 325440/400000 [00:42<00:09, 7698.83it/s] 82%|████████▏ | 326221/400000 [00:42<00:09, 7730.56it/s] 82%|████████▏ | 326995/400000 [00:42<00:09, 7643.47it/s] 82%|████████▏ | 327766/400000 [00:42<00:09, 7662.49it/s] 82%|████████▏ | 328543/400000 [00:43<00:09, 7693.86it/s] 82%|████████▏ | 329313/400000 [00:43<00:09, 7686.32it/s] 83%|████████▎ | 330082/400000 [00:43<00:09, 7669.00it/s] 83%|████████▎ | 330850/400000 [00:43<00:09, 7658.83it/s] 83%|████████▎ | 331617/400000 [00:43<00:08, 7662.07it/s] 83%|████████▎ | 332396/400000 [00:43<00:08, 7698.50it/s] 83%|████████▎ | 333176/400000 [00:43<00:08, 7728.20it/s] 83%|████████▎ | 333955/400000 [00:43<00:08, 7746.28it/s] 84%|████████▎ | 334730/400000 [00:43<00:08, 7729.05it/s] 84%|████████▍ | 335503/400000 [00:43<00:08, 7692.52it/s] 84%|████████▍ | 336283/400000 [00:44<00:08, 7722.21it/s] 84%|████████▍ | 337056/400000 [00:44<00:08, 7699.81it/s] 84%|████████▍ | 337827/400000 [00:44<00:08, 7683.57it/s] 85%|████████▍ | 338596/400000 [00:44<00:08, 7654.70it/s] 85%|████████▍ | 339362/400000 [00:44<00:07, 7579.90it/s] 85%|████████▌ | 340141/400000 [00:44<00:07, 7639.63it/s] 85%|████████▌ | 340921/400000 [00:44<00:07, 7685.63it/s] 85%|████████▌ | 341690/400000 [00:44<00:07, 7685.61it/s] 86%|████████▌ | 342459/400000 [00:44<00:07, 7673.85it/s] 86%|████████▌ | 343227/400000 [00:44<00:07, 7667.24it/s] 86%|████████▌ | 344005/400000 [00:45<00:07, 7697.98it/s] 86%|████████▌ | 344785/400000 [00:45<00:07, 7727.40it/s] 86%|████████▋ | 345566/400000 [00:45<00:07, 7750.04it/s] 87%|████████▋ | 346345/400000 [00:45<00:06, 7761.84it/s] 87%|████████▋ | 347122/400000 [00:45<00:06, 7705.72it/s] 87%|████████▋ | 347893/400000 [00:45<00:06, 7635.88it/s] 87%|████████▋ | 348671/400000 [00:45<00:06, 7677.63it/s] 87%|████████▋ | 349452/400000 [00:45<00:06, 7715.20it/s] 88%|████████▊ | 350231/400000 [00:45<00:06, 7735.43it/s] 88%|████████▊ | 351005/400000 [00:45<00:06, 7669.81it/s] 88%|████████▊ | 351780/400000 [00:46<00:06, 7691.05it/s] 88%|████████▊ | 352553/400000 [00:46<00:06, 7700.87it/s] 88%|████████▊ | 353324/400000 [00:46<00:06, 7694.50it/s] 89%|████████▊ | 354102/400000 [00:46<00:05, 7719.86it/s] 89%|████████▊ | 354875/400000 [00:46<00:05, 7622.77it/s] 89%|████████▉ | 355648/400000 [00:46<00:05, 7654.21it/s] 89%|████████▉ | 356422/400000 [00:46<00:05, 7677.68it/s] 89%|████████▉ | 357200/400000 [00:46<00:05, 7707.51it/s] 89%|████████▉ | 357971/400000 [00:46<00:05, 7688.99it/s] 90%|████████▉ | 358741/400000 [00:46<00:05, 7640.39it/s] 90%|████████▉ | 359516/400000 [00:47<00:05, 7670.98it/s] 90%|█████████ | 360296/400000 [00:47<00:05, 7709.20it/s] 90%|█████████ | 361076/400000 [00:47<00:05, 7735.32it/s] 90%|█████████ | 361854/400000 [00:47<00:04, 7747.47it/s] 91%|█████████ | 362629/400000 [00:47<00:04, 7686.24it/s] 91%|█████████ | 363405/400000 [00:47<00:04, 7706.28it/s] 91%|█████████ | 364185/400000 [00:47<00:04, 7731.89it/s] 91%|█████████ | 364964/400000 [00:47<00:04, 7747.37it/s] 91%|█████████▏| 365741/400000 [00:47<00:04, 7751.26it/s] 92%|█████████▏| 366517/400000 [00:47<00:04, 7699.83it/s] 92%|█████████▏| 367288/400000 [00:48<00:04, 7684.24it/s] 92%|█████████▏| 368057/400000 [00:48<00:04, 7658.47it/s] 92%|█████████▏| 368823/400000 [00:48<00:04, 7627.08it/s] 92%|█████████▏| 369594/400000 [00:48<00:03, 7650.82it/s] 93%|█████████▎| 370360/400000 [00:48<00:03, 7583.42it/s] 93%|█████████▎| 371137/400000 [00:48<00:03, 7636.87it/s] 93%|█████████▎| 371909/400000 [00:48<00:03, 7661.14it/s] 93%|█████████▎| 372681/400000 [00:48<00:03, 7675.93it/s] 93%|█████████▎| 373449/400000 [00:48<00:03, 7653.55it/s] 94%|█████████▎| 374215/400000 [00:48<00:03, 7579.96it/s] 94%|█████████▎| 374990/400000 [00:49<00:03, 7628.02it/s] 94%|█████████▍| 375771/400000 [00:49<00:03, 7680.21it/s] 94%|█████████▍| 376540/400000 [00:49<00:03, 7646.35it/s] 94%|█████████▍| 377308/400000 [00:49<00:02, 7655.10it/s] 95%|█████████▍| 378074/400000 [00:49<00:02, 7638.25it/s] 95%|█████████▍| 378840/400000 [00:49<00:02, 7641.96it/s] 95%|█████████▍| 379616/400000 [00:49<00:02, 7675.68it/s] 95%|█████████▌| 380387/400000 [00:49<00:02, 7684.85it/s] 95%|█████████▌| 381157/400000 [00:49<00:02, 7687.91it/s] 95%|█████████▌| 381926/400000 [00:49<00:02, 7642.67it/s] 96%|█████████▌| 382695/400000 [00:50<00:02, 7655.81it/s] 96%|█████████▌| 383469/400000 [00:50<00:02, 7679.12it/s] 96%|█████████▌| 384245/400000 [00:50<00:02, 7701.10it/s] 96%|█████████▋| 385023/400000 [00:50<00:01, 7724.49it/s] 96%|█████████▋| 385796/400000 [00:50<00:01, 7657.55it/s] 97%|█████████▋| 386577/400000 [00:50<00:01, 7700.77it/s] 97%|█████████▋| 387356/400000 [00:50<00:01, 7725.03it/s] 97%|█████████▋| 388129/400000 [00:50<00:01, 7723.18it/s] 97%|█████████▋| 388903/400000 [00:50<00:01, 7726.17it/s] 97%|█████████▋| 389676/400000 [00:50<00:01, 7657.96it/s] 98%|█████████▊| 390458/400000 [00:51<00:01, 7705.57it/s] 98%|█████████▊| 391241/400000 [00:51<00:01, 7740.64it/s] 98%|█████████▊| 392016/400000 [00:51<00:01, 7719.94it/s] 98%|█████████▊| 392789/400000 [00:51<00:00, 7708.30it/s] 98%|█████████▊| 393560/400000 [00:51<00:00, 7544.96it/s] 99%|█████████▊| 394337/400000 [00:51<00:00, 7609.27it/s] 99%|█████████▉| 395116/400000 [00:51<00:00, 7661.14it/s] 99%|█████████▉| 395893/400000 [00:51<00:00, 7691.65it/s] 99%|█████████▉| 396670/400000 [00:51<00:00, 7714.62it/s] 99%|█████████▉| 397442/400000 [00:51<00:00, 7681.88it/s]100%|█████████▉| 398213/400000 [00:52<00:00, 7690.10it/s]100%|█████████▉| 398983/400000 [00:52<00:00, 7366.09it/s]100%|█████████▉| 399759/400000 [00:52<00:00, 7479.40it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7644.82it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f81377124e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.0108536709531694 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.01081740995713301 	 Accuracy: 70

  model saves at 70% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  ### Calculate Metrics    ######################################## 

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

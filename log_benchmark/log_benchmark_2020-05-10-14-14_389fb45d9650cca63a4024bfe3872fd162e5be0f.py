
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/389fb45d9650cca63a4024bfe3872fd162e5be0f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '389fb45d9650cca63a4024bfe3872fd162e5be0f', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/389fb45d9650cca63a4024bfe3872fd162e5be0f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/389fb45d9650cca63a4024bfe3872fd162e5be0f

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fd7cf19f470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 14:14:47.568302
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 14:14:47.572246
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 14:14:47.575641
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 14:14:47.578995
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fd7a791eef0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353972.1562
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 211519.7344
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 118040.4844
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 58132.1328
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 30565.9434
Epoch 6/10

1/1 [==============================] - 0s 90ms/step - loss: 18049.2520
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 11855.7705
Epoch 8/10

1/1 [==============================] - 0s 91ms/step - loss: 8417.4844
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 6332.9136
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 5006.1724

  #### Inference Need return ypred, ytrue ######################### 
[[  0.27955103   8.027704     9.66627     12.845969     9.687016
    9.419851    11.182077    10.593372    10.93451      8.699664
   10.339674     8.760859    10.822442    10.613782     9.22054
    9.469662     9.647644     9.634924     8.203164     9.235016
    9.291108    10.65686     11.337231     8.362933    11.633388
    9.934678     9.656028    10.421693    11.983723     8.293214
    8.916082     7.5842733    9.199034    11.211034    10.512229
   10.463863    10.004901     9.084613     9.529079     9.280899
    8.336473     8.198759    10.604508     9.941236    10.278456
   10.210743    10.234216     9.487845     8.774209     7.961303
   11.378402     9.663394    11.283125     9.170227     8.458509
    8.219393    10.65577      9.980225    11.084179     9.10535
    0.7510643   -1.8510942    1.2246032   -0.52471125  -2.0022488
    2.8642714   -0.5995517   -0.7062633    0.72823405   0.40059656
    0.56806755   1.9181828    0.01525062   0.9505631    1.7370602
    0.2256408    1.4324691   -2.034892     1.1037388   -0.6511294
   -0.24957395   0.45816356  -0.02438378   1.6056874    1.2532697
   -2.3052492   -0.40501484  -0.61596024   0.9524747    0.31316704
    0.6347912    0.34991455  -0.23717663  -1.5737985   -1.182866
   -0.47897628   2.0857637    0.9685569   -2.4571643   -0.04984504
    2.1607704   -1.0596824   -0.87366784  -1.9156843   -1.7457688
   -0.5300015   -0.28023252   0.40566805   1.2146738    0.7256056
    2.4824462   -2.4572158   -1.3332326    2.8009744    1.5131376
    0.039518    -0.75272596   2.1575925    0.8391414    1.9557183
    2.422263     0.10681652  -0.25641185  -0.11048357   2.5205116
    2.2783952    1.1547567   -0.38313663  -2.1209736    0.35251743
   -0.5816743   -1.515635    -0.08025742   1.3558546    0.38049442
   -1.590372     0.78554666  -2.2284725   -1.7951374    0.93952143
   -0.8659091   -1.7638423    0.22711325  -0.10564774   1.3250097
   -0.7502568   -0.27728587   0.33554757   2.855759    -1.469904
   -0.30485958  -1.4865174   -1.7243955   -0.02351975   2.192005
   -1.0261649    0.77856284   1.7566912   -1.2842747    0.5308869
   -0.09994555  -1.0735955   -0.2884437    1.9980009   -0.58118486
   -0.4396132   -0.6570311    0.56221014   0.01466404  -1.0652677
    2.5705543   -0.11062539   0.02173057  -0.37362796  -0.98908246
   -0.6748463    1.3664067    1.7882032    1.290442     2.1598482
    0.34194905  11.535182    10.233549    12.3911915    9.782476
    8.486933    10.958588    10.787828    10.778103     9.633831
   11.212314     8.236234    11.103965     9.204959     8.2660475
   10.248351     8.021004     9.926106    10.545834     8.526037
    8.395935     8.210016     8.139193    11.148135     9.960152
    8.560653     9.676339     9.357015    10.288234     8.12599
    9.984241     9.375616    10.4558       9.266386     8.885235
   11.758115     9.763669    12.561133    11.396083     8.316731
   11.179743    10.969713     9.318801     9.421827    10.838303
    9.817342    11.26557      7.677832    10.253722    11.096274
    9.384092     8.512578    12.219961     8.082285    10.768538
   10.119747     7.179206     8.090413    10.643213    10.723099
    0.89400053   1.3468037    0.563498     0.60230976   0.21604955
    0.46117938   0.38756794   0.4198057    0.42611003   1.4252594
    1.9264234    0.27354038   3.3170376    2.2450118    1.6004512
    0.22858292   3.140994     0.4392665    1.8735701    2.4985576
    0.58081734   0.74279773   1.7772659    1.5617564    0.6361285
    0.72712266   0.30541068   0.788026     2.2705526    1.1207852
    2.2963471    0.49594414   1.1788782    0.51322633   1.0952477
    1.2081752    0.08464772   2.4154773    1.823224     1.1801062
    0.27158517   0.9180306    0.05127293   1.177998     0.7377546
    0.84351295   3.158019     2.2896585    1.7026744    2.0872917
    1.9921952    0.43868732   0.08910614   1.2698319    2.693438
    0.6698674    0.48635364   1.6306096    0.83737767   2.2165675
    1.037043     0.84796476   0.15140778   2.754747     2.2087865
    0.21501058   0.39235318   1.3637799    1.0742543    3.1208863
    3.2713313    0.49392915   3.086969     0.04665202   1.1151831
    0.23873758   2.779102     0.15584111   0.49854082   1.972365
    1.7303126    0.43620956   0.82037944   0.36018914   0.18837756
    0.1833297    0.4229179    0.505971     1.4080546    2.1210194
    0.79372334   1.5323749    0.69518983   2.1106591    1.320371
    2.4534626    2.0956922    0.7230475    0.26207215   2.2005439
    0.08147848   0.9407182    0.7599444    0.42518073   0.29389834
    1.6612659    1.0375471    1.3087392    1.654296     1.8467121
    0.27384567   1.3283255    0.36038232   0.17464036   2.5906968
    1.9307035    1.833428     0.38265443   2.765993     0.2598536
    1.6114354  -10.1604595   -8.396181  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 14:14:56.338832
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.8263
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 14:14:56.342884
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8642.29
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 14:14:56.346323
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.9579
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 14:14:56.349508
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -772.978
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140564197586536
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140563239019464
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140563239019968
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140563239020472
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140563239020976
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140563239021480

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fd7b4f73b00> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.612588
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.572688
grad_step = 000002, loss = 0.538237
grad_step = 000003, loss = 0.504546
grad_step = 000004, loss = 0.472970
grad_step = 000005, loss = 0.449724
grad_step = 000006, loss = 0.433748
grad_step = 000007, loss = 0.414148
grad_step = 000008, loss = 0.390825
grad_step = 000009, loss = 0.370372
grad_step = 000010, loss = 0.354841
grad_step = 000011, loss = 0.343036
grad_step = 000012, loss = 0.328730
grad_step = 000013, loss = 0.312926
grad_step = 000014, loss = 0.297224
grad_step = 000015, loss = 0.283074
grad_step = 000016, loss = 0.270108
grad_step = 000017, loss = 0.257772
grad_step = 000018, loss = 0.245035
grad_step = 000019, loss = 0.232437
grad_step = 000020, loss = 0.221486
grad_step = 000021, loss = 0.211829
grad_step = 000022, loss = 0.202168
grad_step = 000023, loss = 0.192391
grad_step = 000024, loss = 0.183026
grad_step = 000025, loss = 0.174485
grad_step = 000026, loss = 0.166397
grad_step = 000027, loss = 0.158052
grad_step = 000028, loss = 0.149415
grad_step = 000029, loss = 0.141140
grad_step = 000030, loss = 0.133477
grad_step = 000031, loss = 0.126102
grad_step = 000032, loss = 0.118881
grad_step = 000033, loss = 0.111990
grad_step = 000034, loss = 0.105489
grad_step = 000035, loss = 0.099312
grad_step = 000036, loss = 0.093330
grad_step = 000037, loss = 0.087448
grad_step = 000038, loss = 0.081748
grad_step = 000039, loss = 0.076296
grad_step = 000040, loss = 0.071040
grad_step = 000041, loss = 0.065989
grad_step = 000042, loss = 0.061156
grad_step = 000043, loss = 0.056596
grad_step = 000044, loss = 0.052408
grad_step = 000045, loss = 0.048467
grad_step = 000046, loss = 0.044642
grad_step = 000047, loss = 0.040999
grad_step = 000048, loss = 0.037575
grad_step = 000049, loss = 0.034362
grad_step = 000050, loss = 0.031317
grad_step = 000051, loss = 0.028486
grad_step = 000052, loss = 0.025902
grad_step = 000053, loss = 0.023520
grad_step = 000054, loss = 0.021322
grad_step = 000055, loss = 0.019310
grad_step = 000056, loss = 0.017482
grad_step = 000057, loss = 0.015803
grad_step = 000058, loss = 0.014271
grad_step = 000059, loss = 0.012867
grad_step = 000060, loss = 0.011594
grad_step = 000061, loss = 0.010459
grad_step = 000062, loss = 0.009445
grad_step = 000063, loss = 0.008523
grad_step = 000064, loss = 0.007698
grad_step = 000065, loss = 0.006969
grad_step = 000066, loss = 0.006318
grad_step = 000067, loss = 0.005735
grad_step = 000068, loss = 0.005220
grad_step = 000069, loss = 0.004764
grad_step = 000070, loss = 0.004362
grad_step = 000071, loss = 0.004014
grad_step = 000072, loss = 0.003711
grad_step = 000073, loss = 0.003447
grad_step = 000074, loss = 0.003220
grad_step = 000075, loss = 0.003029
grad_step = 000076, loss = 0.002866
grad_step = 000077, loss = 0.002727
grad_step = 000078, loss = 0.002611
grad_step = 000079, loss = 0.002514
grad_step = 000080, loss = 0.002429
grad_step = 000081, loss = 0.002358
grad_step = 000082, loss = 0.002300
grad_step = 000083, loss = 0.002252
grad_step = 000084, loss = 0.002211
grad_step = 000085, loss = 0.002177
grad_step = 000086, loss = 0.002148
grad_step = 000087, loss = 0.002121
grad_step = 000088, loss = 0.002097
grad_step = 000089, loss = 0.002074
grad_step = 000090, loss = 0.002055
grad_step = 000091, loss = 0.002038
grad_step = 000092, loss = 0.002021
grad_step = 000093, loss = 0.002007
grad_step = 000094, loss = 0.001995
grad_step = 000095, loss = 0.001986
grad_step = 000096, loss = 0.001976
grad_step = 000097, loss = 0.001961
grad_step = 000098, loss = 0.001943
grad_step = 000099, loss = 0.001929
grad_step = 000100, loss = 0.001920
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001915
grad_step = 000102, loss = 0.001911
grad_step = 000103, loss = 0.001906
grad_step = 000104, loss = 0.001896
grad_step = 000105, loss = 0.001883
grad_step = 000106, loss = 0.001869
grad_step = 000107, loss = 0.001858
grad_step = 000108, loss = 0.001851
grad_step = 000109, loss = 0.001847
grad_step = 000110, loss = 0.001847
grad_step = 000111, loss = 0.001851
grad_step = 000112, loss = 0.001862
grad_step = 000113, loss = 0.001868
grad_step = 000114, loss = 0.001863
grad_step = 000115, loss = 0.001831
grad_step = 000116, loss = 0.001802
grad_step = 000117, loss = 0.001794
grad_step = 000118, loss = 0.001804
grad_step = 000119, loss = 0.001817
grad_step = 000120, loss = 0.001812
grad_step = 000121, loss = 0.001793
grad_step = 000122, loss = 0.001770
grad_step = 000123, loss = 0.001760
grad_step = 000124, loss = 0.001764
grad_step = 000125, loss = 0.001772
grad_step = 000126, loss = 0.001778
grad_step = 000127, loss = 0.001772
grad_step = 000128, loss = 0.001759
grad_step = 000129, loss = 0.001741
grad_step = 000130, loss = 0.001727
grad_step = 000131, loss = 0.001721
grad_step = 000132, loss = 0.001721
grad_step = 000133, loss = 0.001726
grad_step = 000134, loss = 0.001733
grad_step = 000135, loss = 0.001745
grad_step = 000136, loss = 0.001756
grad_step = 000137, loss = 0.001770
grad_step = 000138, loss = 0.001760
grad_step = 000139, loss = 0.001738
grad_step = 000140, loss = 0.001701
grad_step = 000141, loss = 0.001680
grad_step = 000142, loss = 0.001682
grad_step = 000143, loss = 0.001696
grad_step = 000144, loss = 0.001711
grad_step = 000145, loss = 0.001708
grad_step = 000146, loss = 0.001694
grad_step = 000147, loss = 0.001671
grad_step = 000148, loss = 0.001655
grad_step = 000149, loss = 0.001649
grad_step = 000150, loss = 0.001653
grad_step = 000151, loss = 0.001662
grad_step = 000152, loss = 0.001672
grad_step = 000153, loss = 0.001684
grad_step = 000154, loss = 0.001687
grad_step = 000155, loss = 0.001688
grad_step = 000156, loss = 0.001669
grad_step = 000157, loss = 0.001647
grad_step = 000158, loss = 0.001623
grad_step = 000159, loss = 0.001610
grad_step = 000160, loss = 0.001609
grad_step = 000161, loss = 0.001616
grad_step = 000162, loss = 0.001625
grad_step = 000163, loss = 0.001630
grad_step = 000164, loss = 0.001634
grad_step = 000165, loss = 0.001629
grad_step = 000166, loss = 0.001622
grad_step = 000167, loss = 0.001606
grad_step = 000168, loss = 0.001591
grad_step = 000169, loss = 0.001577
grad_step = 000170, loss = 0.001567
grad_step = 000171, loss = 0.001561
grad_step = 000172, loss = 0.001559
grad_step = 000173, loss = 0.001560
grad_step = 000174, loss = 0.001565
grad_step = 000175, loss = 0.001577
grad_step = 000176, loss = 0.001599
grad_step = 000177, loss = 0.001638
grad_step = 000178, loss = 0.001680
grad_step = 000179, loss = 0.001718
grad_step = 000180, loss = 0.001669
grad_step = 000181, loss = 0.001584
grad_step = 000182, loss = 0.001519
grad_step = 000183, loss = 0.001536
grad_step = 000184, loss = 0.001585
grad_step = 000185, loss = 0.001579
grad_step = 000186, loss = 0.001533
grad_step = 000187, loss = 0.001505
grad_step = 000188, loss = 0.001518
grad_step = 000189, loss = 0.001532
grad_step = 000190, loss = 0.001521
grad_step = 000191, loss = 0.001506
grad_step = 000192, loss = 0.001496
grad_step = 000193, loss = 0.001485
grad_step = 000194, loss = 0.001483
grad_step = 000195, loss = 0.001494
grad_step = 000196, loss = 0.001497
grad_step = 000197, loss = 0.001486
grad_step = 000198, loss = 0.001464
grad_step = 000199, loss = 0.001449
grad_step = 000200, loss = 0.001451
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001462
grad_step = 000202, loss = 0.001466
grad_step = 000203, loss = 0.001459
grad_step = 000204, loss = 0.001450
grad_step = 000205, loss = 0.001437
grad_step = 000206, loss = 0.001426
grad_step = 000207, loss = 0.001421
grad_step = 000208, loss = 0.001424
grad_step = 000209, loss = 0.001429
grad_step = 000210, loss = 0.001436
grad_step = 000211, loss = 0.001450
grad_step = 000212, loss = 0.001467
grad_step = 000213, loss = 0.001494
grad_step = 000214, loss = 0.001503
grad_step = 000215, loss = 0.001530
grad_step = 000216, loss = 0.001531
grad_step = 000217, loss = 0.001570
grad_step = 000218, loss = 0.001600
grad_step = 000219, loss = 0.001561
grad_step = 000220, loss = 0.001459
grad_step = 000221, loss = 0.001383
grad_step = 000222, loss = 0.001439
grad_step = 000223, loss = 0.001503
grad_step = 000224, loss = 0.001450
grad_step = 000225, loss = 0.001394
grad_step = 000226, loss = 0.001411
grad_step = 000227, loss = 0.001430
grad_step = 000228, loss = 0.001407
grad_step = 000229, loss = 0.001370
grad_step = 000230, loss = 0.001384
grad_step = 000231, loss = 0.001419
grad_step = 000232, loss = 0.001380
grad_step = 000233, loss = 0.001354
grad_step = 000234, loss = 0.001371
grad_step = 000235, loss = 0.001376
grad_step = 000236, loss = 0.001364
grad_step = 000237, loss = 0.001347
grad_step = 000238, loss = 0.001345
grad_step = 000239, loss = 0.001354
grad_step = 000240, loss = 0.001352
grad_step = 000241, loss = 0.001339
grad_step = 000242, loss = 0.001332
grad_step = 000243, loss = 0.001339
grad_step = 000244, loss = 0.001348
grad_step = 000245, loss = 0.001348
grad_step = 000246, loss = 0.001348
grad_step = 000247, loss = 0.001368
grad_step = 000248, loss = 0.001379
grad_step = 000249, loss = 0.001396
grad_step = 000250, loss = 0.001364
grad_step = 000251, loss = 0.001345
grad_step = 000252, loss = 0.001331
grad_step = 000253, loss = 0.001324
grad_step = 000254, loss = 0.001318
grad_step = 000255, loss = 0.001310
grad_step = 000256, loss = 0.001309
grad_step = 000257, loss = 0.001316
grad_step = 000258, loss = 0.001327
grad_step = 000259, loss = 0.001334
grad_step = 000260, loss = 0.001330
grad_step = 000261, loss = 0.001321
grad_step = 000262, loss = 0.001316
grad_step = 000263, loss = 0.001312
grad_step = 000264, loss = 0.001318
grad_step = 000265, loss = 0.001310
grad_step = 000266, loss = 0.001307
grad_step = 000267, loss = 0.001295
grad_step = 000268, loss = 0.001291
grad_step = 000269, loss = 0.001290
grad_step = 000270, loss = 0.001297
grad_step = 000271, loss = 0.001309
grad_step = 000272, loss = 0.001320
grad_step = 000273, loss = 0.001333
grad_step = 000274, loss = 0.001337
grad_step = 000275, loss = 0.001338
grad_step = 000276, loss = 0.001352
grad_step = 000277, loss = 0.001344
grad_step = 000278, loss = 0.001351
grad_step = 000279, loss = 0.001307
grad_step = 000280, loss = 0.001270
grad_step = 000281, loss = 0.001250
grad_step = 000282, loss = 0.001258
grad_step = 000283, loss = 0.001277
grad_step = 000284, loss = 0.001277
grad_step = 000285, loss = 0.001269
grad_step = 000286, loss = 0.001255
grad_step = 000287, loss = 0.001257
grad_step = 000288, loss = 0.001274
grad_step = 000289, loss = 0.001285
grad_step = 000290, loss = 0.001290
grad_step = 000291, loss = 0.001276
grad_step = 000292, loss = 0.001263
grad_step = 000293, loss = 0.001253
grad_step = 000294, loss = 0.001252
grad_step = 000295, loss = 0.001260
grad_step = 000296, loss = 0.001255
grad_step = 000297, loss = 0.001250
grad_step = 000298, loss = 0.001232
grad_step = 000299, loss = 0.001219
grad_step = 000300, loss = 0.001214
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001217
grad_step = 000302, loss = 0.001222
grad_step = 000303, loss = 0.001223
grad_step = 000304, loss = 0.001220
grad_step = 000305, loss = 0.001213
grad_step = 000306, loss = 0.001207
grad_step = 000307, loss = 0.001204
grad_step = 000308, loss = 0.001207
grad_step = 000309, loss = 0.001216
grad_step = 000310, loss = 0.001235
grad_step = 000311, loss = 0.001281
grad_step = 000312, loss = 0.001383
grad_step = 000313, loss = 0.001536
grad_step = 000314, loss = 0.001703
grad_step = 000315, loss = 0.001508
grad_step = 000316, loss = 0.001256
grad_step = 000317, loss = 0.001277
grad_step = 000318, loss = 0.001396
grad_step = 000319, loss = 0.001338
grad_step = 000320, loss = 0.001202
grad_step = 000321, loss = 0.001297
grad_step = 000322, loss = 0.001372
grad_step = 000323, loss = 0.001220
grad_step = 000324, loss = 0.001225
grad_step = 000325, loss = 0.001302
grad_step = 000326, loss = 0.001214
grad_step = 000327, loss = 0.001210
grad_step = 000328, loss = 0.001275
grad_step = 000329, loss = 0.001213
grad_step = 000330, loss = 0.001182
grad_step = 000331, loss = 0.001230
grad_step = 000332, loss = 0.001203
grad_step = 000333, loss = 0.001178
grad_step = 000334, loss = 0.001210
grad_step = 000335, loss = 0.001199
grad_step = 000336, loss = 0.001166
grad_step = 000337, loss = 0.001179
grad_step = 000338, loss = 0.001187
grad_step = 000339, loss = 0.001167
grad_step = 000340, loss = 0.001167
grad_step = 000341, loss = 0.001181
grad_step = 000342, loss = 0.001169
grad_step = 000343, loss = 0.001155
grad_step = 000344, loss = 0.001162
grad_step = 000345, loss = 0.001164
grad_step = 000346, loss = 0.001152
grad_step = 000347, loss = 0.001152
grad_step = 000348, loss = 0.001159
grad_step = 000349, loss = 0.001154
grad_step = 000350, loss = 0.001146
grad_step = 000351, loss = 0.001146
grad_step = 000352, loss = 0.001148
grad_step = 000353, loss = 0.001142
grad_step = 000354, loss = 0.001137
grad_step = 000355, loss = 0.001138
grad_step = 000356, loss = 0.001139
grad_step = 000357, loss = 0.001136
grad_step = 000358, loss = 0.001132
grad_step = 000359, loss = 0.001131
grad_step = 000360, loss = 0.001132
grad_step = 000361, loss = 0.001131
grad_step = 000362, loss = 0.001128
grad_step = 000363, loss = 0.001127
grad_step = 000364, loss = 0.001128
grad_step = 000365, loss = 0.001130
grad_step = 000366, loss = 0.001131
grad_step = 000367, loss = 0.001134
grad_step = 000368, loss = 0.001137
grad_step = 000369, loss = 0.001149
grad_step = 000370, loss = 0.001158
grad_step = 000371, loss = 0.001178
grad_step = 000372, loss = 0.001182
grad_step = 000373, loss = 0.001195
grad_step = 000374, loss = 0.001175
grad_step = 000375, loss = 0.001159
grad_step = 000376, loss = 0.001128
grad_step = 000377, loss = 0.001109
grad_step = 000378, loss = 0.001104
grad_step = 000379, loss = 0.001111
grad_step = 000380, loss = 0.001123
grad_step = 000381, loss = 0.001129
grad_step = 000382, loss = 0.001133
grad_step = 000383, loss = 0.001127
grad_step = 000384, loss = 0.001126
grad_step = 000385, loss = 0.001125
grad_step = 000386, loss = 0.001135
grad_step = 000387, loss = 0.001143
grad_step = 000388, loss = 0.001157
grad_step = 000389, loss = 0.001154
grad_step = 000390, loss = 0.001147
grad_step = 000391, loss = 0.001126
grad_step = 000392, loss = 0.001109
grad_step = 000393, loss = 0.001098
grad_step = 000394, loss = 0.001098
grad_step = 000395, loss = 0.001102
grad_step = 000396, loss = 0.001108
grad_step = 000397, loss = 0.001110
grad_step = 000398, loss = 0.001107
grad_step = 000399, loss = 0.001103
grad_step = 000400, loss = 0.001097
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001095
grad_step = 000402, loss = 0.001095
grad_step = 000403, loss = 0.001101
grad_step = 000404, loss = 0.001109
grad_step = 000405, loss = 0.001126
grad_step = 000406, loss = 0.001138
grad_step = 000407, loss = 0.001161
grad_step = 000408, loss = 0.001160
grad_step = 000409, loss = 0.001167
grad_step = 000410, loss = 0.001143
grad_step = 000411, loss = 0.001123
grad_step = 000412, loss = 0.001097
grad_step = 000413, loss = 0.001082
grad_step = 000414, loss = 0.001076
grad_step = 000415, loss = 0.001076
grad_step = 000416, loss = 0.001080
grad_step = 000417, loss = 0.001077
grad_step = 000418, loss = 0.001077
grad_step = 000419, loss = 0.001074
grad_step = 000420, loss = 0.001078
grad_step = 000421, loss = 0.001088
grad_step = 000422, loss = 0.001114
grad_step = 000423, loss = 0.001141
grad_step = 000424, loss = 0.001187
grad_step = 000425, loss = 0.001192
grad_step = 000426, loss = 0.001183
grad_step = 000427, loss = 0.001115
grad_step = 000428, loss = 0.001058
grad_step = 000429, loss = 0.001038
grad_step = 000430, loss = 0.001060
grad_step = 000431, loss = 0.001097
grad_step = 000432, loss = 0.001105
grad_step = 000433, loss = 0.001085
grad_step = 000434, loss = 0.001048
grad_step = 000435, loss = 0.001029
grad_step = 000436, loss = 0.001036
grad_step = 000437, loss = 0.001055
grad_step = 000438, loss = 0.001071
grad_step = 000439, loss = 0.001066
grad_step = 000440, loss = 0.001053
grad_step = 000441, loss = 0.001033
grad_step = 000442, loss = 0.001021
grad_step = 000443, loss = 0.001021
grad_step = 000444, loss = 0.001030
grad_step = 000445, loss = 0.001040
grad_step = 000446, loss = 0.001048
grad_step = 000447, loss = 0.001050
grad_step = 000448, loss = 0.001049
grad_step = 000449, loss = 0.001045
grad_step = 000450, loss = 0.001048
grad_step = 000451, loss = 0.001051
grad_step = 000452, loss = 0.001074
grad_step = 000453, loss = 0.001094
grad_step = 000454, loss = 0.001146
grad_step = 000455, loss = 0.001165
grad_step = 000456, loss = 0.001203
grad_step = 000457, loss = 0.001162
grad_step = 000458, loss = 0.001116
grad_step = 000459, loss = 0.001042
grad_step = 000460, loss = 0.001010
grad_step = 000461, loss = 0.001028
grad_step = 000462, loss = 0.001065
grad_step = 000463, loss = 0.001093
grad_step = 000464, loss = 0.001071
grad_step = 000465, loss = 0.001036
grad_step = 000466, loss = 0.001000
grad_step = 000467, loss = 0.000987
grad_step = 000468, loss = 0.000996
grad_step = 000469, loss = 0.001017
grad_step = 000470, loss = 0.001043
grad_step = 000471, loss = 0.001053
grad_step = 000472, loss = 0.001061
grad_step = 000473, loss = 0.001043
grad_step = 000474, loss = 0.001027
grad_step = 000475, loss = 0.001002
grad_step = 000476, loss = 0.000987
grad_step = 000477, loss = 0.000981
grad_step = 000478, loss = 0.000986
grad_step = 000479, loss = 0.000994
grad_step = 000480, loss = 0.000999
grad_step = 000481, loss = 0.000999
grad_step = 000482, loss = 0.000991
grad_step = 000483, loss = 0.000981
grad_step = 000484, loss = 0.000970
grad_step = 000485, loss = 0.000963
grad_step = 000486, loss = 0.000961
grad_step = 000487, loss = 0.000962
grad_step = 000488, loss = 0.000965
grad_step = 000489, loss = 0.000969
grad_step = 000490, loss = 0.000976
grad_step = 000491, loss = 0.000981
grad_step = 000492, loss = 0.000990
grad_step = 000493, loss = 0.000997
grad_step = 000494, loss = 0.001011
grad_step = 000495, loss = 0.001017
grad_step = 000496, loss = 0.001028
grad_step = 000497, loss = 0.001019
grad_step = 000498, loss = 0.001009
grad_step = 000499, loss = 0.000983
grad_step = 000500, loss = 0.000961
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000945
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

  date_run                              2020-05-10 14:15:15.722061
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.230446
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 14:15:15.728128
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.134177
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 14:15:15.735231
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.136101
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 14:15:15.740352
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.03886
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
0   2020-05-10 14:14:47.568302  ...    mean_absolute_error
1   2020-05-10 14:14:47.572246  ...     mean_squared_error
2   2020-05-10 14:14:47.575641  ...  median_absolute_error
3   2020-05-10 14:14:47.578995  ...               r2_score
4   2020-05-10 14:14:56.338832  ...    mean_absolute_error
5   2020-05-10 14:14:56.342884  ...     mean_squared_error
6   2020-05-10 14:14:56.346323  ...  median_absolute_error
7   2020-05-10 14:14:56.349508  ...               r2_score
8   2020-05-10 14:15:15.722061  ...    mean_absolute_error
9   2020-05-10 14:15:15.728128  ...     mean_squared_error
10  2020-05-10 14:15:15.735231  ...  median_absolute_error
11  2020-05-10 14:15:15.740352  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3661824/9912422 [00:00<00:00, 36567533.90it/s]9920512it [00:00, 37233847.00it/s]                             
0it [00:00, ?it/s]32768it [00:00, 706009.93it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 488303.76it/s]1654784it [00:00, 12123457.25it/s]                         
0it [00:00, ?it/s]8192it [00:00, 152830.17it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a5e1d0780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a10b84cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a5e1d0e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a10b84cf8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f19fb9150b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a05035550> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a5e1d0780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a05035550> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a5e1d0780> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a05035550> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a5e1d0780> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f48633871d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=89f0863bf5d404bf626726fdcb0a9618a5263a1a1a240b32e6d830244a411840
  Stored in directory: /tmp/pip-ephem-wheel-cache-3jzleygb/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f47faf6b080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 3s
  663552/17464789 [>.............................] - ETA: 1s
 5660672/17464789 [========>.....................] - ETA: 0s
10346496/17464789 [================>.............] - ETA: 0s
15581184/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 14:16:42.421524: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 14:16:42.426076: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-10 14:16:42.426218: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f8496866a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 14:16:42.426232: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7791 - accuracy: 0.4927
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8391 - accuracy: 0.4888
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8169 - accuracy: 0.4902
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7842 - accuracy: 0.4923
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8703 - accuracy: 0.4867
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.8372 - accuracy: 0.4889
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.8029 - accuracy: 0.4911
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7387 - accuracy: 0.4953
11000/25000 [============>.................] - ETA: 3s - loss: 7.7266 - accuracy: 0.4961
12000/25000 [=============>................] - ETA: 3s - loss: 7.7203 - accuracy: 0.4965
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7150 - accuracy: 0.4968
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6929 - accuracy: 0.4983
15000/25000 [=================>............] - ETA: 2s - loss: 7.6891 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6963 - accuracy: 0.4981
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7092 - accuracy: 0.4972
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6803 - accuracy: 0.4991
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6758 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 14:16:56.147673
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 14:16:56.147673  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 14:17:02.443988: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 14:17:02.449470: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-10 14:17:02.450094: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562e47ac55c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 14:17:02.450440: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1da763ad30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6659 - crf_viterbi_accuracy: 0.2533 - val_loss: 1.6303 - val_crf_viterbi_accuracy: 0.2667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1d9c9e2f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9043 - accuracy: 0.4845 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8455 - accuracy: 0.4883
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8506 - accuracy: 0.4880
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7832 - accuracy: 0.4924
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7184 - accuracy: 0.4966
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6559 - accuracy: 0.5007
11000/25000 [============>.................] - ETA: 3s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 3s - loss: 7.7088 - accuracy: 0.4972
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6655 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6605 - accuracy: 0.5004
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6772 - accuracy: 0.4993
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6868 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6843 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6780 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6794 - accuracy: 0.4992
25000/25000 [==============================] - 7s 286us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1d5874b278> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:20:51, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:28:34, 16.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:11:14, 23.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:08:14, 33.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.66M/862M [00:01<4:59:23, 47.8kB/s].vector_cache/glove.6B.zip:   1%|          | 6.35M/862M [00:01<3:28:47, 68.3kB/s].vector_cache/glove.6B.zip:   1%|          | 10.4M/862M [00:01<2:25:34, 97.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:01<1:41:28, 139kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.0M/862M [00:01<1:10:47, 198kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<49:23, 283kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<34:30, 403kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:01<24:07, 574kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:02<16:54, 814kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:02<11:51, 1.15MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.6M/862M [00:02<08:22, 1.63MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:02<05:54, 2.29MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<04:35, 2.94MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:06, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<05:39, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<04:23, 3.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:04<03:15, 4.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<10:14, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<08:31, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<06:14, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<07:32, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<06:38, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:08<04:58, 2.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<06:36, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<05:59, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:10<04:27, 2.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<06:15, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<07:04, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<05:37, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<06:03, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<05:37, 2.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<04:16, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<06:00, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<05:32, 2.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<04:09, 3.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<05:56, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<05:31, 2.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<04:11, 3.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<06:47, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<05:18, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<03:52, 3.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<09:18, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<07:49, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<05:44, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:00, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<06:12, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<04:37, 2.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:14, 2.03MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:58, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:31, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:54, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:27, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:07, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:21, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:01, 3.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:17, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:00, 3.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:44, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:17, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:57, 3.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:42, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:31, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:11, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<03:46, 3.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<11:41, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:28, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:56, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<07:40, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:37, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:55, 2.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:19, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:40, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:13, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:49, 2.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<05:18, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:01, 2.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:38, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<06:24, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:05, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:29, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:03, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<03:50, 3.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:27, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:03, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<03:47, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:13, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<04:51, 2.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:37, 3.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:01, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:24, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<04:05, 2.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:35, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:18, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:59, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<03:36, 3.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<11:13:19, 17.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<7:52:16, 24.5kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<5:30:07, 35.0kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<3:53:03, 49.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<2:44:02, 70.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<1:54:48, 100kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<1:20:17, 143kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<1:39:27, 115kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<1:11:56, 159kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<50:53, 225kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<37:18, 305kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<27:16, 417kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<19:20, 587kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<16:08, 701kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<12:30, 904kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<08:58, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:54, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:25, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:28, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:27, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:39, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<04:11, 2.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:33, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:48, 2.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:35, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<02:39, 4.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<43:34, 253kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<32:51, 336kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<23:33, 468kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<16:33, 662kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:29:39, 122kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:03:52, 172kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<44:53, 244kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<33:51, 322kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<26:03, 418kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<18:43, 581kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<13:15, 819kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<12:54, 839kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:08, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<07:21, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:39, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:18, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:40, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:47, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:09, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:52, 2.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:12, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:45, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:36, 2.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:00, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:36, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:29, 3.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:54, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:31, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:25, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:29, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:23, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:48, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:26, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:19, 3.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:46, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:20, 3.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:22, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:17, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:42, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:18, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:14, 3.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:39, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:19, 1.89MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:40<04:11, 2.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:05, 3.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:53, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:09, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<03:51, 2.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:01, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:23, 2.93MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:42, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:18, 2.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:13, 3.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:35, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:13, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:11, 3.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:29, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:01, 2.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:23, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:04, 2.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:04, 3.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:21, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:01, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:01, 3.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:21, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:00, 2.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:00, 3.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:21, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:00, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:02, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<02:59, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:00, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:59, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:16, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:55, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:58, 3.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:49, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:46, 2.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:46, 3.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:43, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:56, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:39, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:39, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:06, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:01, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:14, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:53, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:54, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:48, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:51, 3.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:03, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:42, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:42, 3.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:05, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:10, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:50, 2.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:42, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:04, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:56, 2.19MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:53, 2.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:05, 1.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:26, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:19, 2.57MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:18, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:45, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:46, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:44, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<1:02:00, 136kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<44:14, 190kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<31:05, 270kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<23:37, 354kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<17:22, 481kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<12:21, 674kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<10:33, 785kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<08:14, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:56, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:05, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:06, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:46, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:32, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:45, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:44, 2.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:14, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:29, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:18, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:11, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:35, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:37, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:35, 3.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<19:39, 404kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<14:34, 545kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<10:22, 763kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<09:04, 868kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:09, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<05:11, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:28, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:37, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:23, 2.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:12, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:44, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:48, 2.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:46, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:25, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:35, 2.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:36, 2.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:09, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:24, 3.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<01:46, 4.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<29:43, 254kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<21:33, 349kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<15:14, 493kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<12:23, 603kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<10:12, 732kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<07:27, 1.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<05:18, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:51, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:34, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:02, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:32, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:44, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:42, 1.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:39, 2.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<27:57, 260kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<20:18, 357kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<14:21, 503kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<11:40, 616kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<08:54, 806kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<06:24, 1.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:08, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:01, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:39, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:14, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:41, 1.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:45, 2.55MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:34, 1.96MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:12, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:25, 2.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:19, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:02, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:17, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:13, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:57, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:14, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:09, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:10, 3.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:07, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:44, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:04, 3.22MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<01:32, 4.33MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<26:11, 254kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<19:00, 349kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<13:25, 493kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<10:53, 604kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<08:17, 792kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<05:55, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:40, 1.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:19, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:00, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:52, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:52, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:47, 1.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:30, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:56, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:24, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:30, 2.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:15, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:11, 2.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:00, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:44, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:04, 3.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:53, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:39, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<01:59, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:50, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:29, 2.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:00, 3.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<01:27, 4.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<23:47, 253kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<17:15, 349kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<12:10, 492kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<09:53, 603kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<07:31, 791kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:38<05:22, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:08, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:06, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:02, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<02:11, 2.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<23:30, 248kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<17:02, 341kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<12:01, 481kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<09:43, 591kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<07:17, 789kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<05:12, 1.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<03:42, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<24:03, 236kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<17:24, 326kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<12:16, 461kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<09:52, 569kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<07:28, 750kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<05:21, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:02, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:40, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:30, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:33, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:22, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:56, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:11, 2.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:46, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:03, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:22, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<01:43, 3.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:55, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:17, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:26, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:55, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:35, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:55, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:34, 2.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:20, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:45, 2.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:26, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:13, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:39, 3.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:22, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:10, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:38, 3.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:19, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:37, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:17, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:34, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:14, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:04, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:34, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:12, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:32, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<01:58, 2.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:26, 3.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:16, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:46, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:03, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:31, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:14, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:39, 2.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:14, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:31, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:59, 2.29MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:06, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:57, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:27, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:03, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:53, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:25, 3.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:52, 2.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:24, 3.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:00, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:50, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:23, 3.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:58, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:48, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:22, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:46, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:20, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:54, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:45, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:19, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:43, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:18, 3.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:50, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:41, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:17, 3.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:48, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:40, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:16, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:47, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:38, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:14, 3.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:44, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:36, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:12, 3.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:43, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:57, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:33, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:39, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:32, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:09, 3.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:38, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:26, 2.45MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:10, 3.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<00:50, 4.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<13:46, 253kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<10:01, 348kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<07:03, 490kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:41, 602kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:19, 790kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<03:04, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:55, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:23, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:44, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:59, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:43, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:16, 2.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<00:55, 3.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<38:56, 82.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<27:31, 117kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<19:10, 166kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<14:01, 225kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<10:07, 310kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<07:06, 438kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:38, 546kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:15, 722kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<03:01, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:48, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:16, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:39, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:50, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:32, 1.91MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:08, 2.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<00:49, 3.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<12:08, 237kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<08:46, 327kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<06:09, 461kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<04:55, 570kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<03:40, 761kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:36, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:50, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<11:37, 236kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<08:41, 315kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<06:11, 439kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<04:18, 623kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<04:26, 601kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:22, 787kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<02:24, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:16, 1.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:07, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:36, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:30, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:19, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:58, 2.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:15, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:07, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:50, 2.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:08, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:02, 2.28MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<00:46, 3.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:04, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:59, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:44, 3.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:57, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:43, 3.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:00, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:09, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:55, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:53, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:39, 3.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:56, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:04, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:50, 2.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:36, 3.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:17, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:06, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:49, 2.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:00, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:05, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:53, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:48, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:36, 2.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:49, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:56, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:43, 2.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:31, 3.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:02, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:54, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:39, 2.55MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:50, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:55, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:42, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:30, 3.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:21, 665kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:48, 866kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:17, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:13, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:10, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:53, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:49, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:43, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:31, 2.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:40, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:35, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:24, 3.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:23, 931kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:06, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:47, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:49, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:49, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:37, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:26, 2.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:45, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:28, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:30, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:22, 2.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:29, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:33, 1.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:17, 3.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:12, 788kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:56, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:39, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:39, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:38, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:29, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:27, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:23, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:17, 2.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:22, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:19, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:14, 2.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:17, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:11, 3.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<04:02, 151kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<02:51, 211kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<01:56, 299kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<01:23, 389kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:01, 525kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:41, 736kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:33, 842kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:26, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:18, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:17, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:10, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 2.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 3.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:01, 3.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 3.07MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 833/400000 [00:00<00:47, 8329.99it/s]  0%|          | 1713/400000 [00:00<00:47, 8465.04it/s]  1%|          | 2585/400000 [00:00<00:46, 8538.92it/s]  1%|          | 3402/400000 [00:00<00:47, 8422.08it/s]  1%|          | 4256/400000 [00:00<00:46, 8455.21it/s]  1%|▏         | 5099/400000 [00:00<00:46, 8447.12it/s]  1%|▏         | 5961/400000 [00:00<00:46, 8497.21it/s]  2%|▏         | 6832/400000 [00:00<00:45, 8557.35it/s]  2%|▏         | 7689/400000 [00:00<00:45, 8559.75it/s]  2%|▏         | 8537/400000 [00:01<00:45, 8533.92it/s]  2%|▏         | 9423/400000 [00:01<00:45, 8628.30it/s]  3%|▎         | 10270/400000 [00:01<00:45, 8513.58it/s]  3%|▎         | 11111/400000 [00:01<00:47, 8168.99it/s]  3%|▎         | 11964/400000 [00:01<00:46, 8273.71it/s]  3%|▎         | 12790/400000 [00:01<00:46, 8268.08it/s]  3%|▎         | 13644/400000 [00:01<00:46, 8346.49it/s]  4%|▎         | 14523/400000 [00:01<00:45, 8474.49it/s]  4%|▍         | 15397/400000 [00:01<00:44, 8552.05it/s]  4%|▍         | 16253/400000 [00:01<00:45, 8505.58it/s]  4%|▍         | 17104/400000 [00:02<00:45, 8425.75it/s]  4%|▍         | 17967/400000 [00:02<00:45, 8485.22it/s]  5%|▍         | 18816/400000 [00:02<00:45, 8463.29it/s]  5%|▍         | 19663/400000 [00:02<00:45, 8422.20it/s]  5%|▌         | 20513/400000 [00:02<00:44, 8442.62it/s]  5%|▌         | 21361/400000 [00:02<00:44, 8451.84it/s]  6%|▌         | 22207/400000 [00:02<00:44, 8412.93it/s]  6%|▌         | 23076/400000 [00:02<00:44, 8493.95it/s]  6%|▌         | 23934/400000 [00:02<00:44, 8518.68it/s]  6%|▌         | 24810/400000 [00:02<00:43, 8589.36it/s]  6%|▋         | 25670/400000 [00:03<00:43, 8569.39it/s]  7%|▋         | 26528/400000 [00:03<00:43, 8528.97it/s]  7%|▋         | 27388/400000 [00:03<00:43, 8549.49it/s]  7%|▋         | 28257/400000 [00:03<00:43, 8589.10it/s]  7%|▋         | 29117/400000 [00:03<00:43, 8563.68it/s]  7%|▋         | 29982/400000 [00:03<00:43, 8588.64it/s]  8%|▊         | 30854/400000 [00:03<00:42, 8626.24it/s]  8%|▊         | 31717/400000 [00:03<00:42, 8605.66it/s]  8%|▊         | 32578/400000 [00:03<00:42, 8567.36it/s]  8%|▊         | 33438/400000 [00:03<00:42, 8575.65it/s]  9%|▊         | 34301/400000 [00:04<00:42, 8589.64it/s]  9%|▉         | 35161/400000 [00:04<00:42, 8573.92it/s]  9%|▉         | 36019/400000 [00:04<00:42, 8562.34it/s]  9%|▉         | 36876/400000 [00:04<00:42, 8510.51it/s]  9%|▉         | 37728/400000 [00:04<00:44, 8115.71it/s] 10%|▉         | 38584/400000 [00:04<00:43, 8242.25it/s] 10%|▉         | 39479/400000 [00:04<00:42, 8440.87it/s] 10%|█         | 40333/400000 [00:04<00:42, 8467.86it/s] 10%|█         | 41208/400000 [00:04<00:41, 8547.97it/s] 11%|█         | 42065/400000 [00:04<00:42, 8509.69it/s] 11%|█         | 42918/400000 [00:05<00:42, 8457.09it/s] 11%|█         | 43766/400000 [00:05<00:42, 8462.23it/s] 11%|█         | 44613/400000 [00:05<00:42, 8425.90it/s] 11%|█▏        | 45457/400000 [00:05<00:42, 8286.04it/s] 12%|█▏        | 46298/400000 [00:05<00:42, 8322.54it/s] 12%|█▏        | 47131/400000 [00:05<00:42, 8271.05it/s] 12%|█▏        | 47994/400000 [00:05<00:42, 8373.08it/s] 12%|█▏        | 48842/400000 [00:05<00:41, 8403.95it/s] 12%|█▏        | 49693/400000 [00:05<00:41, 8434.14it/s] 13%|█▎        | 50557/400000 [00:05<00:41, 8493.64it/s] 13%|█▎        | 51414/400000 [00:06<00:40, 8514.36it/s] 13%|█▎        | 52285/400000 [00:06<00:40, 8570.34it/s] 13%|█▎        | 53166/400000 [00:06<00:40, 8638.68it/s] 14%|█▎        | 54045/400000 [00:06<00:39, 8682.17it/s] 14%|█▎        | 54914/400000 [00:06<00:39, 8679.97it/s] 14%|█▍        | 55783/400000 [00:06<00:40, 8595.97it/s] 14%|█▍        | 56643/400000 [00:06<00:40, 8576.36it/s] 14%|█▍        | 57501/400000 [00:06<00:39, 8573.63it/s] 15%|█▍        | 58391/400000 [00:06<00:39, 8668.59it/s] 15%|█▍        | 59259/400000 [00:06<00:39, 8655.28it/s] 15%|█▌        | 60125/400000 [00:07<00:39, 8561.27it/s] 15%|█▌        | 60993/400000 [00:07<00:39, 8593.50it/s] 15%|█▌        | 61853/400000 [00:07<00:39, 8514.46it/s] 16%|█▌        | 62714/400000 [00:07<00:39, 8542.84it/s] 16%|█▌        | 63599/400000 [00:07<00:38, 8630.55it/s] 16%|█▌        | 64463/400000 [00:07<00:39, 8572.66it/s] 16%|█▋        | 65324/400000 [00:07<00:38, 8581.56it/s] 17%|█▋        | 66196/400000 [00:07<00:38, 8622.09it/s] 17%|█▋        | 67073/400000 [00:07<00:38, 8665.92it/s] 17%|█▋        | 67940/400000 [00:07<00:38, 8662.03it/s] 17%|█▋        | 68812/400000 [00:08<00:38, 8677.25it/s] 17%|█▋        | 69681/400000 [00:08<00:38, 8680.06it/s] 18%|█▊        | 70573/400000 [00:08<00:37, 8747.89it/s] 18%|█▊        | 71456/400000 [00:08<00:37, 8772.05it/s] 18%|█▊        | 72334/400000 [00:08<00:37, 8737.01it/s] 18%|█▊        | 73208/400000 [00:08<00:37, 8719.81it/s] 19%|█▊        | 74083/400000 [00:08<00:37, 8726.41it/s] 19%|█▊        | 74968/400000 [00:08<00:37, 8762.00it/s] 19%|█▉        | 75845/400000 [00:08<00:37, 8742.23it/s] 19%|█▉        | 76734/400000 [00:08<00:36, 8784.97it/s] 19%|█▉        | 77613/400000 [00:09<00:37, 8533.16it/s] 20%|█▉        | 78468/400000 [00:09<00:38, 8441.97it/s] 20%|█▉        | 79369/400000 [00:09<00:37, 8602.70it/s] 20%|██        | 80236/400000 [00:09<00:37, 8620.19it/s] 20%|██        | 81100/400000 [00:09<00:37, 8547.52it/s] 20%|██        | 81956/400000 [00:09<00:37, 8531.10it/s] 21%|██        | 82810/400000 [00:09<00:37, 8509.47it/s] 21%|██        | 83678/400000 [00:09<00:36, 8557.22it/s] 21%|██        | 84550/400000 [00:09<00:36, 8602.46it/s] 21%|██▏       | 85411/400000 [00:10<00:36, 8512.09it/s] 22%|██▏       | 86263/400000 [00:10<00:37, 8465.37it/s] 22%|██▏       | 87140/400000 [00:10<00:36, 8553.43it/s] 22%|██▏       | 87996/400000 [00:10<00:36, 8463.51it/s] 22%|██▏       | 88843/400000 [00:10<00:37, 8404.42it/s] 22%|██▏       | 89684/400000 [00:10<00:37, 8374.26it/s] 23%|██▎       | 90522/400000 [00:10<00:37, 8218.91it/s] 23%|██▎       | 91377/400000 [00:10<00:37, 8313.86it/s] 23%|██▎       | 92226/400000 [00:10<00:36, 8365.34it/s] 23%|██▎       | 93087/400000 [00:10<00:36, 8435.66it/s] 23%|██▎       | 93932/400000 [00:11<00:36, 8434.09it/s] 24%|██▎       | 94800/400000 [00:11<00:35, 8506.17it/s] 24%|██▍       | 95652/400000 [00:11<00:35, 8459.91it/s] 24%|██▍       | 96499/400000 [00:11<00:36, 8325.22it/s] 24%|██▍       | 97379/400000 [00:11<00:35, 8461.81it/s] 25%|██▍       | 98251/400000 [00:11<00:35, 8537.22it/s] 25%|██▍       | 99114/400000 [00:11<00:35, 8562.54it/s] 25%|██▍       | 99971/400000 [00:11<00:35, 8560.17it/s] 25%|██▌       | 100872/400000 [00:11<00:34, 8688.26it/s] 25%|██▌       | 101760/400000 [00:11<00:34, 8742.31it/s] 26%|██▌       | 102635/400000 [00:12<00:34, 8720.84it/s] 26%|██▌       | 103508/400000 [00:12<00:34, 8604.98it/s] 26%|██▌       | 104370/400000 [00:12<00:34, 8596.77it/s] 26%|██▋       | 105231/400000 [00:12<00:34, 8522.95it/s] 27%|██▋       | 106084/400000 [00:12<00:34, 8479.97it/s] 27%|██▋       | 106933/400000 [00:12<00:34, 8433.57it/s] 27%|██▋       | 107777/400000 [00:12<00:34, 8432.15it/s] 27%|██▋       | 108633/400000 [00:12<00:34, 8469.92it/s] 27%|██▋       | 109505/400000 [00:12<00:34, 8541.86it/s] 28%|██▊       | 110370/400000 [00:12<00:33, 8573.32it/s] 28%|██▊       | 111250/400000 [00:13<00:33, 8638.14it/s] 28%|██▊       | 112115/400000 [00:13<00:33, 8632.59it/s] 28%|██▊       | 112979/400000 [00:13<00:33, 8559.12it/s] 28%|██▊       | 113845/400000 [00:13<00:33, 8587.29it/s] 29%|██▊       | 114717/400000 [00:13<00:33, 8626.36it/s] 29%|██▉       | 115580/400000 [00:13<00:33, 8559.45it/s] 29%|██▉       | 116437/400000 [00:13<00:33, 8526.08it/s] 29%|██▉       | 117290/400000 [00:13<00:33, 8437.92it/s] 30%|██▉       | 118147/400000 [00:13<00:33, 8475.15it/s] 30%|██▉       | 119012/400000 [00:13<00:32, 8525.32it/s] 30%|██▉       | 119875/400000 [00:14<00:32, 8555.99it/s] 30%|███       | 120731/400000 [00:14<00:32, 8547.76it/s] 30%|███       | 121604/400000 [00:14<00:32, 8599.26it/s] 31%|███       | 122465/400000 [00:14<00:32, 8586.35it/s] 31%|███       | 123324/400000 [00:14<00:32, 8541.09it/s] 31%|███       | 124179/400000 [00:14<00:32, 8506.92it/s] 31%|███▏      | 125030/400000 [00:14<00:32, 8500.06it/s] 31%|███▏      | 125881/400000 [00:14<00:33, 8279.16it/s] 32%|███▏      | 126730/400000 [00:14<00:32, 8339.86it/s] 32%|███▏      | 127587/400000 [00:14<00:32, 8405.33it/s] 32%|███▏      | 128431/400000 [00:15<00:32, 8415.26it/s] 32%|███▏      | 129274/400000 [00:15<00:33, 8170.75it/s] 33%|███▎      | 130094/400000 [00:15<00:33, 8113.96it/s] 33%|███▎      | 130907/400000 [00:15<00:33, 8068.21it/s] 33%|███▎      | 131760/400000 [00:15<00:32, 8200.34it/s] 33%|███▎      | 132645/400000 [00:15<00:31, 8384.25it/s] 33%|███▎      | 133502/400000 [00:15<00:31, 8438.55it/s] 34%|███▎      | 134348/400000 [00:15<00:31, 8433.86it/s] 34%|███▍      | 135193/400000 [00:15<00:31, 8380.88it/s] 34%|███▍      | 136047/400000 [00:15<00:31, 8425.68it/s] 34%|███▍      | 136918/400000 [00:16<00:30, 8507.42it/s] 34%|███▍      | 137770/400000 [00:16<00:30, 8498.29it/s] 35%|███▍      | 138621/400000 [00:16<00:31, 8375.26it/s] 35%|███▍      | 139476/400000 [00:16<00:30, 8425.50it/s] 35%|███▌      | 140327/400000 [00:16<00:30, 8449.72it/s] 35%|███▌      | 141173/400000 [00:16<00:30, 8424.94it/s] 36%|███▌      | 142050/400000 [00:16<00:30, 8522.73it/s] 36%|███▌      | 142903/400000 [00:16<00:30, 8514.93it/s] 36%|███▌      | 143779/400000 [00:16<00:29, 8586.51it/s] 36%|███▌      | 144646/400000 [00:16<00:29, 8609.03it/s] 36%|███▋      | 145508/400000 [00:17<00:29, 8587.01it/s] 37%|███▋      | 146367/400000 [00:17<00:29, 8532.39it/s] 37%|███▋      | 147221/400000 [00:17<00:29, 8449.04it/s] 37%|███▋      | 148094/400000 [00:17<00:29, 8528.85it/s] 37%|███▋      | 148948/400000 [00:17<00:29, 8499.93it/s] 37%|███▋      | 149799/400000 [00:17<00:29, 8468.78it/s] 38%|███▊      | 150647/400000 [00:17<00:29, 8442.43it/s] 38%|███▊      | 151497/400000 [00:17<00:29, 8457.18it/s] 38%|███▊      | 152364/400000 [00:17<00:29, 8519.22it/s] 38%|███▊      | 153218/400000 [00:18<00:28, 8523.80it/s] 39%|███▊      | 154071/400000 [00:18<00:28, 8506.92it/s] 39%|███▊      | 154922/400000 [00:18<00:28, 8464.43it/s] 39%|███▉      | 155770/400000 [00:18<00:28, 8468.17it/s] 39%|███▉      | 156643/400000 [00:18<00:28, 8544.96it/s] 39%|███▉      | 157501/400000 [00:18<00:28, 8555.05it/s] 40%|███▉      | 158362/400000 [00:18<00:28, 8571.19it/s] 40%|███▉      | 159220/400000 [00:18<00:28, 8461.65it/s] 40%|████      | 160067/400000 [00:18<00:28, 8390.13it/s] 40%|████      | 160931/400000 [00:18<00:28, 8460.65it/s] 40%|████      | 161783/400000 [00:19<00:28, 8478.10it/s] 41%|████      | 162636/400000 [00:19<00:27, 8491.19it/s] 41%|████      | 163506/400000 [00:19<00:27, 8551.18it/s] 41%|████      | 164369/400000 [00:19<00:27, 8573.76it/s] 41%|████▏     | 165227/400000 [00:19<00:27, 8546.06it/s] 42%|████▏     | 166082/400000 [00:19<00:27, 8454.40it/s] 42%|████▏     | 166954/400000 [00:19<00:27, 8530.43it/s] 42%|████▏     | 167813/400000 [00:19<00:27, 8546.53it/s] 42%|████▏     | 168668/400000 [00:19<00:27, 8519.82it/s] 42%|████▏     | 169538/400000 [00:19<00:26, 8571.99it/s] 43%|████▎     | 170396/400000 [00:20<00:26, 8573.01it/s] 43%|████▎     | 171255/400000 [00:20<00:26, 8575.47it/s] 43%|████▎     | 172113/400000 [00:20<00:26, 8525.84it/s] 43%|████▎     | 172966/400000 [00:20<00:26, 8411.21it/s] 43%|████▎     | 173816/400000 [00:20<00:26, 8437.23it/s] 44%|████▎     | 174661/400000 [00:20<00:26, 8431.84it/s] 44%|████▍     | 175520/400000 [00:20<00:26, 8476.90it/s] 44%|████▍     | 176385/400000 [00:20<00:26, 8526.08it/s] 44%|████▍     | 177238/400000 [00:20<00:26, 8406.91it/s] 45%|████▍     | 178099/400000 [00:20<00:26, 8466.22it/s] 45%|████▍     | 178976/400000 [00:21<00:25, 8552.92it/s] 45%|████▍     | 179839/400000 [00:21<00:25, 8573.87it/s] 45%|████▌     | 180700/400000 [00:21<00:25, 8583.17it/s] 45%|████▌     | 181559/400000 [00:21<00:25, 8521.06it/s] 46%|████▌     | 182412/400000 [00:21<00:25, 8456.55it/s] 46%|████▌     | 183258/400000 [00:21<00:25, 8391.32it/s] 46%|████▌     | 184146/400000 [00:21<00:25, 8532.06it/s] 46%|████▋     | 185001/400000 [00:21<00:25, 8517.06it/s] 46%|████▋     | 185854/400000 [00:21<00:25, 8500.37it/s] 47%|████▋     | 186705/400000 [00:21<00:25, 8502.52it/s] 47%|████▋     | 187570/400000 [00:22<00:24, 8543.51it/s] 47%|████▋     | 188437/400000 [00:22<00:24, 8578.04it/s] 47%|████▋     | 189297/400000 [00:22<00:24, 8584.34it/s] 48%|████▊     | 190177/400000 [00:22<00:24, 8646.67it/s] 48%|████▊     | 191042/400000 [00:22<00:24, 8541.63it/s] 48%|████▊     | 191897/400000 [00:22<00:24, 8503.35it/s] 48%|████▊     | 192749/400000 [00:22<00:24, 8505.61it/s] 48%|████▊     | 193615/400000 [00:22<00:24, 8549.54it/s] 49%|████▊     | 194471/400000 [00:22<00:24, 8524.62it/s] 49%|████▉     | 195332/400000 [00:22<00:23, 8549.66it/s] 49%|████▉     | 196197/400000 [00:23<00:23, 8578.95it/s] 49%|████▉     | 197056/400000 [00:23<00:23, 8555.32it/s] 49%|████▉     | 197912/400000 [00:23<00:24, 8283.68it/s] 50%|████▉     | 198758/400000 [00:23<00:24, 8333.89it/s] 50%|████▉     | 199598/400000 [00:23<00:23, 8353.19it/s] 50%|█████     | 200444/400000 [00:23<00:23, 8382.35it/s] 50%|█████     | 201309/400000 [00:23<00:23, 8459.50it/s] 51%|█████     | 202181/400000 [00:23<00:23, 8535.51it/s] 51%|█████     | 203040/400000 [00:23<00:23, 8549.07it/s] 51%|█████     | 203896/400000 [00:23<00:22, 8549.35it/s] 51%|█████     | 204752/400000 [00:24<00:22, 8550.39it/s] 51%|█████▏    | 205608/400000 [00:24<00:22, 8547.27it/s] 52%|█████▏    | 206466/400000 [00:24<00:22, 8554.62it/s] 52%|█████▏    | 207322/400000 [00:24<00:22, 8453.92it/s] 52%|█████▏    | 208168/400000 [00:24<00:22, 8406.61it/s] 52%|█████▏    | 209029/400000 [00:24<00:22, 8464.51it/s] 52%|█████▏    | 209908/400000 [00:24<00:22, 8557.43it/s] 53%|█████▎    | 210765/400000 [00:24<00:22, 8524.65it/s] 53%|█████▎    | 211618/400000 [00:24<00:22, 8467.82it/s] 53%|█████▎    | 212466/400000 [00:24<00:22, 8404.62it/s] 53%|█████▎    | 213307/400000 [00:25<00:22, 8386.79it/s] 54%|█████▎    | 214146/400000 [00:25<00:22, 8343.71it/s] 54%|█████▍    | 215002/400000 [00:25<00:22, 8406.60it/s] 54%|█████▍    | 215855/400000 [00:25<00:21, 8437.87it/s] 54%|█████▍    | 216708/400000 [00:25<00:21, 8462.74it/s] 54%|█████▍    | 217575/400000 [00:25<00:21, 8521.12it/s] 55%|█████▍    | 218435/400000 [00:25<00:21, 8543.47it/s] 55%|█████▍    | 219290/400000 [00:25<00:21, 8484.45it/s] 55%|█████▌    | 220140/400000 [00:25<00:21, 8487.08it/s] 55%|█████▌    | 221000/400000 [00:25<00:21, 8519.63it/s] 55%|█████▌    | 221871/400000 [00:26<00:20, 8573.14it/s] 56%|█████▌    | 222729/400000 [00:26<00:20, 8574.24it/s] 56%|█████▌    | 223587/400000 [00:26<00:20, 8566.02it/s] 56%|█████▌    | 224445/400000 [00:26<00:20, 8569.85it/s] 56%|█████▋    | 225303/400000 [00:26<00:20, 8540.23it/s] 57%|█████▋    | 226158/400000 [00:26<00:20, 8425.07it/s] 57%|█████▋    | 227001/400000 [00:26<00:20, 8341.59it/s] 57%|█████▋    | 227836/400000 [00:26<00:20, 8225.95it/s] 57%|█████▋    | 228694/400000 [00:26<00:20, 8327.88it/s] 57%|█████▋    | 229545/400000 [00:26<00:20, 8378.91it/s] 58%|█████▊    | 230391/400000 [00:27<00:20, 8402.33it/s] 58%|█████▊    | 231232/400000 [00:27<00:20, 8347.38it/s] 58%|█████▊    | 232073/400000 [00:27<00:20, 8363.38it/s] 58%|█████▊    | 232910/400000 [00:27<00:20, 8324.16it/s] 58%|█████▊    | 233743/400000 [00:27<00:20, 8277.15it/s] 59%|█████▊    | 234616/400000 [00:27<00:19, 8406.60it/s] 59%|█████▉    | 235488/400000 [00:27<00:19, 8497.10it/s] 59%|█████▉    | 236362/400000 [00:27<00:19, 8565.78it/s] 59%|█████▉    | 237227/400000 [00:27<00:18, 8588.56it/s] 60%|█████▉    | 238095/400000 [00:28<00:18, 8615.33it/s] 60%|█████▉    | 238968/400000 [00:28<00:18, 8647.94it/s] 60%|█████▉    | 239839/400000 [00:28<00:18, 8664.24it/s] 60%|██████    | 240706/400000 [00:28<00:18, 8567.68it/s] 60%|██████    | 241585/400000 [00:28<00:18, 8630.89it/s] 61%|██████    | 242449/400000 [00:28<00:18, 8621.85it/s] 61%|██████    | 243322/400000 [00:28<00:18, 8653.73it/s] 61%|██████    | 244206/400000 [00:28<00:17, 8708.46it/s] 61%|██████▏   | 245080/400000 [00:28<00:17, 8716.64it/s] 61%|██████▏   | 245952/400000 [00:28<00:17, 8695.56it/s] 62%|██████▏   | 246822/400000 [00:29<00:17, 8695.12it/s] 62%|██████▏   | 247694/400000 [00:29<00:17, 8702.31it/s] 62%|██████▏   | 248565/400000 [00:29<00:17, 8696.21it/s] 62%|██████▏   | 249438/400000 [00:29<00:17, 8706.25it/s] 63%|██████▎   | 250314/400000 [00:29<00:17, 8719.83it/s] 63%|██████▎   | 251187/400000 [00:29<00:17, 8716.94it/s] 63%|██████▎   | 252059/400000 [00:29<00:17, 8701.20it/s] 63%|██████▎   | 252931/400000 [00:29<00:16, 8704.67it/s] 63%|██████▎   | 253802/400000 [00:29<00:16, 8600.06it/s] 64%|██████▎   | 254675/400000 [00:29<00:16, 8638.01it/s] 64%|██████▍   | 255543/400000 [00:30<00:16, 8647.72it/s] 64%|██████▍   | 256421/400000 [00:30<00:16, 8685.79it/s] 64%|██████▍   | 257302/400000 [00:30<00:16, 8720.50it/s] 65%|██████▍   | 258177/400000 [00:30<00:16, 8728.03it/s] 65%|██████▍   | 259050/400000 [00:30<00:16, 8709.20it/s] 65%|██████▍   | 259922/400000 [00:30<00:16, 8688.20it/s] 65%|██████▌   | 260791/400000 [00:30<00:16, 8663.73it/s] 65%|██████▌   | 261660/400000 [00:30<00:15, 8669.25it/s] 66%|██████▌   | 262530/400000 [00:30<00:15, 8678.28it/s] 66%|██████▌   | 263398/400000 [00:30<00:15, 8672.44it/s] 66%|██████▌   | 264266/400000 [00:31<00:15, 8574.75it/s] 66%|██████▋   | 265131/400000 [00:31<00:15, 8595.98it/s] 67%|██████▋   | 266002/400000 [00:31<00:15, 8629.27it/s] 67%|██████▋   | 266882/400000 [00:31<00:15, 8677.58it/s] 67%|██████▋   | 267750/400000 [00:31<00:15, 8674.61it/s] 67%|██████▋   | 268618/400000 [00:31<00:15, 8637.13it/s] 67%|██████▋   | 269482/400000 [00:31<00:15, 8561.59it/s] 68%|██████▊   | 270368/400000 [00:31<00:14, 8647.99it/s] 68%|██████▊   | 271251/400000 [00:31<00:14, 8699.22it/s] 68%|██████▊   | 272122/400000 [00:31<00:14, 8671.07it/s] 68%|██████▊   | 273000/400000 [00:32<00:14, 8702.14it/s] 68%|██████▊   | 273871/400000 [00:32<00:14, 8697.21it/s] 69%|██████▊   | 274750/400000 [00:32<00:14, 8724.05it/s] 69%|██████▉   | 275634/400000 [00:32<00:14, 8757.20it/s] 69%|██████▉   | 276510/400000 [00:32<00:14, 8742.18it/s] 69%|██████▉   | 277385/400000 [00:32<00:14, 8716.97it/s] 70%|██████▉   | 278257/400000 [00:32<00:14, 8545.56it/s] 70%|██████▉   | 279126/400000 [00:32<00:14, 8587.31it/s] 70%|██████▉   | 279986/400000 [00:32<00:14, 8564.39it/s] 70%|███████   | 280853/400000 [00:32<00:13, 8593.91it/s] 70%|███████   | 281713/400000 [00:33<00:13, 8534.68it/s] 71%|███████   | 282567/400000 [00:33<00:13, 8525.60it/s] 71%|███████   | 283420/400000 [00:33<00:13, 8517.31it/s] 71%|███████   | 284272/400000 [00:33<00:13, 8458.16it/s] 71%|███████▏  | 285148/400000 [00:33<00:13, 8545.61it/s] 72%|███████▏  | 286014/400000 [00:33<00:13, 8578.13it/s] 72%|███████▏  | 286905/400000 [00:33<00:13, 8673.18it/s] 72%|███████▏  | 287773/400000 [00:33<00:12, 8640.77it/s] 72%|███████▏  | 288648/400000 [00:33<00:12, 8672.47it/s] 72%|███████▏  | 289530/400000 [00:33<00:12, 8716.20it/s] 73%|███████▎  | 290409/400000 [00:34<00:12, 8735.84it/s] 73%|███████▎  | 291283/400000 [00:34<00:12, 8697.86it/s] 73%|███████▎  | 292153/400000 [00:34<00:12, 8622.24it/s] 73%|███████▎  | 293018/400000 [00:34<00:12, 8630.33it/s] 73%|███████▎  | 293900/400000 [00:34<00:12, 8684.51it/s] 74%|███████▎  | 294769/400000 [00:34<00:12, 8685.77it/s] 74%|███████▍  | 295638/400000 [00:34<00:12, 8522.51it/s] 74%|███████▍  | 296501/400000 [00:34<00:12, 8552.89it/s] 74%|███████▍  | 297365/400000 [00:34<00:11, 8576.12it/s] 75%|███████▍  | 298237/400000 [00:34<00:11, 8617.33it/s] 75%|███████▍  | 299102/400000 [00:35<00:11, 8625.11it/s] 75%|███████▍  | 299965/400000 [00:35<00:11, 8624.02it/s] 75%|███████▌  | 300836/400000 [00:35<00:11, 8648.36it/s] 75%|███████▌  | 301720/400000 [00:35<00:11, 8703.91it/s] 76%|███████▌  | 302591/400000 [00:35<00:11, 8704.69it/s] 76%|███████▌  | 303462/400000 [00:35<00:11, 8692.00it/s] 76%|███████▌  | 304332/400000 [00:35<00:11, 8639.47it/s] 76%|███████▋  | 305211/400000 [00:35<00:10, 8683.85it/s] 77%|███████▋  | 306080/400000 [00:35<00:11, 8454.99it/s] 77%|███████▋  | 306950/400000 [00:35<00:10, 8525.29it/s] 77%|███████▋  | 307827/400000 [00:36<00:10, 8597.25it/s] 77%|███████▋  | 308694/400000 [00:36<00:10, 8618.69it/s] 77%|███████▋  | 309564/400000 [00:36<00:10, 8640.11it/s] 78%|███████▊  | 310435/400000 [00:36<00:10, 8659.02it/s] 78%|███████▊  | 311302/400000 [00:36<00:10, 8646.98it/s] 78%|███████▊  | 312167/400000 [00:36<00:10, 8564.34it/s] 78%|███████▊  | 313040/400000 [00:36<00:10, 8613.23it/s] 78%|███████▊  | 313902/400000 [00:36<00:10, 8604.68it/s] 79%|███████▊  | 314777/400000 [00:36<00:09, 8645.16it/s] 79%|███████▉  | 315645/400000 [00:36<00:09, 8655.08it/s] 79%|███████▉  | 316534/400000 [00:37<00:09, 8722.61it/s] 79%|███████▉  | 317407/400000 [00:37<00:09, 8667.18it/s] 80%|███████▉  | 318280/400000 [00:37<00:09, 8684.09it/s] 80%|███████▉  | 319165/400000 [00:37<00:09, 8732.29it/s] 80%|████████  | 320055/400000 [00:37<00:09, 8780.42it/s] 80%|████████  | 320938/400000 [00:37<00:08, 8794.51it/s] 80%|████████  | 321818/400000 [00:37<00:08, 8719.50it/s] 81%|████████  | 322691/400000 [00:37<00:08, 8720.82it/s] 81%|████████  | 323564/400000 [00:37<00:08, 8659.97it/s] 81%|████████  | 324431/400000 [00:37<00:08, 8655.04it/s] 81%|████████▏ | 325297/400000 [00:38<00:08, 8650.78it/s] 82%|████████▏ | 326163/400000 [00:38<00:08, 8596.58it/s] 82%|████████▏ | 327023/400000 [00:38<00:08, 8588.78it/s] 82%|████████▏ | 327886/400000 [00:38<00:08, 8598.40it/s] 82%|████████▏ | 328746/400000 [00:38<00:08, 8590.30it/s] 82%|████████▏ | 329606/400000 [00:38<00:08, 8439.58it/s] 83%|████████▎ | 330464/400000 [00:38<00:08, 8480.12it/s] 83%|████████▎ | 331313/400000 [00:38<00:08, 8459.88it/s] 83%|████████▎ | 332176/400000 [00:38<00:07, 8510.07it/s] 83%|████████▎ | 333041/400000 [00:38<00:07, 8550.61it/s] 83%|████████▎ | 333906/400000 [00:39<00:07, 8579.09it/s] 84%|████████▎ | 334765/400000 [00:39<00:07, 8434.71it/s] 84%|████████▍ | 335610/400000 [00:39<00:07, 8316.48it/s] 84%|████████▍ | 336465/400000 [00:39<00:07, 8384.61it/s] 84%|████████▍ | 337306/400000 [00:39<00:07, 8390.41it/s] 85%|████████▍ | 338146/400000 [00:39<00:07, 8352.06it/s] 85%|████████▍ | 338988/400000 [00:39<00:07, 8371.06it/s] 85%|████████▍ | 339826/400000 [00:39<00:07, 8303.94it/s] 85%|████████▌ | 340676/400000 [00:39<00:07, 8360.67it/s] 85%|████████▌ | 341513/400000 [00:40<00:07, 8336.98it/s] 86%|████████▌ | 342357/400000 [00:40<00:06, 8366.67it/s] 86%|████████▌ | 343226/400000 [00:40<00:06, 8459.62it/s] 86%|████████▌ | 344073/400000 [00:40<00:06, 8397.46it/s] 86%|████████▌ | 344931/400000 [00:40<00:06, 8450.09it/s] 86%|████████▋ | 345793/400000 [00:40<00:06, 8498.04it/s] 87%|████████▋ | 346658/400000 [00:40<00:06, 8541.39it/s] 87%|████████▋ | 347523/400000 [00:40<00:06, 8571.31it/s] 87%|████████▋ | 348381/400000 [00:40<00:06, 8524.87it/s] 87%|████████▋ | 349255/400000 [00:40<00:05, 8588.15it/s] 88%|████████▊ | 350128/400000 [00:41<00:05, 8628.60it/s] 88%|████████▊ | 350992/400000 [00:41<00:05, 8579.51it/s] 88%|████████▊ | 351851/400000 [00:41<00:05, 8548.07it/s] 88%|████████▊ | 352706/400000 [00:41<00:05, 8508.76it/s] 88%|████████▊ | 353558/400000 [00:41<00:05, 8476.52it/s] 89%|████████▊ | 354406/400000 [00:41<00:05, 8426.47it/s] 89%|████████▉ | 355249/400000 [00:41<00:05, 8093.53it/s] 89%|████████▉ | 356131/400000 [00:41<00:05, 8296.92it/s] 89%|████████▉ | 356965/400000 [00:41<00:05, 8297.75it/s] 89%|████████▉ | 357852/400000 [00:41<00:04, 8459.37it/s] 90%|████████▉ | 358707/400000 [00:42<00:04, 8482.72it/s] 90%|████████▉ | 359557/400000 [00:42<00:04, 8384.11it/s] 90%|█████████ | 360397/400000 [00:42<00:04, 8388.74it/s] 90%|█████████ | 361237/400000 [00:42<00:04, 8326.30it/s] 91%|█████████ | 362094/400000 [00:42<00:04, 8397.80it/s] 91%|█████████ | 362959/400000 [00:42<00:04, 8469.29it/s] 91%|█████████ | 363814/400000 [00:42<00:04, 8492.69it/s] 91%|█████████ | 364680/400000 [00:42<00:04, 8540.83it/s] 91%|█████████▏| 365536/400000 [00:42<00:04, 8546.29it/s] 92%|█████████▏| 366391/400000 [00:42<00:03, 8527.86it/s] 92%|█████████▏| 367259/400000 [00:43<00:03, 8571.70it/s] 92%|█████████▏| 368130/400000 [00:43<00:03, 8611.65it/s] 92%|█████████▏| 368992/400000 [00:43<00:03, 8591.95it/s] 92%|█████████▏| 369852/400000 [00:43<00:03, 8476.82it/s] 93%|█████████▎| 370711/400000 [00:43<00:03, 8508.97it/s] 93%|█████████▎| 371563/400000 [00:43<00:03, 8491.41it/s] 93%|█████████▎| 372414/400000 [00:43<00:03, 8495.97it/s] 93%|█████████▎| 373273/400000 [00:43<00:03, 8523.31it/s] 94%|█████████▎| 374126/400000 [00:43<00:03, 8416.84it/s] 94%|█████████▎| 374983/400000 [00:43<00:02, 8460.17it/s] 94%|█████████▍| 375830/400000 [00:44<00:02, 8444.65it/s] 94%|█████████▍| 376698/400000 [00:44<00:02, 8511.94it/s] 94%|█████████▍| 377581/400000 [00:44<00:02, 8604.20it/s] 95%|█████████▍| 378442/400000 [00:44<00:02, 8586.50it/s] 95%|█████████▍| 379301/400000 [00:44<00:02, 8545.96it/s] 95%|█████████▌| 380156/400000 [00:44<00:02, 8471.95it/s] 95%|█████████▌| 381004/400000 [00:44<00:02, 8447.45it/s] 95%|█████████▌| 381883/400000 [00:44<00:02, 8545.13it/s] 96%|█████████▌| 382739/400000 [00:44<00:02, 8549.31it/s] 96%|█████████▌| 383619/400000 [00:44<00:01, 8621.81it/s] 96%|█████████▌| 384489/400000 [00:45<00:01, 8643.37it/s] 96%|█████████▋| 385371/400000 [00:45<00:01, 8694.11it/s] 97%|█████████▋| 386241/400000 [00:45<00:01, 8644.88it/s] 97%|█████████▋| 387106/400000 [00:45<00:01, 8513.02it/s] 97%|█████████▋| 387958/400000 [00:45<00:01, 8375.50it/s] 97%|█████████▋| 388828/400000 [00:45<00:01, 8467.94it/s] 97%|█████████▋| 389709/400000 [00:45<00:01, 8565.70it/s] 98%|█████████▊| 390599/400000 [00:45<00:01, 8661.81it/s] 98%|█████████▊| 391467/400000 [00:45<00:00, 8641.39it/s] 98%|█████████▊| 392334/400000 [00:45<00:00, 8649.57it/s] 98%|█████████▊| 393203/400000 [00:46<00:00, 8661.61it/s] 99%|█████████▊| 394070/400000 [00:46<00:00, 8615.65it/s] 99%|█████████▊| 394932/400000 [00:46<00:00, 8600.18it/s] 99%|█████████▉| 395793/400000 [00:46<00:00, 8596.74it/s] 99%|█████████▉| 396666/400000 [00:46<00:00, 8635.58it/s] 99%|█████████▉| 397530/400000 [00:46<00:00, 8437.59it/s]100%|█████████▉| 398380/400000 [00:46<00:00, 8453.51it/s]100%|█████████▉| 399271/400000 [00:46<00:00, 8584.28it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8534.03it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1d6174a518> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010890848391045828 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.010791621877995622 	 Accuracy: 71

  model saves at 71% accuracy 

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

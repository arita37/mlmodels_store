
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fc77ef41f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 15:11:55.634679
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 15:11:55.638274
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 15:11:55.641113
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 15:11:55.643889
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fc78ad06438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355790.8750
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 258962.1094
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 136562.1094
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 57950.3906
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 26637.1016
Epoch 6/10

1/1 [==============================] - 0s 91ms/step - loss: 14196.3145
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 8625.0068
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 5826.9062
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 4269.0645
Epoch 10/10

1/1 [==============================] - 0s 92ms/step - loss: 3350.5991

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.87506723e-01  2.75979972e+00 -8.27046990e-01 -1.03535676e+00
  -1.69456410e+00  4.29536700e-02 -1.51413786e+00 -3.10200155e-01
  -1.47082210e+00 -1.35956705e+00 -5.57021618e-01 -5.35016060e-01
  -1.33160269e+00 -6.23832941e-01  4.79892999e-01 -2.02597213e+00
  -1.51739240e-01 -2.15248227e-01  1.42922354e+00  3.42444718e-01
   2.15824723e-01  1.26531273e-01  1.79361200e+00  9.82985258e-01
  -1.34862733e+00  1.25796628e+00  1.54533732e+00 -8.81417692e-01
   2.05682501e-01 -1.80765986e-01  9.83068645e-01 -4.51966941e-01
   1.34254491e+00 -8.29312563e-01 -2.33393222e-01 -1.46608472e+00
   1.63680911e+00  4.97816652e-01  2.98327655e-01  1.85510945e+00
   5.42543471e-01 -4.09762800e-01 -2.71756101e+00  6.52778149e-02
   1.60176063e+00 -4.16913629e-02  1.09144592e+00  8.18801522e-01
  -1.08527803e+00 -6.46899402e-01 -9.46545005e-01  1.23219562e+00
   1.74749553e+00  1.57556474e-01 -4.38390970e-02  6.30633473e-01
  -8.71803880e-01  2.37903863e-01 -7.91695356e-01  3.08721375e+00
   1.96164501e+00  1.93076146e+00 -2.82207775e+00 -6.56507134e-01
  -7.84124494e-01  1.55825126e+00  9.96791303e-01  2.44160265e-01
   1.87718034e+00  1.60897100e+00  1.11873555e+00 -1.61296773e+00
   1.62541002e-01 -5.25458932e-01 -4.77391362e-01  1.52279472e+00
  -1.11124933e+00  9.92259443e-01 -5.78473091e-01  1.70283675e+00
   2.65211034e+00 -2.11187482e+00 -6.74005628e-01 -1.13962317e+00
  -1.77226782e+00 -1.18601680e-01 -1.87872314e+00 -1.73754048e+00
   1.46673238e+00  5.07207990e-01 -1.22065592e+00  1.62185955e+00
  -7.59007514e-01 -9.52735305e-01  6.30455315e-01  3.36863607e-01
  -4.33852255e-01  1.03312719e+00 -6.28056943e-01  9.21547413e-04
   2.00334644e+00  3.47528577e-01  1.00481904e+00 -1.70946121e-03
   8.13702345e-01 -1.15415239e+00 -4.15933132e-01 -3.72677088e-01
   1.56394839e-01  3.07657003e-01  4.03649628e-01 -2.29012752e+00
  -6.52669489e-01  1.80152607e+00 -1.30633807e+00 -1.55478299e+00
  -2.50771976e+00  4.19957995e-01  2.43045092e-01  4.45127428e-01
  -8.32466364e-01  1.30824718e+01  1.20384216e+01  1.27408152e+01
   1.32402878e+01  1.01852970e+01  1.36307707e+01  1.24125566e+01
   1.24568825e+01  1.18972492e+01  1.24127712e+01  1.12349920e+01
   1.33089294e+01  1.44255667e+01  1.20183020e+01  1.32394753e+01
   1.27667713e+01  1.34885187e+01  1.31123018e+01  1.29447145e+01
   1.21728678e+01  1.34844093e+01  1.27340078e+01  1.15731716e+01
   1.19074621e+01  1.36789608e+01  1.55016832e+01  1.37503567e+01
   1.17746878e+01  1.52073326e+01  1.04008923e+01  1.27183962e+01
   1.17706137e+01  1.07874508e+01  1.18811798e+01  1.10218153e+01
   1.25611830e+01  1.39062529e+01  1.52102413e+01  1.35986423e+01
   1.19232635e+01  1.08333483e+01  1.29374628e+01  1.08672419e+01
   1.43350267e+01  1.23138981e+01  1.32009506e+01  1.12145643e+01
   1.26073933e+01  1.33107624e+01  1.36675644e+01  1.35136862e+01
   1.39817858e+01  1.18341894e+01  1.06505861e+01  1.15670805e+01
   1.27242098e+01  1.18315125e+01  1.20511160e+01  1.26538963e+01
   7.85587907e-01  4.22542214e-01  2.99605131e+00  2.86228657e+00
   2.52467394e+00  2.38754749e+00  1.82823074e+00  1.48621774e+00
   1.62765384e+00  2.28924823e+00  6.61056876e-01  1.06958771e+00
   2.11592102e+00  8.00538361e-01  2.42316246e+00  1.99742818e+00
   3.99981022e-01  7.63353705e-01  1.62213469e+00  1.78300631e+00
   1.48546338e-01  2.42175245e+00  1.18979216e-01  2.60781765e+00
   8.46554101e-01  4.52688098e-01  8.68017435e-01  8.98418427e-01
   3.17993343e-01  1.79973316e+00  8.92099679e-01  1.29815674e+00
   2.61516905e+00  4.15781915e-01  1.32075143e+00  7.91959405e-01
   3.00248563e-01  2.80295682e+00  2.22114468e+00  7.57489204e-01
   2.77632570e+00  2.89228821e+00  1.50428247e+00  1.11947393e+00
   8.27026963e-01  2.93935418e-01  9.88971293e-01  2.39744377e+00
   1.46268654e+00  2.20387888e+00  1.64280283e+00  5.60466349e-01
   4.04364920e+00  1.00052428e+00  2.18265390e+00  8.06333542e-01
   1.93948388e-01  4.77940798e-01  6.29846454e-01  1.09274566e-01
   8.99169445e-01  1.40966487e+00  1.43820202e+00  1.75670242e+00
   1.90363884e+00  2.12578058e-01  5.32646775e-02  1.40716577e+00
   1.87549567e+00  1.89496255e+00  9.57943976e-01  7.05683827e-01
   3.71001303e-01  4.09507942e+00  1.60417342e+00  1.39830351e-01
   2.38092518e+00  2.09919786e+00  1.25021577e-01  1.49130285e-01
   2.62449169e+00  1.89798951e-01  1.45383239e+00  6.04982793e-01
   1.47447610e+00  4.89815116e-01  6.03931844e-01  7.55962849e-01
   6.08977079e-01  1.54256225e-01  2.17787504e+00  4.04192924e-01
   4.29617047e-01  1.16389871e-01  3.40096807e+00  1.77302885e+00
   4.94959354e-02  3.35438013e+00  4.25003231e-01  3.31755638e+00
   1.86711109e+00  1.57216644e+00  2.22504663e+00  2.07478428e+00
   2.50798941e+00  2.02256536e+00  2.40172291e+00  1.06627226e-01
   2.98963785e-01  9.97350216e-02  2.33751249e+00  2.66878748e+00
   2.34025061e-01  7.85173953e-01  8.85764182e-01  1.67428768e+00
   2.02764845e+00  8.83762002e-01  4.28268313e-01  2.10483491e-01
   1.13437665e+00  1.17936039e+01  1.30113993e+01  1.29183111e+01
   1.33151531e+01  1.27583838e+01  1.35887041e+01  1.21473560e+01
   1.07467880e+01  1.19786100e+01  1.23294506e+01  1.23787594e+01
   1.19344378e+01  1.16218519e+01  1.23521004e+01  1.10057640e+01
   1.22608919e+01  1.23470993e+01  1.22020178e+01  1.08114386e+01
   1.13067770e+01  1.26081448e+01  1.22148876e+01  1.22077513e+01
   1.24583597e+01  1.29332323e+01  1.04614964e+01  1.36549654e+01
   1.03371620e+01  1.41174345e+01  1.24297409e+01  1.35691586e+01
   1.25157118e+01  1.22689028e+01  1.09400101e+01  1.25948200e+01
   1.30301971e+01  1.25115252e+01  1.31684704e+01  1.15494967e+01
   1.25136156e+01  1.28069258e+01  9.81648636e+00  1.32188482e+01
   1.28913555e+01  1.14827862e+01  1.15552521e+01  1.26600628e+01
   1.29079142e+01  1.28175697e+01  1.30086794e+01  1.32884426e+01
   1.38899441e+01  1.29571419e+01  1.17263536e+01  1.44157715e+01
   1.21666355e+01  1.29342451e+01  9.37836456e+00  1.35655107e+01
  -1.11575813e+01 -2.03530483e+01  1.37421408e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 15:12:04.778362
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.7341
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 15:12:04.785872
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8087.29
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 15:12:04.792352
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.1689
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 15:12:04.798674
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -723.274
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140494445949280
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140493504648024
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140493504648528
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140493504649032
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140493504649536
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140493504650040

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fc786418080> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.601468
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.570298
grad_step = 000002, loss = 0.549892
grad_step = 000003, loss = 0.528219
grad_step = 000004, loss = 0.501443
grad_step = 000005, loss = 0.469992
grad_step = 000006, loss = 0.437746
grad_step = 000007, loss = 0.413736
grad_step = 000008, loss = 0.409682
grad_step = 000009, loss = 0.395651
grad_step = 000010, loss = 0.372463
grad_step = 000011, loss = 0.354883
grad_step = 000012, loss = 0.346349
grad_step = 000013, loss = 0.339876
grad_step = 000014, loss = 0.330745
grad_step = 000015, loss = 0.319250
grad_step = 000016, loss = 0.306839
grad_step = 000017, loss = 0.294711
grad_step = 000018, loss = 0.283888
grad_step = 000019, loss = 0.274530
grad_step = 000020, loss = 0.265333
grad_step = 000021, loss = 0.255900
grad_step = 000022, loss = 0.246544
grad_step = 000023, loss = 0.237039
grad_step = 000024, loss = 0.227908
grad_step = 000025, loss = 0.219539
grad_step = 000026, loss = 0.211560
grad_step = 000027, loss = 0.203448
grad_step = 000028, loss = 0.195181
grad_step = 000029, loss = 0.187067
grad_step = 000030, loss = 0.179453
grad_step = 000031, loss = 0.172590
grad_step = 000032, loss = 0.166174
grad_step = 000033, loss = 0.159433
grad_step = 000034, loss = 0.152428
grad_step = 000035, loss = 0.145882
grad_step = 000036, loss = 0.139989
grad_step = 000037, loss = 0.134315
grad_step = 000038, loss = 0.128482
grad_step = 000039, loss = 0.122596
grad_step = 000040, loss = 0.117095
grad_step = 000041, loss = 0.112180
grad_step = 000042, loss = 0.107454
grad_step = 000043, loss = 0.102517
grad_step = 000044, loss = 0.097632
grad_step = 000045, loss = 0.093202
grad_step = 000046, loss = 0.089109
grad_step = 000047, loss = 0.085004
grad_step = 000048, loss = 0.080857
grad_step = 000049, loss = 0.076950
grad_step = 000050, loss = 0.073395
grad_step = 000051, loss = 0.069930
grad_step = 000052, loss = 0.066418
grad_step = 000053, loss = 0.063078
grad_step = 000054, loss = 0.060018
grad_step = 000055, loss = 0.057075
grad_step = 000056, loss = 0.054149
grad_step = 000057, loss = 0.051346
grad_step = 000058, loss = 0.048749
grad_step = 000059, loss = 0.046271
grad_step = 000060, loss = 0.043837
grad_step = 000061, loss = 0.041511
grad_step = 000062, loss = 0.039328
grad_step = 000063, loss = 0.037241
grad_step = 000064, loss = 0.035230
grad_step = 000065, loss = 0.033317
grad_step = 000066, loss = 0.031504
grad_step = 000067, loss = 0.029775
grad_step = 000068, loss = 0.028136
grad_step = 000069, loss = 0.026578
grad_step = 000070, loss = 0.025087
grad_step = 000071, loss = 0.023680
grad_step = 000072, loss = 0.022355
grad_step = 000073, loss = 0.021085
grad_step = 000074, loss = 0.019878
grad_step = 000075, loss = 0.018756
grad_step = 000076, loss = 0.017691
grad_step = 000077, loss = 0.016669
grad_step = 000078, loss = 0.015715
grad_step = 000079, loss = 0.014825
grad_step = 000080, loss = 0.013972
grad_step = 000081, loss = 0.013168
grad_step = 000082, loss = 0.012424
grad_step = 000083, loss = 0.011718
grad_step = 000084, loss = 0.011048
grad_step = 000085, loss = 0.010425
grad_step = 000086, loss = 0.009841
grad_step = 000087, loss = 0.009288
grad_step = 000088, loss = 0.008773
grad_step = 000089, loss = 0.008292
grad_step = 000090, loss = 0.007840
grad_step = 000091, loss = 0.007417
grad_step = 000092, loss = 0.007024
grad_step = 000093, loss = 0.006655
grad_step = 000094, loss = 0.006312
grad_step = 000095, loss = 0.005991
grad_step = 000096, loss = 0.005691
grad_step = 000097, loss = 0.005413
grad_step = 000098, loss = 0.005153
grad_step = 000099, loss = 0.004911
grad_step = 000100, loss = 0.004687
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004477
grad_step = 000102, loss = 0.004282
grad_step = 000103, loss = 0.004102
grad_step = 000104, loss = 0.003934
grad_step = 000105, loss = 0.003777
grad_step = 000106, loss = 0.003633
grad_step = 000107, loss = 0.003499
grad_step = 000108, loss = 0.003375
grad_step = 000109, loss = 0.003260
grad_step = 000110, loss = 0.003153
grad_step = 000111, loss = 0.003055
grad_step = 000112, loss = 0.002964
grad_step = 000113, loss = 0.002880
grad_step = 000114, loss = 0.002802
grad_step = 000115, loss = 0.002730
grad_step = 000116, loss = 0.002664
grad_step = 000117, loss = 0.002603
grad_step = 000118, loss = 0.002547
grad_step = 000119, loss = 0.002495
grad_step = 000120, loss = 0.002448
grad_step = 000121, loss = 0.002404
grad_step = 000122, loss = 0.002364
grad_step = 000123, loss = 0.002328
grad_step = 000124, loss = 0.002294
grad_step = 000125, loss = 0.002263
grad_step = 000126, loss = 0.002235
grad_step = 000127, loss = 0.002209
grad_step = 000128, loss = 0.002185
grad_step = 000129, loss = 0.002163
grad_step = 000130, loss = 0.002143
grad_step = 000131, loss = 0.002125
grad_step = 000132, loss = 0.002108
grad_step = 000133, loss = 0.002093
grad_step = 000134, loss = 0.002079
grad_step = 000135, loss = 0.002066
grad_step = 000136, loss = 0.002054
grad_step = 000137, loss = 0.002043
grad_step = 000138, loss = 0.002033
grad_step = 000139, loss = 0.002023
grad_step = 000140, loss = 0.002015
grad_step = 000141, loss = 0.002007
grad_step = 000142, loss = 0.001999
grad_step = 000143, loss = 0.001992
grad_step = 000144, loss = 0.001986
grad_step = 000145, loss = 0.001979
grad_step = 000146, loss = 0.001974
grad_step = 000147, loss = 0.001968
grad_step = 000148, loss = 0.001963
grad_step = 000149, loss = 0.001958
grad_step = 000150, loss = 0.001953
grad_step = 000151, loss = 0.001949
grad_step = 000152, loss = 0.001944
grad_step = 000153, loss = 0.001940
grad_step = 000154, loss = 0.001936
grad_step = 000155, loss = 0.001932
grad_step = 000156, loss = 0.001928
grad_step = 000157, loss = 0.001924
grad_step = 000158, loss = 0.001920
grad_step = 000159, loss = 0.001916
grad_step = 000160, loss = 0.001912
grad_step = 000161, loss = 0.001908
grad_step = 000162, loss = 0.001904
grad_step = 000163, loss = 0.001900
grad_step = 000164, loss = 0.001896
grad_step = 000165, loss = 0.001892
grad_step = 000166, loss = 0.001888
grad_step = 000167, loss = 0.001884
grad_step = 000168, loss = 0.001880
grad_step = 000169, loss = 0.001876
grad_step = 000170, loss = 0.001872
grad_step = 000171, loss = 0.001867
grad_step = 000172, loss = 0.001864
grad_step = 000173, loss = 0.001859
grad_step = 000174, loss = 0.001855
grad_step = 000175, loss = 0.001851
grad_step = 000176, loss = 0.001846
grad_step = 000177, loss = 0.001842
grad_step = 000178, loss = 0.001837
grad_step = 000179, loss = 0.001833
grad_step = 000180, loss = 0.001828
grad_step = 000181, loss = 0.001823
grad_step = 000182, loss = 0.001819
grad_step = 000183, loss = 0.001814
grad_step = 000184, loss = 0.001809
grad_step = 000185, loss = 0.001805
grad_step = 000186, loss = 0.001799
grad_step = 000187, loss = 0.001795
grad_step = 000188, loss = 0.001790
grad_step = 000189, loss = 0.001785
grad_step = 000190, loss = 0.001782
grad_step = 000191, loss = 0.001779
grad_step = 000192, loss = 0.001776
grad_step = 000193, loss = 0.001770
grad_step = 000194, loss = 0.001762
grad_step = 000195, loss = 0.001756
grad_step = 000196, loss = 0.001753
grad_step = 000197, loss = 0.001751
grad_step = 000198, loss = 0.001747
grad_step = 000199, loss = 0.001741
grad_step = 000200, loss = 0.001734
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001728
grad_step = 000202, loss = 0.001723
grad_step = 000203, loss = 0.001720
grad_step = 000204, loss = 0.001717
grad_step = 000205, loss = 0.001716
grad_step = 000206, loss = 0.001715
grad_step = 000207, loss = 0.001717
grad_step = 000208, loss = 0.001714
grad_step = 000209, loss = 0.001707
grad_step = 000210, loss = 0.001694
grad_step = 000211, loss = 0.001684
grad_step = 000212, loss = 0.001678
grad_step = 000213, loss = 0.001677
grad_step = 000214, loss = 0.001678
grad_step = 000215, loss = 0.001681
grad_step = 000216, loss = 0.001688
grad_step = 000217, loss = 0.001689
grad_step = 000218, loss = 0.001684
grad_step = 000219, loss = 0.001666
grad_step = 000220, loss = 0.001649
grad_step = 000221, loss = 0.001641
grad_step = 000222, loss = 0.001643
grad_step = 000223, loss = 0.001647
grad_step = 000224, loss = 0.001646
grad_step = 000225, loss = 0.001637
grad_step = 000226, loss = 0.001624
grad_step = 000227, loss = 0.001615
grad_step = 000228, loss = 0.001613
grad_step = 000229, loss = 0.001613
grad_step = 000230, loss = 0.001614
grad_step = 000231, loss = 0.001612
grad_step = 000232, loss = 0.001611
grad_step = 000233, loss = 0.001608
grad_step = 000234, loss = 0.001605
grad_step = 000235, loss = 0.001596
grad_step = 000236, loss = 0.001585
grad_step = 000237, loss = 0.001580
grad_step = 000238, loss = 0.001579
grad_step = 000239, loss = 0.001578
grad_step = 000240, loss = 0.001573
grad_step = 000241, loss = 0.001566
grad_step = 000242, loss = 0.001560
grad_step = 000243, loss = 0.001557
grad_step = 000244, loss = 0.001556
grad_step = 000245, loss = 0.001556
grad_step = 000246, loss = 0.001556
grad_step = 000247, loss = 0.001553
grad_step = 000248, loss = 0.001552
grad_step = 000249, loss = 0.001552
grad_step = 000250, loss = 0.001563
grad_step = 000251, loss = 0.001598
grad_step = 000252, loss = 0.001640
grad_step = 000253, loss = 0.001672
grad_step = 000254, loss = 0.001597
grad_step = 000255, loss = 0.001531
grad_step = 000256, loss = 0.001534
grad_step = 000257, loss = 0.001579
grad_step = 000258, loss = 0.001595
grad_step = 000259, loss = 0.001542
grad_step = 000260, loss = 0.001516
grad_step = 000261, loss = 0.001539
grad_step = 000262, loss = 0.001560
grad_step = 000263, loss = 0.001545
grad_step = 000264, loss = 0.001510
grad_step = 000265, loss = 0.001503
grad_step = 000266, loss = 0.001523
grad_step = 000267, loss = 0.001532
grad_step = 000268, loss = 0.001520
grad_step = 000269, loss = 0.001498
grad_step = 000270, loss = 0.001492
grad_step = 000271, loss = 0.001502
grad_step = 000272, loss = 0.001510
grad_step = 000273, loss = 0.001505
grad_step = 000274, loss = 0.001491
grad_step = 000275, loss = 0.001485
grad_step = 000276, loss = 0.001492
grad_step = 000277, loss = 0.001502
grad_step = 000278, loss = 0.001506
grad_step = 000279, loss = 0.001506
grad_step = 000280, loss = 0.001509
grad_step = 000281, loss = 0.001529
grad_step = 000282, loss = 0.001557
grad_step = 000283, loss = 0.001594
grad_step = 000284, loss = 0.001589
grad_step = 000285, loss = 0.001561
grad_step = 000286, loss = 0.001500
grad_step = 000287, loss = 0.001466
grad_step = 000288, loss = 0.001474
grad_step = 000289, loss = 0.001502
grad_step = 000290, loss = 0.001521
grad_step = 000291, loss = 0.001502
grad_step = 000292, loss = 0.001471
grad_step = 000293, loss = 0.001456
grad_step = 000294, loss = 0.001465
grad_step = 000295, loss = 0.001482
grad_step = 000296, loss = 0.001485
grad_step = 000297, loss = 0.001474
grad_step = 000298, loss = 0.001457
grad_step = 000299, loss = 0.001448
grad_step = 000300, loss = 0.001450
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001459
grad_step = 000302, loss = 0.001467
grad_step = 000303, loss = 0.001465
grad_step = 000304, loss = 0.001459
grad_step = 000305, loss = 0.001450
grad_step = 000306, loss = 0.001443
grad_step = 000307, loss = 0.001438
grad_step = 000308, loss = 0.001435
grad_step = 000309, loss = 0.001436
grad_step = 000310, loss = 0.001439
grad_step = 000311, loss = 0.001442
grad_step = 000312, loss = 0.001444
grad_step = 000313, loss = 0.001444
grad_step = 000314, loss = 0.001444
grad_step = 000315, loss = 0.001444
grad_step = 000316, loss = 0.001443
grad_step = 000317, loss = 0.001441
grad_step = 000318, loss = 0.001439
grad_step = 000319, loss = 0.001438
grad_step = 000320, loss = 0.001436
grad_step = 000321, loss = 0.001435
grad_step = 000322, loss = 0.001435
grad_step = 000323, loss = 0.001436
grad_step = 000324, loss = 0.001437
grad_step = 000325, loss = 0.001440
grad_step = 000326, loss = 0.001443
grad_step = 000327, loss = 0.001447
grad_step = 000328, loss = 0.001452
grad_step = 000329, loss = 0.001460
grad_step = 000330, loss = 0.001464
grad_step = 000331, loss = 0.001471
grad_step = 000332, loss = 0.001472
grad_step = 000333, loss = 0.001471
grad_step = 000334, loss = 0.001462
grad_step = 000335, loss = 0.001453
grad_step = 000336, loss = 0.001435
grad_step = 000337, loss = 0.001421
grad_step = 000338, loss = 0.001410
grad_step = 000339, loss = 0.001401
grad_step = 000340, loss = 0.001394
grad_step = 000341, loss = 0.001392
grad_step = 000342, loss = 0.001392
grad_step = 000343, loss = 0.001394
grad_step = 000344, loss = 0.001400
grad_step = 000345, loss = 0.001408
grad_step = 000346, loss = 0.001422
grad_step = 000347, loss = 0.001440
grad_step = 000348, loss = 0.001470
grad_step = 000349, loss = 0.001501
grad_step = 000350, loss = 0.001539
grad_step = 000351, loss = 0.001549
grad_step = 000352, loss = 0.001539
grad_step = 000353, loss = 0.001484
grad_step = 000354, loss = 0.001421
grad_step = 000355, loss = 0.001379
grad_step = 000356, loss = 0.001380
grad_step = 000357, loss = 0.001410
grad_step = 000358, loss = 0.001439
grad_step = 000359, loss = 0.001446
grad_step = 000360, loss = 0.001423
grad_step = 000361, loss = 0.001390
grad_step = 000362, loss = 0.001368
grad_step = 000363, loss = 0.001365
grad_step = 000364, loss = 0.001379
grad_step = 000365, loss = 0.001397
grad_step = 000366, loss = 0.001411
grad_step = 000367, loss = 0.001416
grad_step = 000368, loss = 0.001413
grad_step = 000369, loss = 0.001403
grad_step = 000370, loss = 0.001388
grad_step = 000371, loss = 0.001372
grad_step = 000372, loss = 0.001358
grad_step = 000373, loss = 0.001349
grad_step = 000374, loss = 0.001347
grad_step = 000375, loss = 0.001348
grad_step = 000376, loss = 0.001353
grad_step = 000377, loss = 0.001359
grad_step = 000378, loss = 0.001367
grad_step = 000379, loss = 0.001378
grad_step = 000380, loss = 0.001390
grad_step = 000381, loss = 0.001407
grad_step = 000382, loss = 0.001424
grad_step = 000383, loss = 0.001443
grad_step = 000384, loss = 0.001454
grad_step = 000385, loss = 0.001455
grad_step = 000386, loss = 0.001435
grad_step = 000387, loss = 0.001399
grad_step = 000388, loss = 0.001359
grad_step = 000389, loss = 0.001331
grad_step = 000390, loss = 0.001325
grad_step = 000391, loss = 0.001336
grad_step = 000392, loss = 0.001355
grad_step = 000393, loss = 0.001371
grad_step = 000394, loss = 0.001379
grad_step = 000395, loss = 0.001376
grad_step = 000396, loss = 0.001366
grad_step = 000397, loss = 0.001352
grad_step = 000398, loss = 0.001337
grad_step = 000399, loss = 0.001323
grad_step = 000400, loss = 0.001313
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001306
grad_step = 000402, loss = 0.001302
grad_step = 000403, loss = 0.001301
grad_step = 000404, loss = 0.001302
grad_step = 000405, loss = 0.001304
grad_step = 000406, loss = 0.001310
grad_step = 000407, loss = 0.001319
grad_step = 000408, loss = 0.001336
grad_step = 000409, loss = 0.001366
grad_step = 000410, loss = 0.001414
grad_step = 000411, loss = 0.001490
grad_step = 000412, loss = 0.001583
grad_step = 000413, loss = 0.001662
grad_step = 000414, loss = 0.001655
grad_step = 000415, loss = 0.001538
grad_step = 000416, loss = 0.001365
grad_step = 000417, loss = 0.001286
grad_step = 000418, loss = 0.001337
grad_step = 000419, loss = 0.001422
grad_step = 000420, loss = 0.001427
grad_step = 000421, loss = 0.001344
grad_step = 000422, loss = 0.001277
grad_step = 000423, loss = 0.001293
grad_step = 000424, loss = 0.001348
grad_step = 000425, loss = 0.001371
grad_step = 000426, loss = 0.001338
grad_step = 000427, loss = 0.001292
grad_step = 000428, loss = 0.001267
grad_step = 000429, loss = 0.001270
grad_step = 000430, loss = 0.001291
grad_step = 000431, loss = 0.001314
grad_step = 000432, loss = 0.001323
grad_step = 000433, loss = 0.001310
grad_step = 000434, loss = 0.001283
grad_step = 000435, loss = 0.001261
grad_step = 000436, loss = 0.001252
grad_step = 000437, loss = 0.001250
grad_step = 000438, loss = 0.001256
grad_step = 000439, loss = 0.001267
grad_step = 000440, loss = 0.001275
grad_step = 000441, loss = 0.001274
grad_step = 000442, loss = 0.001268
grad_step = 000443, loss = 0.001261
grad_step = 000444, loss = 0.001250
grad_step = 000445, loss = 0.001240
grad_step = 000446, loss = 0.001235
grad_step = 000447, loss = 0.001233
grad_step = 000448, loss = 0.001233
grad_step = 000449, loss = 0.001234
grad_step = 000450, loss = 0.001236
grad_step = 000451, loss = 0.001235
grad_step = 000452, loss = 0.001236
grad_step = 000453, loss = 0.001241
grad_step = 000454, loss = 0.001249
grad_step = 000455, loss = 0.001258
grad_step = 000456, loss = 0.001273
grad_step = 000457, loss = 0.001297
grad_step = 000458, loss = 0.001333
grad_step = 000459, loss = 0.001376
grad_step = 000460, loss = 0.001429
grad_step = 000461, loss = 0.001479
grad_step = 000462, loss = 0.001479
grad_step = 000463, loss = 0.001424
grad_step = 000464, loss = 0.001338
grad_step = 000465, loss = 0.001244
grad_step = 000466, loss = 0.001207
grad_step = 000467, loss = 0.001235
grad_step = 000468, loss = 0.001281
grad_step = 000469, loss = 0.001301
grad_step = 000470, loss = 0.001278
grad_step = 000471, loss = 0.001232
grad_step = 000472, loss = 0.001203
grad_step = 000473, loss = 0.001205
grad_step = 000474, loss = 0.001230
grad_step = 000475, loss = 0.001252
grad_step = 000476, loss = 0.001252
grad_step = 000477, loss = 0.001231
grad_step = 000478, loss = 0.001207
grad_step = 000479, loss = 0.001191
grad_step = 000480, loss = 0.001188
grad_step = 000481, loss = 0.001196
grad_step = 000482, loss = 0.001208
grad_step = 000483, loss = 0.001218
grad_step = 000484, loss = 0.001222
grad_step = 000485, loss = 0.001219
grad_step = 000486, loss = 0.001212
grad_step = 000487, loss = 0.001201
grad_step = 000488, loss = 0.001190
grad_step = 000489, loss = 0.001181
grad_step = 000490, loss = 0.001175
grad_step = 000491, loss = 0.001171
grad_step = 000492, loss = 0.001172
grad_step = 000493, loss = 0.001173
grad_step = 000494, loss = 0.001176
grad_step = 000495, loss = 0.001179
grad_step = 000496, loss = 0.001184
grad_step = 000497, loss = 0.001188
grad_step = 000498, loss = 0.001193
grad_step = 000499, loss = 0.001199
grad_step = 000500, loss = 0.001207
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001217
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

  date_run                              2020-05-10 15:12:22.163089
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.280601
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 15:12:22.170422
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.204929
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 15:12:22.176332
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.153258
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 15:12:22.181186
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.11397
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
0   2020-05-10 15:11:55.634679  ...    mean_absolute_error
1   2020-05-10 15:11:55.638274  ...     mean_squared_error
2   2020-05-10 15:11:55.641113  ...  median_absolute_error
3   2020-05-10 15:11:55.643889  ...               r2_score
4   2020-05-10 15:12:04.778362  ...    mean_absolute_error
5   2020-05-10 15:12:04.785872  ...     mean_squared_error
6   2020-05-10 15:12:04.792352  ...  median_absolute_error
7   2020-05-10 15:12:04.798674  ...               r2_score
8   2020-05-10 15:12:22.163089  ...    mean_absolute_error
9   2020-05-10 15:12:22.170422  ...     mean_squared_error
10  2020-05-10 15:12:22.176332  ...  median_absolute_error
11  2020-05-10 15:12:22.181186  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:23, 118911.99it/s] 77%|███████▋  | 7610368/9912422 [00:00<00:13, 169760.34it/s]9920512it [00:00, 38309261.94it/s]                           
0it [00:00, ?it/s]32768it [00:00, 518117.49it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162083.03it/s]1654784it [00:00, 11574646.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 206682.58it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a2ab5b780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69c829f9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a2ab10e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69c829fda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a2ab10e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69dd50ecf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a2ab5bf98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69dd50ecf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a2ab5bf98> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69dd50ecf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69c82a00b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb322cc11d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=628c12463cbcaf693a71157e9e66433ab0965f5ff026cea167d710069418b96a
  Stored in directory: /tmp/pip-ephem-wheel-cache-stu77t54/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb2ba8a6048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2686976/17464789 [===>..........................] - ETA: 0s
 9150464/17464789 [==============>...............] - ETA: 0s
15556608/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 15:13:46.896481: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 15:13:46.900747: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095175000 Hz
2020-05-10 15:13:46.900878: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b44e031130 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 15:13:46.900891: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5235 - accuracy: 0.5093
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4750 - accuracy: 0.5125
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5317 - accuracy: 0.5088
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5235 - accuracy: 0.5093
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5352 - accuracy: 0.5086
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5842 - accuracy: 0.5054
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5559 - accuracy: 0.5072
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5486 - accuracy: 0.5077
11000/25000 [============>.................] - ETA: 3s - loss: 7.5509 - accuracy: 0.5075
12000/25000 [=============>................] - ETA: 3s - loss: 7.6027 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6430 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6458 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 2s - loss: 7.6288 - accuracy: 0.5025
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6254 - accuracy: 0.5027
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6441 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6555 - accuracy: 0.5007
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6699 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6766 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6538 - accuracy: 0.5008
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 15:13:59.484683
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 15:13:59.484683  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 15:14:05.173988: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 15:14:05.179148: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095175000 Hz
2020-05-10 15:14:05.179427: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d828c59f70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 15:14:05.179552: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd23944dd30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2544 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.2315 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd22e7f5f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9886 - accuracy: 0.4790
 2000/25000 [=>............................] - ETA: 7s - loss: 7.9196 - accuracy: 0.4835 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8008 - accuracy: 0.4913
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7849 - accuracy: 0.4923
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7797 - accuracy: 0.4926
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7603 - accuracy: 0.4939
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7464 - accuracy: 0.4948
11000/25000 [============>.................] - ETA: 3s - loss: 7.7517 - accuracy: 0.4945
12000/25000 [=============>................] - ETA: 3s - loss: 7.7343 - accuracy: 0.4956
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7170 - accuracy: 0.4967
15000/25000 [=================>............] - ETA: 2s - loss: 7.7361 - accuracy: 0.4955
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7222 - accuracy: 0.4964
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7117 - accuracy: 0.4971
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6989 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 7s 264us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd224110470> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<47:44:09, 5.02kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<33:38:52, 7.12kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:36:17, 10.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:31:27, 14.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:02<11:32:00, 20.7kB/s].vector_cache/glove.6B.zip:   1%|          | 7.71M/862M [00:02<8:02:12, 29.5kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:02<5:35:49, 42.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:02<3:53:36, 60.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:02<2:42:31, 86.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:02<1:53:02, 123kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:02<1:18:59, 175kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<55:00, 250kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:03<38:18, 356kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:03<26:52, 506kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<18:46, 719kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<19:16, 700kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<15:20, 875kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<13:00, 1.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:06<09:35, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<09:01, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<07:57, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:08<05:58, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<06:52, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<07:45, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:10<06:09, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<04:27, 2.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<17:16, 765kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<13:27, 982kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:12<09:41, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<09:51, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<09:34, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<07:16, 1.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<05:14, 2.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<11:29, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:15<09:21, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<06:52, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:17<07:51, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<08:08, 1.60MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<06:14, 2.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<04:32, 2.86MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<09:18, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:19<07:51, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<05:46, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<07:02, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:21<07:32, 1.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:21<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:21<04:15, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<1:35:31, 134kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<1:08:10, 188kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<47:53, 267kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<36:26, 350kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:25<28:05, 453kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<20:17, 627kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<16:12, 782kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<12:39, 1.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<09:09, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:20, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:08, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<07:02, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:56, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<06:07, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<04:33, 2.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:06, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:50, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:19, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<03:53, 3.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<08:09, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:59, 1.77MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<05:09, 2.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:03, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:33, 2.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:52, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:21, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:03, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:41, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:21, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:04, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:30, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:06, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<03:53, 3.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:30, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:04, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<03:48, 3.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<05:31, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<06:18, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:01, 2.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<05:26, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<05:01, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<03:48, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<05:26, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:00, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<03:45, 3.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:23, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:10, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<04:49, 2.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<03:31, 3.32MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:01, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<06:49, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:55<05:03, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:16, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<05:34, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<04:11, 2.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:40, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:07, 2.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<03:52, 2.98MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:25, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:57, 2.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<03:45, 3.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:20, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:04, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<04:43, 2.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<03:27, 3.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:45, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:35, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:03, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:25, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:04, 2.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:29, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:59, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:46, 2.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<05:16, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:49, 2.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<03:39, 3.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:10, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:54, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:37, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:20, 3.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<11:48, 934kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<09:25, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:51, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:21, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<06:15, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:39, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:09, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:52, 2.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:16, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:55, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:39, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:22, 3.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<10:20:19, 17.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<7:14:53, 24.7kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<5:04:04, 35.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<3:32:13, 50.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<3:11:32, 55.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<2:16:13, 78.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<1:35:47, 111kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<1:08:26, 155kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<48:59, 217kB/s]  .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<34:29, 307kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<26:30, 398kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<19:36, 538kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<13:57, 753kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<12:12, 858kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<10:45, 973kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:05, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<05:45, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<28:04, 371kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<20:44, 501kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<14:43, 705kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<12:37, 819kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<09:55, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:12, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<07:20, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<07:19, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:35, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:10, 2.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:21, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:49, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<03:36, 2.81MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<04:48, 2.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:26, 2.28MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:22, 3.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<04:37, 2.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<05:21, 1.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:17, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:06, 3.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<26:14, 381kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<19:23, 515kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<13:48, 721kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<11:52, 836kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<10:24, 952kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:48, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:33, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<26:40, 369kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<19:42, 500kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<13:59, 702kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<11:57, 818kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<10:26, 937kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<07:49, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:34, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<34:44, 280kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<25:21, 383kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<17:58, 539kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<14:41, 656kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<12:20, 782kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<09:08, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:29, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<33:20, 287kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<24:20, 393kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<17:15, 553kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<14:10, 671kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<11:56, 796kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<08:46, 1.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:16, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:55, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<06:32, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:46, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:27, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<05:48, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:30, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:14, 2.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<11:15, 826kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<08:51, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<06:23, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:33, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:33, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:07, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:57, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:24, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<04:11, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:03, 2.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:55, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:06, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<03:48, 2.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:41, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:14, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<03:11, 2.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:15, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:52, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<03:48, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<02:46, 3.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:37, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:34, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:07, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:51, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:16, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:04, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<02:58, 2.95MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<06:17, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:19, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:55, 2.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:41, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:07, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:02, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<02:54, 2.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<21:57, 392kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<16:16, 529kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<11:35, 741kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<09:59, 854kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<08:48, 970kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:36, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:42, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<30:11, 281kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<22:01, 385kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<15:35, 541kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<12:45, 659kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<10:43, 784kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:52, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:36, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<07:34, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:10, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:30, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<05:01, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:33<05:16, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:07, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:58, 2.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<22:35, 363kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<16:39, 491kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<11:48, 691kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<10:05, 806kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<08:42, 933kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<06:28, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<04:36, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<38:59, 207kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<28:06, 287kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<19:49, 405kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<15:39, 510kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<12:39, 631kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<09:15, 861kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<06:32, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<22:54, 346kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<16:50, 470kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:57, 660kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<10:09, 773kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<08:52, 885kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<06:36, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:41, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<21:10, 368kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<15:37, 498kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<11:04, 700kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<09:28, 814kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<08:12, 940kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<06:07, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:21, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<57:40, 133kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<41:09, 186kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<28:54, 263kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<21:50, 347kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<16:53, 448kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<12:08, 623kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<08:33, 879kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<09:35, 782kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<07:29, 1.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:25, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<05:30, 1.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<05:26, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:12, 1.77MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:00, 2.45MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<15:26, 477kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<11:35, 635kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<08:15, 889kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<07:23, 987kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<06:44, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:06, 1.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:38, 1.98MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<24:12, 299kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<17:40, 409kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<12:29, 576kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<10:21, 691kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<08:45, 817kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<06:30, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:37, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<24:32, 289kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<17:55, 396kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<12:41, 557kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<10:25, 674kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<08:47, 799kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<06:30, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<04:36, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<20:41, 336kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<15:11, 457kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<10:46, 642kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<09:05, 758kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<07:05, 971kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:07, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:06, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:02, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<03:48, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<02:48, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:49, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:23, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:31, 2.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:16, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:39, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<02:50, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:04, 3.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<04:27, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:49, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<02:50, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:27, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:47, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<02:59, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:09, 2.99MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<47:47, 136kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<34:06, 190kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<23:57, 269kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<18:07, 354kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<13:19, 480kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<09:26, 674kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<08:04, 785kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<06:59, 906kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:10, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<06:18, 994kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<05:03, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:40, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:59, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:02, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:06, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:14, 2.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<05:05, 1.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:12, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:06, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:31, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:45, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:57, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:07, 2.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<15:51, 378kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<11:43, 511kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<08:19, 716kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<07:09, 827kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<05:38, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:05, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:09, 1.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:09, 1.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:12, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:17, 2.53MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<29:31, 196kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<21:14, 272kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<14:57, 385kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<11:43, 487kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<08:48, 648kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<06:16, 907kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<05:38, 1.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<05:08, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:51, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:44, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<06:08, 909kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<04:52, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:31, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<03:43, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:43, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:50, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:02, 2.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<04:39, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<03:49, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:47, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:09, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:20, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:37, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:53, 2.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<17:16, 307kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<12:38, 419kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:56, 589kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<07:25, 705kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<06:15, 836kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:38, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<03:16, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<27:05, 191kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<19:29, 265kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<13:42, 374kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<10:41, 477kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<08:33, 595kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<06:14, 813kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:23, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<14:45, 341kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<10:51, 463kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<07:41, 650kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:08<06:27, 767kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<05:34, 890kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<04:09, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:56, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<13:49, 354kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<10:10, 480kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<07:11, 675kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<06:05, 792kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<05:16, 913kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:56, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:46, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<24:59, 190kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<17:59, 264kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<12:37, 374kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<09:49, 476kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<07:52, 595kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<05:42, 819kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:02, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<04:35, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<03:41, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:40, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:54, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:57, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:15, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:37, 2.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<04:05, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:19, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:26, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:41, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:20, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:44, 2.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:11, 1.98MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:24, 1.80MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<01:54, 2.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:22, 3.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<31:28, 136kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<22:27, 190kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<15:44, 269kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<11:51, 354kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<09:11, 457kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<06:37, 631kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<04:38, 891kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<32:45, 126kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<23:20, 177kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<16:20, 251kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<12:15, 332kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<09:25, 431kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<06:47, 596kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:44, 842kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<12:58, 308kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<09:29, 420kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<06:42, 591kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<05:31, 709kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:42, 834kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:29, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:27, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<10:56, 352kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<08:03, 478kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<05:42, 670kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<04:49, 785kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<04:08, 913kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<03:05, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<02:10, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<25:25, 146kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<18:09, 204kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<12:42, 290kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<09:39, 378kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<07:31, 485kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<05:25, 669kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<03:47, 944kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<31:50, 112kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<22:34, 158kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<15:47, 225kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<11:43, 300kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<08:56, 392kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<06:25, 544kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<04:28, 771kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<11:06, 310kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<08:07, 423kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:43, 596kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<04:44, 712kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<03:39, 920kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:37, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:35, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:30, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:55, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:21, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<08:45, 370kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<06:27, 501kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:34, 702kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<03:52, 817kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<03:23, 936kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:30, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:46, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<02:25, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<02:01, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:28, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:42, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:31, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:07, 2.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:26, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:36, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<01:16, 2.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:54, 3.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<21:19, 136kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<15:12, 190kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<10:36, 270kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<07:58, 354kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<06:11, 456kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<04:27, 630kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<03:06, 889kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<10:53, 253kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<07:54, 348kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<05:33, 491kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<04:26, 605kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<03:40, 730kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:40, 997kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:53, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:34, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:04, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:29, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:36, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:39, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:16, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:54, 2.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:56, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:36, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:10, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:22, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:09, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:51, 2.75MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:37, 3.74MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<05:00, 467kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<03:58, 588kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:52, 809kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:00, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:29, 915kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:58, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:25, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:28, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:30, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:08, 1.92MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:48, 2.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:44, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:26, 1.48MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:02, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:11, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:16, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:59, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:42, 2.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<06:49, 293kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<04:56, 403kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<03:27, 567kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<02:49, 682kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:10, 885kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:32, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:29, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:26, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:05, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:45, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:36, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:18, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:57, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:02, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:54, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:39, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:49, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:55, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:43, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:30, 3.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:14, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:02, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:51, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:55, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:43, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:30, 2.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<03:42, 390kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<02:44, 526kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:55, 739kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<01:36, 853kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:25, 968kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:02, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:43, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:29, 875kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:10, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:50, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:51, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:51, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:39, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:27, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:49, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:42, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:30, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:35, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:39, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:21, 2.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<03:21, 308kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<02:25, 424kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:40, 598kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<01:08, 844kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<02:54, 332kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:59, 470kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<01:31, 586kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:09, 769kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:48, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:43, 1.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:41, 1.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:30, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:21, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:37, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:30, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:21, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:23, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:25, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:19, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:13, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:33, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:26, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:19, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:20, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:15, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:10, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:18, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:11, 2.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:12, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:08, 2.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:05, 3.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:10, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:08, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:05, 2.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:26, 309kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:18, 422kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:10, 593kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:05, 710kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:04, 832kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.57MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 939/400000 [00:00<00:42, 9382.49it/s]  0%|          | 1858/400000 [00:00<00:42, 9322.41it/s]  1%|          | 2790/400000 [00:00<00:42, 9321.26it/s]  1%|          | 3707/400000 [00:00<00:42, 9273.59it/s]  1%|          | 4704/400000 [00:00<00:41, 9471.12it/s]  1%|▏         | 5616/400000 [00:00<00:42, 9361.70it/s]  2%|▏         | 6522/400000 [00:00<00:42, 9267.33it/s]  2%|▏         | 7373/400000 [00:00<00:43, 8980.41it/s]  2%|▏         | 8367/400000 [00:00<00:42, 9247.43it/s]  2%|▏         | 9362/400000 [00:01<00:41, 9446.79it/s]  3%|▎         | 10331/400000 [00:01<00:40, 9515.74it/s]  3%|▎         | 11267/400000 [00:01<00:41, 9328.06it/s]  3%|▎         | 12214/400000 [00:01<00:41, 9369.96it/s]  3%|▎         | 13169/400000 [00:01<00:41, 9420.64it/s]  4%|▎         | 14131/400000 [00:01<00:40, 9477.19it/s]  4%|▍         | 15085/400000 [00:01<00:40, 9494.18it/s]  4%|▍         | 16047/400000 [00:01<00:40, 9529.58it/s]  4%|▍         | 17009/400000 [00:01<00:40, 9554.99it/s]  4%|▍         | 17973/400000 [00:01<00:39, 9579.89it/s]  5%|▍         | 18940/400000 [00:02<00:39, 9604.82it/s]  5%|▍         | 19910/400000 [00:02<00:39, 9630.63it/s]  5%|▌         | 20902/400000 [00:02<00:39, 9713.12it/s]  5%|▌         | 21874/400000 [00:02<00:39, 9596.25it/s]  6%|▌         | 22834/400000 [00:02<00:39, 9579.71it/s]  6%|▌         | 23793/400000 [00:02<00:39, 9549.57it/s]  6%|▌         | 24762/400000 [00:02<00:39, 9590.06it/s]  6%|▋         | 25722/400000 [00:02<00:39, 9588.58it/s]  7%|▋         | 26681/400000 [00:02<00:39, 9506.42it/s]  7%|▋         | 27632/400000 [00:02<00:40, 9266.61it/s]  7%|▋         | 28572/400000 [00:03<00:39, 9304.37it/s]  7%|▋         | 29504/400000 [00:03<00:40, 9155.00it/s]  8%|▊         | 30426/400000 [00:03<00:40, 9173.45it/s]  8%|▊         | 31373/400000 [00:03<00:39, 9259.68it/s]  8%|▊         | 32304/400000 [00:03<00:39, 9272.82it/s]  8%|▊         | 33273/400000 [00:03<00:39, 9393.08it/s]  9%|▊         | 34245/400000 [00:03<00:38, 9487.52it/s]  9%|▉         | 35205/400000 [00:03<00:38, 9520.51it/s]  9%|▉         | 36173/400000 [00:03<00:38, 9566.90it/s]  9%|▉         | 37139/400000 [00:03<00:37, 9592.30it/s] 10%|▉         | 38111/400000 [00:04<00:37, 9628.44it/s] 10%|▉         | 39085/400000 [00:04<00:37, 9660.61it/s] 10%|█         | 40055/400000 [00:04<00:37, 9670.50it/s] 10%|█         | 41023/400000 [00:04<00:37, 9661.35it/s] 10%|█         | 41990/400000 [00:04<00:37, 9553.60it/s] 11%|█         | 42946/400000 [00:04<00:37, 9457.91it/s] 11%|█         | 43893/400000 [00:04<00:37, 9389.74it/s] 11%|█         | 44833/400000 [00:04<00:38, 9265.31it/s] 11%|█▏        | 45761/400000 [00:04<00:38, 9216.89it/s] 12%|█▏        | 46706/400000 [00:04<00:38, 9285.14it/s] 12%|█▏        | 47728/400000 [00:05<00:36, 9545.16it/s] 12%|█▏        | 48710/400000 [00:05<00:36, 9623.08it/s] 12%|█▏        | 49684/400000 [00:05<00:36, 9655.91it/s] 13%|█▎        | 50651/400000 [00:05<00:36, 9525.37it/s] 13%|█▎        | 51605/400000 [00:05<00:36, 9447.88it/s] 13%|█▎        | 52551/400000 [00:05<00:36, 9400.91it/s] 13%|█▎        | 53512/400000 [00:05<00:36, 9461.81it/s] 14%|█▎        | 54463/400000 [00:05<00:36, 9473.84it/s] 14%|█▍        | 55411/400000 [00:05<00:36, 9429.35it/s] 14%|█▍        | 56355/400000 [00:05<00:36, 9386.82it/s] 14%|█▍        | 57306/400000 [00:06<00:36, 9421.53it/s] 15%|█▍        | 58263/400000 [00:06<00:36, 9464.76it/s] 15%|█▍        | 59233/400000 [00:06<00:35, 9532.99it/s] 15%|█▌        | 60187/400000 [00:06<00:36, 9418.92it/s] 15%|█▌        | 61130/400000 [00:06<00:36, 9301.81it/s] 16%|█▌        | 62105/400000 [00:06<00:35, 9429.23it/s] 16%|█▌        | 63056/400000 [00:06<00:35, 9452.36it/s] 16%|█▌        | 64033/400000 [00:06<00:35, 9543.33it/s] 16%|█▌        | 64999/400000 [00:06<00:34, 9577.63it/s] 16%|█▋        | 65960/400000 [00:06<00:34, 9584.77it/s] 17%|█▋        | 66919/400000 [00:07<00:35, 9512.34it/s] 17%|█▋        | 67877/400000 [00:07<00:34, 9530.48it/s] 17%|█▋        | 68877/400000 [00:07<00:34, 9666.38it/s] 17%|█▋        | 69845/400000 [00:07<00:34, 9665.96it/s] 18%|█▊        | 70813/400000 [00:07<00:34, 9646.09it/s] 18%|█▊        | 71785/400000 [00:07<00:33, 9666.61it/s] 18%|█▊        | 72756/400000 [00:07<00:33, 9678.12it/s] 18%|█▊        | 73729/400000 [00:07<00:33, 9689.92it/s] 19%|█▊        | 74716/400000 [00:07<00:33, 9742.06it/s] 19%|█▉        | 75691/400000 [00:07<00:33, 9678.62it/s] 19%|█▉        | 76660/400000 [00:08<00:34, 9451.61it/s] 19%|█▉        | 77621/400000 [00:08<00:33, 9496.92it/s] 20%|█▉        | 78585/400000 [00:08<00:33, 9536.59it/s] 20%|█▉        | 79566/400000 [00:08<00:33, 9615.60it/s] 20%|██        | 80529/400000 [00:08<00:33, 9602.83it/s] 20%|██        | 81494/400000 [00:08<00:33, 9614.78it/s] 21%|██        | 82457/400000 [00:08<00:33, 9617.23it/s] 21%|██        | 83435/400000 [00:08<00:32, 9664.56it/s] 21%|██        | 84441/400000 [00:08<00:32, 9776.86it/s] 21%|██▏       | 85421/400000 [00:08<00:32, 9783.53it/s] 22%|██▏       | 86400/400000 [00:09<00:32, 9767.92it/s] 22%|██▏       | 87378/400000 [00:09<00:32, 9490.22it/s] 22%|██▏       | 88329/400000 [00:09<00:32, 9474.59it/s] 22%|██▏       | 89283/400000 [00:09<00:32, 9492.14it/s] 23%|██▎       | 90273/400000 [00:09<00:32, 9610.47it/s] 23%|██▎       | 91237/400000 [00:09<00:32, 9616.41it/s] 23%|██▎       | 92200/400000 [00:09<00:32, 9553.07it/s] 23%|██▎       | 93156/400000 [00:09<00:32, 9468.80it/s] 24%|██▎       | 94122/400000 [00:09<00:32, 9524.25it/s] 24%|██▍       | 95116/400000 [00:10<00:31, 9643.23it/s] 24%|██▍       | 96082/400000 [00:10<00:31, 9569.76it/s] 24%|██▍       | 97040/400000 [00:10<00:31, 9519.89it/s] 25%|██▍       | 98004/400000 [00:10<00:31, 9552.18it/s] 25%|██▍       | 99010/400000 [00:10<00:31, 9698.33it/s] 25%|██▌       | 100007/400000 [00:10<00:30, 9776.99it/s] 25%|██▌       | 100991/400000 [00:10<00:30, 9795.77it/s] 25%|██▌       | 101972/400000 [00:10<00:30, 9736.93it/s] 26%|██▌       | 102970/400000 [00:10<00:30, 9807.60it/s] 26%|██▌       | 103955/400000 [00:10<00:30, 9819.76it/s] 26%|██▌       | 104940/400000 [00:11<00:30, 9827.29it/s] 26%|██▋       | 105923/400000 [00:11<00:30, 9674.60it/s] 27%|██▋       | 106899/400000 [00:11<00:30, 9699.59it/s] 27%|██▋       | 107870/400000 [00:11<00:30, 9652.88it/s] 27%|██▋       | 108836/400000 [00:11<00:30, 9635.83it/s] 27%|██▋       | 109844/400000 [00:11<00:29, 9763.19it/s] 28%|██▊       | 110821/400000 [00:11<00:29, 9697.13it/s] 28%|██▊       | 111792/400000 [00:11<00:29, 9673.68it/s] 28%|██▊       | 112771/400000 [00:11<00:29, 9707.66it/s] 28%|██▊       | 113767/400000 [00:11<00:29, 9779.73it/s] 29%|██▊       | 114746/400000 [00:12<00:29, 9764.82it/s] 29%|██▉       | 115723/400000 [00:12<00:29, 9667.31it/s] 29%|██▉       | 116716/400000 [00:12<00:29, 9744.44it/s] 29%|██▉       | 117691/400000 [00:12<00:29, 9696.10it/s] 30%|██▉       | 118675/400000 [00:12<00:28, 9736.87it/s] 30%|██▉       | 119649/400000 [00:12<00:29, 9636.33it/s] 30%|███       | 120615/400000 [00:12<00:28, 9642.91it/s] 30%|███       | 121580/400000 [00:12<00:29, 9599.90it/s] 31%|███       | 122541/400000 [00:12<00:28, 9571.83it/s] 31%|███       | 123499/400000 [00:12<00:28, 9562.73it/s] 31%|███       | 124456/400000 [00:13<00:29, 9426.49it/s] 31%|███▏      | 125403/400000 [00:13<00:29, 9437.56it/s] 32%|███▏      | 126381/400000 [00:13<00:28, 9535.54it/s] 32%|███▏      | 127345/400000 [00:13<00:28, 9565.26it/s] 32%|███▏      | 128305/400000 [00:13<00:28, 9575.24it/s] 32%|███▏      | 129267/400000 [00:13<00:28, 9585.82it/s] 33%|███▎      | 130226/400000 [00:13<00:28, 9545.14it/s] 33%|███▎      | 131181/400000 [00:13<00:28, 9440.66it/s] 33%|███▎      | 132161/400000 [00:13<00:28, 9544.59it/s] 33%|███▎      | 133116/400000 [00:13<00:28, 9488.34it/s] 34%|███▎      | 134077/400000 [00:14<00:27, 9521.64it/s] 34%|███▍      | 135045/400000 [00:14<00:27, 9567.49it/s] 34%|███▍      | 136043/400000 [00:14<00:27, 9684.62it/s] 34%|███▍      | 137013/400000 [00:14<00:27, 9669.94it/s] 34%|███▍      | 137996/400000 [00:14<00:26, 9715.10it/s] 35%|███▍      | 138968/400000 [00:14<00:26, 9709.84it/s] 35%|███▍      | 139940/400000 [00:14<00:27, 9604.70it/s] 35%|███▌      | 140901/400000 [00:14<00:26, 9602.63it/s] 35%|███▌      | 141862/400000 [00:14<00:26, 9596.21it/s] 36%|███▌      | 142859/400000 [00:14<00:26, 9705.26it/s] 36%|███▌      | 143877/400000 [00:15<00:26, 9841.48it/s] 36%|███▌      | 144862/400000 [00:15<00:26, 9591.78it/s] 36%|███▋      | 145824/400000 [00:15<00:26, 9532.04it/s] 37%|███▋      | 146819/400000 [00:15<00:26, 9651.06it/s] 37%|███▋      | 147802/400000 [00:15<00:25, 9702.91it/s] 37%|███▋      | 148780/400000 [00:15<00:25, 9724.62it/s] 37%|███▋      | 149754/400000 [00:15<00:25, 9685.45it/s] 38%|███▊      | 150724/400000 [00:15<00:25, 9662.18it/s] 38%|███▊      | 151691/400000 [00:15<00:25, 9600.37it/s] 38%|███▊      | 152672/400000 [00:15<00:25, 9661.75it/s] 38%|███▊      | 153656/400000 [00:16<00:25, 9714.01it/s] 39%|███▊      | 154634/400000 [00:16<00:25, 9731.73it/s] 39%|███▉      | 155608/400000 [00:16<00:25, 9681.22it/s] 39%|███▉      | 156577/400000 [00:16<00:25, 9539.92it/s] 39%|███▉      | 157539/400000 [00:16<00:25, 9562.19it/s] 40%|███▉      | 158515/400000 [00:16<00:25, 9618.37it/s] 40%|███▉      | 159521/400000 [00:16<00:24, 9743.85it/s] 40%|████      | 160501/400000 [00:16<00:24, 9760.13it/s] 40%|████      | 161478/400000 [00:16<00:24, 9720.03it/s] 41%|████      | 162460/400000 [00:16<00:24, 9749.26it/s] 41%|████      | 163450/400000 [00:17<00:24, 9791.53it/s] 41%|████      | 164430/400000 [00:17<00:24, 9701.42it/s] 41%|████▏     | 165404/400000 [00:17<00:24, 9711.80it/s] 42%|████▏     | 166376/400000 [00:17<00:24, 9661.32it/s] 42%|████▏     | 167366/400000 [00:17<00:23, 9729.17it/s] 42%|████▏     | 168354/400000 [00:17<00:23, 9773.85it/s] 42%|████▏     | 169332/400000 [00:17<00:23, 9741.36it/s] 43%|████▎     | 170307/400000 [00:17<00:23, 9717.90it/s] 43%|████▎     | 171295/400000 [00:17<00:23, 9763.18it/s] 43%|████▎     | 172289/400000 [00:17<00:23, 9813.08it/s] 43%|████▎     | 173271/400000 [00:18<00:23, 9766.56it/s] 44%|████▎     | 174275/400000 [00:18<00:22, 9845.93it/s] 44%|████▍     | 175260/400000 [00:18<00:23, 9692.44it/s] 44%|████▍     | 176253/400000 [00:18<00:22, 9759.87it/s] 44%|████▍     | 177252/400000 [00:18<00:22, 9826.74it/s] 45%|████▍     | 178246/400000 [00:18<00:22, 9858.38it/s] 45%|████▍     | 179239/400000 [00:18<00:22, 9879.67it/s] 45%|████▌     | 180264/400000 [00:18<00:22, 9985.99it/s] 45%|████▌     | 181294/400000 [00:18<00:21, 10076.96it/s] 46%|████▌     | 182305/400000 [00:18<00:21, 10085.85it/s] 46%|████▌     | 183340/400000 [00:19<00:21, 10161.55it/s] 46%|████▌     | 184367/400000 [00:19<00:21, 10192.91it/s] 46%|████▋     | 185387/400000 [00:19<00:21, 10065.47it/s] 47%|████▋     | 186395/400000 [00:19<00:21, 9900.55it/s]  47%|████▋     | 187387/400000 [00:19<00:21, 9894.88it/s] 47%|████▋     | 188378/400000 [00:19<00:21, 9886.86it/s] 47%|████▋     | 189368/400000 [00:19<00:21, 9727.66it/s] 48%|████▊     | 190342/400000 [00:19<00:21, 9689.91it/s] 48%|████▊     | 191312/400000 [00:19<00:21, 9650.47it/s] 48%|████▊     | 192364/400000 [00:20<00:20, 9892.77it/s] 48%|████▊     | 193420/400000 [00:20<00:20, 10082.08it/s] 49%|████▊     | 194431/400000 [00:20<00:20, 10082.75it/s] 49%|████▉     | 195441/400000 [00:20<00:20, 10018.01it/s] 49%|████▉     | 196444/400000 [00:20<00:20, 9962.97it/s]  49%|████▉     | 197442/400000 [00:20<00:20, 9775.71it/s] 50%|████▉     | 198421/400000 [00:20<00:20, 9742.83it/s] 50%|████▉     | 199397/400000 [00:20<00:20, 9690.01it/s] 50%|█████     | 200405/400000 [00:20<00:20, 9801.45it/s] 50%|█████     | 201392/400000 [00:20<00:20, 9820.14it/s] 51%|█████     | 202376/400000 [00:21<00:20, 9824.35it/s] 51%|█████     | 203359/400000 [00:21<00:20, 9818.16it/s] 51%|█████     | 204342/400000 [00:21<00:20, 9766.28it/s] 51%|█████▏    | 205319/400000 [00:21<00:20, 9660.41it/s] 52%|█████▏    | 206286/400000 [00:21<00:20, 9608.77it/s] 52%|█████▏    | 207252/400000 [00:21<00:20, 9623.56it/s] 52%|█████▏    | 208275/400000 [00:21<00:19, 9796.77it/s] 52%|█████▏    | 209287/400000 [00:21<00:19, 9887.62it/s] 53%|█████▎    | 210279/400000 [00:21<00:19, 9896.82it/s] 53%|█████▎    | 211309/400000 [00:21<00:18, 10013.37it/s] 53%|█████▎    | 212321/400000 [00:22<00:18, 10045.06it/s] 53%|█████▎    | 213327/400000 [00:22<00:19, 9811.50it/s]  54%|█████▎    | 214318/400000 [00:22<00:18, 9838.06it/s] 54%|█████▍    | 215303/400000 [00:22<00:19, 9707.63it/s] 54%|█████▍    | 216275/400000 [00:22<00:19, 9596.94it/s] 54%|█████▍    | 217255/400000 [00:22<00:18, 9656.03it/s] 55%|█████▍    | 218226/400000 [00:22<00:18, 9671.72it/s] 55%|█████▍    | 219195/400000 [00:22<00:18, 9674.36it/s] 55%|█████▌    | 220163/400000 [00:22<00:18, 9669.68it/s] 55%|█████▌    | 221135/400000 [00:22<00:18, 9683.25it/s] 56%|█████▌    | 222104/400000 [00:23<00:18, 9684.41it/s] 56%|█████▌    | 223073/400000 [00:23<00:18, 9672.20it/s] 56%|█████▌    | 224041/400000 [00:23<00:18, 9576.85it/s] 56%|█████▋    | 225000/400000 [00:23<00:18, 9577.94it/s] 57%|█████▋    | 226001/400000 [00:23<00:17, 9701.15it/s] 57%|█████▋    | 227020/400000 [00:23<00:17, 9841.13it/s] 57%|█████▋    | 228014/400000 [00:23<00:17, 9868.43it/s] 57%|█████▋    | 229002/400000 [00:23<00:17, 9870.63it/s] 57%|█████▋    | 229990/400000 [00:23<00:17, 9799.05it/s] 58%|█████▊    | 231005/400000 [00:23<00:17, 9899.25it/s] 58%|█████▊    | 232021/400000 [00:24<00:16, 9972.69it/s] 58%|█████▊    | 233034/400000 [00:24<00:16, 10017.41it/s] 59%|█████▊    | 234061/400000 [00:24<00:16, 10091.19it/s] 59%|█████▉    | 235071/400000 [00:24<00:16, 9867.85it/s]  59%|█████▉    | 236060/400000 [00:24<00:17, 9630.21it/s] 59%|█████▉    | 237026/400000 [00:24<00:17, 9535.87it/s] 60%|█████▉    | 238029/400000 [00:24<00:16, 9677.83it/s] 60%|█████▉    | 238999/400000 [00:24<00:16, 9652.99it/s] 60%|█████▉    | 239966/400000 [00:24<00:16, 9607.12it/s] 60%|██████    | 240943/400000 [00:24<00:16, 9653.33it/s] 60%|██████    | 241913/400000 [00:25<00:16, 9666.57it/s] 61%|██████    | 242881/400000 [00:25<00:16, 9670.41it/s] 61%|██████    | 243853/400000 [00:25<00:16, 9684.67it/s] 61%|██████    | 244840/400000 [00:25<00:15, 9736.96it/s] 61%|██████▏   | 245840/400000 [00:25<00:15, 9814.33it/s] 62%|██████▏   | 246826/400000 [00:25<00:15, 9825.64it/s] 62%|██████▏   | 247809/400000 [00:25<00:15, 9820.54it/s] 62%|██████▏   | 248800/400000 [00:25<00:15, 9847.12it/s] 62%|██████▏   | 249785/400000 [00:25<00:15, 9814.23it/s] 63%|██████▎   | 250767/400000 [00:25<00:15, 9808.86it/s] 63%|██████▎   | 251778/400000 [00:26<00:14, 9895.18it/s] 63%|██████▎   | 252773/400000 [00:26<00:14, 9909.07it/s] 63%|██████▎   | 253765/400000 [00:26<00:14, 9796.06it/s] 64%|██████▎   | 254746/400000 [00:26<00:14, 9776.82it/s] 64%|██████▍   | 255724/400000 [00:26<00:14, 9740.64it/s] 64%|██████▍   | 256699/400000 [00:26<00:14, 9667.66it/s] 64%|██████▍   | 257667/400000 [00:26<00:14, 9563.28it/s] 65%|██████▍   | 258624/400000 [00:26<00:14, 9493.95it/s] 65%|██████▍   | 259610/400000 [00:26<00:14, 9598.68it/s] 65%|██████▌   | 260615/400000 [00:26<00:14, 9729.35it/s] 65%|██████▌   | 261601/400000 [00:27<00:14, 9767.49it/s] 66%|██████▌   | 262579/400000 [00:27<00:14, 9753.38it/s] 66%|██████▌   | 263555/400000 [00:27<00:13, 9751.30it/s] 66%|██████▌   | 264556/400000 [00:27<00:13, 9826.99it/s] 66%|██████▋   | 265540/400000 [00:27<00:13, 9781.41it/s] 67%|██████▋   | 266543/400000 [00:27<00:13, 9853.92it/s] 67%|██████▋   | 267551/400000 [00:27<00:13, 9918.93it/s] 67%|██████▋   | 268544/400000 [00:27<00:13, 9761.47it/s] 67%|██████▋   | 269523/400000 [00:27<00:13, 9769.17it/s] 68%|██████▊   | 270503/400000 [00:28<00:13, 9775.61it/s] 68%|██████▊   | 271481/400000 [00:28<00:13, 9643.48it/s] 68%|██████▊   | 272447/400000 [00:28<00:13, 9270.88it/s] 68%|██████▊   | 273403/400000 [00:28<00:13, 9354.53it/s] 69%|██████▊   | 274361/400000 [00:28<00:13, 9418.65it/s] 69%|██████▉   | 275339/400000 [00:28<00:13, 9523.18it/s] 69%|██████▉   | 276313/400000 [00:28<00:12, 9585.44it/s] 69%|██████▉   | 277288/400000 [00:28<00:12, 9632.58it/s] 70%|██████▉   | 278281/400000 [00:28<00:12, 9719.84it/s] 70%|██████▉   | 279254/400000 [00:28<00:12, 9675.62it/s] 70%|███████   | 280258/400000 [00:29<00:12, 9779.57it/s] 70%|███████   | 281239/400000 [00:29<00:12, 9787.29it/s] 71%|███████   | 282219/400000 [00:29<00:12, 9783.63it/s] 71%|███████   | 283198/400000 [00:29<00:11, 9759.98it/s] 71%|███████   | 284175/400000 [00:29<00:12, 9627.85it/s] 71%|███████▏  | 285160/400000 [00:29<00:11, 9691.16it/s] 72%|███████▏  | 286130/400000 [00:29<00:11, 9638.20it/s] 72%|███████▏  | 287095/400000 [00:29<00:11, 9611.92it/s] 72%|███████▏  | 288071/400000 [00:29<00:11, 9654.44it/s] 72%|███████▏  | 289037/400000 [00:29<00:11, 9419.17it/s] 72%|███████▏  | 289990/400000 [00:30<00:11, 9450.11it/s] 73%|███████▎  | 291024/400000 [00:30<00:11, 9698.52it/s] 73%|███████▎  | 292046/400000 [00:30<00:10, 9848.63it/s] 73%|███████▎  | 293034/400000 [00:30<00:10, 9800.78it/s] 74%|███████▎  | 294023/400000 [00:30<00:10, 9827.07it/s] 74%|███████▍  | 295007/400000 [00:30<00:10, 9755.13it/s] 74%|███████▍  | 295984/400000 [00:30<00:10, 9720.63it/s] 74%|███████▍  | 296957/400000 [00:30<00:10, 9682.32it/s] 74%|███████▍  | 297959/400000 [00:30<00:10, 9780.48it/s] 75%|███████▍  | 298953/400000 [00:30<00:10, 9823.79it/s] 75%|███████▍  | 299936/400000 [00:31<00:10, 9772.77it/s] 75%|███████▌  | 300922/400000 [00:31<00:10, 9797.76it/s] 75%|███████▌  | 301905/400000 [00:31<00:10, 9804.66it/s] 76%|███████▌  | 302886/400000 [00:31<00:10, 9706.57it/s] 76%|███████▌  | 303858/400000 [00:31<00:09, 9678.00it/s] 76%|███████▌  | 304827/400000 [00:31<00:09, 9605.43it/s] 76%|███████▋  | 305797/400000 [00:31<00:09, 9632.56it/s] 77%|███████▋  | 306784/400000 [00:31<00:09, 9701.15it/s] 77%|███████▋  | 307782/400000 [00:31<00:09, 9781.56it/s] 77%|███████▋  | 308761/400000 [00:31<00:09, 9737.22it/s] 77%|███████▋  | 309766/400000 [00:32<00:09, 9827.10it/s] 78%|███████▊  | 310750/400000 [00:32<00:09, 9800.09it/s] 78%|███████▊  | 311731/400000 [00:32<00:09, 9774.53it/s] 78%|███████▊  | 312709/400000 [00:32<00:08, 9769.24it/s] 78%|███████▊  | 313687/400000 [00:32<00:08, 9679.75it/s] 79%|███████▊  | 314656/400000 [00:32<00:08, 9666.85it/s] 79%|███████▉  | 315624/400000 [00:32<00:08, 9669.06it/s] 79%|███████▉  | 316592/400000 [00:32<00:08, 9655.99it/s] 79%|███████▉  | 317582/400000 [00:32<00:08, 9725.85it/s] 80%|███████▉  | 318562/400000 [00:32<00:08, 9747.33it/s] 80%|███████▉  | 319537/400000 [00:33<00:08, 9556.78it/s] 80%|████████  | 320494/400000 [00:33<00:08, 9528.86it/s] 80%|████████  | 321448/400000 [00:33<00:08, 9443.21it/s] 81%|████████  | 322393/400000 [00:33<00:08, 9360.57it/s] 81%|████████  | 323330/400000 [00:33<00:08, 9308.02it/s] 81%|████████  | 324262/400000 [00:33<00:08, 9258.84it/s] 81%|████████▏ | 325201/400000 [00:33<00:08, 9297.06it/s] 82%|████████▏ | 326177/400000 [00:33<00:07, 9427.80it/s] 82%|████████▏ | 327155/400000 [00:33<00:07, 9530.22it/s] 82%|████████▏ | 328134/400000 [00:33<00:07, 9604.49it/s] 82%|████████▏ | 329096/400000 [00:34<00:07, 9562.26it/s] 83%|████████▎ | 330053/400000 [00:34<00:07, 9540.79it/s] 83%|████████▎ | 331008/400000 [00:34<00:07, 9495.02it/s] 83%|████████▎ | 331970/400000 [00:34<00:07, 9531.29it/s] 83%|████████▎ | 332929/400000 [00:34<00:07, 9547.76it/s] 83%|████████▎ | 333884/400000 [00:34<00:06, 9487.28it/s] 84%|████████▎ | 334833/400000 [00:34<00:06, 9353.87it/s] 84%|████████▍ | 335769/400000 [00:34<00:07, 8928.04it/s] 84%|████████▍ | 336700/400000 [00:34<00:07, 9037.92it/s] 84%|████████▍ | 337608/400000 [00:35<00:06, 8957.81it/s] 85%|████████▍ | 338513/400000 [00:35<00:06, 8984.98it/s] 85%|████████▍ | 339437/400000 [00:35<00:06, 9057.99it/s] 85%|████████▌ | 340347/400000 [00:35<00:06, 9069.41it/s] 85%|████████▌ | 341280/400000 [00:35<00:06, 9143.96it/s] 86%|████████▌ | 342215/400000 [00:35<00:06, 9203.54it/s] 86%|████████▌ | 343141/400000 [00:35<00:06, 9217.50it/s] 86%|████████▌ | 344097/400000 [00:35<00:06, 9316.38it/s] 86%|████████▋ | 345051/400000 [00:35<00:05, 9380.21it/s] 87%|████████▋ | 346021/400000 [00:35<00:05, 9471.83it/s] 87%|████████▋ | 347006/400000 [00:36<00:05, 9579.88it/s] 87%|████████▋ | 347965/400000 [00:36<00:05, 9570.27it/s] 87%|████████▋ | 348923/400000 [00:36<00:05, 9512.47it/s] 87%|████████▋ | 349897/400000 [00:36<00:05, 9576.85it/s] 88%|████████▊ | 350861/400000 [00:36<00:05, 9593.61it/s] 88%|████████▊ | 351826/400000 [00:36<00:05, 9609.05it/s] 88%|████████▊ | 352788/400000 [00:36<00:04, 9545.85it/s] 88%|████████▊ | 353743/400000 [00:36<00:04, 9474.87it/s] 89%|████████▊ | 354691/400000 [00:36<00:04, 9451.60it/s] 89%|████████▉ | 355642/400000 [00:36<00:04, 9466.34it/s] 89%|████████▉ | 356589/400000 [00:37<00:04, 9394.21it/s] 89%|████████▉ | 357529/400000 [00:37<00:04, 9282.76it/s] 90%|████████▉ | 358496/400000 [00:37<00:04, 9394.41it/s] 90%|████████▉ | 359464/400000 [00:37<00:04, 9477.03it/s] 90%|█████████ | 360413/400000 [00:37<00:04, 9294.82it/s] 90%|█████████ | 361363/400000 [00:37<00:04, 9352.22it/s] 91%|█████████ | 362305/400000 [00:37<00:04, 9370.32it/s] 91%|█████████ | 363268/400000 [00:37<00:03, 9445.49it/s] 91%|█████████ | 364240/400000 [00:37<00:03, 9525.64it/s] 91%|█████████▏| 365194/400000 [00:37<00:03, 9511.80it/s] 92%|█████████▏| 366146/400000 [00:38<00:03, 9335.62it/s] 92%|█████████▏| 367081/400000 [00:38<00:03, 9339.85it/s] 92%|█████████▏| 368054/400000 [00:38<00:03, 9450.69it/s] 92%|█████████▏| 369000/400000 [00:38<00:03, 9390.41it/s] 92%|█████████▏| 369940/400000 [00:38<00:03, 9324.50it/s] 93%|█████████▎| 370888/400000 [00:38<00:03, 9367.88it/s] 93%|█████████▎| 371829/400000 [00:38<00:03, 9378.10it/s] 93%|█████████▎| 372802/400000 [00:38<00:02, 9480.57it/s] 93%|█████████▎| 373799/400000 [00:38<00:02, 9620.07it/s] 94%|█████████▎| 374762/400000 [00:38<00:02, 9622.75it/s] 94%|█████████▍| 375759/400000 [00:39<00:02, 9723.89it/s] 94%|█████████▍| 376733/400000 [00:39<00:02, 9661.39it/s] 94%|█████████▍| 377700/400000 [00:39<00:02, 9644.29it/s] 95%|█████████▍| 378665/400000 [00:39<00:02, 9566.77it/s] 95%|█████████▍| 379623/400000 [00:39<00:02, 9472.14it/s] 95%|█████████▌| 380575/400000 [00:39<00:02, 9485.02it/s] 95%|█████████▌| 381524/400000 [00:39<00:01, 9476.34it/s] 96%|█████████▌| 382482/400000 [00:39<00:01, 9507.03it/s] 96%|█████████▌| 383441/400000 [00:39<00:01, 9531.23it/s] 96%|█████████▌| 384415/400000 [00:39<00:01, 9590.94it/s] 96%|█████████▋| 385375/400000 [00:40<00:01, 9338.00it/s] 97%|█████████▋| 386311/400000 [00:40<00:01, 9227.41it/s] 97%|█████████▋| 387288/400000 [00:40<00:01, 9381.85it/s] 97%|█████████▋| 388231/400000 [00:40<00:01, 9395.40it/s] 97%|█████████▋| 389172/400000 [00:40<00:01, 9394.76it/s] 98%|█████████▊| 390121/400000 [00:40<00:01, 9421.50it/s] 98%|█████████▊| 391064/400000 [00:40<00:00, 9358.69it/s] 98%|█████████▊| 392052/400000 [00:40<00:00, 9507.86it/s] 98%|█████████▊| 393050/400000 [00:40<00:00, 9644.12it/s] 99%|█████████▊| 394041/400000 [00:40<00:00, 9719.11it/s] 99%|█████████▉| 395014/400000 [00:41<00:00, 9682.88it/s] 99%|█████████▉| 395983/400000 [00:41<00:00, 9640.09it/s] 99%|█████████▉| 396948/400000 [00:41<00:00, 9630.25it/s] 99%|█████████▉| 397921/400000 [00:41<00:00, 9659.95it/s]100%|█████████▉| 398888/400000 [00:41<00:00, 9633.63it/s]100%|█████████▉| 399875/400000 [00:41<00:00, 9701.43it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9616.48it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd261b184e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.0116498534259878 	 Accuracy: 46
Train Epoch: 1 	 Loss: 0.01173095001424834 	 Accuracy: 48

  model saves at 48% accuracy 

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

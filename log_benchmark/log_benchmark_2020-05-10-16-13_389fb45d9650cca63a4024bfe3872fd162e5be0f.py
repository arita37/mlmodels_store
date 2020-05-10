
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fc4aab3e470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 16:13:21.853658
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 16:13:21.857481
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 16:13:21.860513
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 16:13:21.863641
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fc4a2e8e438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356072.9062
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 283602.0000
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 171329.2969
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 103997.1250
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 60639.7031
Epoch 6/10

1/1 [==============================] - 0s 91ms/step - loss: 33932.4727
Epoch 7/10

1/1 [==============================] - 0s 112ms/step - loss: 20440.9219
Epoch 8/10

1/1 [==============================] - 0s 98ms/step - loss: 13662.9746
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 9844.8799
Epoch 10/10

1/1 [==============================] - 0s 107ms/step - loss: 7495.0117

  #### Inference Need return ypred, ytrue ######################### 
[[-3.24862450e-03  7.10632467e+00  7.90610123e+00  6.98219919e+00
   8.53865719e+00  7.64763212e+00  8.06737328e+00  7.86302137e+00
   5.72360992e+00  8.03190994e+00  7.21282673e+00  7.71811342e+00
   8.39780140e+00  8.56686687e+00  7.56449413e+00  6.66773939e+00
   7.58420420e+00  9.16849041e+00  7.37424564e+00  8.22808933e+00
   7.77990103e+00  7.22889996e+00  8.36122990e+00  7.20219374e+00
   7.27257824e+00  7.71248817e+00  7.22360516e+00  7.46849346e+00
   8.21630287e+00  6.70090866e+00  7.86043644e+00  6.76347351e+00
   8.25018406e+00  6.89010382e+00  8.60076427e+00  6.34663343e+00
   6.90650415e+00  7.82977343e+00  6.93135738e+00  6.65743971e+00
   7.63089371e+00  8.03859901e+00  6.15805006e+00  8.75708389e+00
   6.52297688e+00  7.03819513e+00  7.21447897e+00  9.31244946e+00
   7.53538418e+00  6.57232809e+00  8.86261654e+00  9.38871479e+00
   6.41520166e+00  7.43307686e+00  7.90115452e+00  9.14376545e+00
   9.40079498e+00  6.94719124e+00  7.49716902e+00  8.33724499e+00
   9.20980096e-01 -4.80154604e-01 -8.33601296e-01  6.93817139e-01
   9.11368549e-01 -5.58449566e-01  1.83283359e-01  6.87637851e-02
  -6.22158289e-01 -6.73695326e-01  4.56238747e-01 -1.39602661e+00
  -5.43990135e-01  1.04217803e+00  1.59852397e+00  2.40055770e-02
   8.59039247e-01  7.09516943e-01 -3.22735310e-01  2.09481025e+00
   7.89986134e-01  1.85286629e+00 -1.18944561e+00  1.98151278e+00
   2.58823782e-01  1.11426294e-01  4.78354216e-01  4.87286240e-01
  -1.08368528e+00  2.93364316e-01 -5.02614379e-02 -1.14256978e+00
   3.64445448e-01  8.35340261e-01 -1.57408237e-01 -1.08259964e+00
  -6.15287572e-03 -2.43164897e-02  1.23936415e-01  1.21969986e+00
   9.51501608e-01  1.24128163e-01  1.84069484e-01 -3.95853639e-01
  -7.09200740e-01 -7.90932477e-01 -5.66617727e-01  2.13335133e+00
  -6.90154791e-01  1.06539190e+00 -1.63633192e+00 -1.27341354e+00
  -2.44357800e+00 -1.24470937e+00  1.85162425e-02 -1.74120873e-01
   8.12957466e-01  5.64538419e-01  6.11150563e-01  5.14507771e-01
   6.29088879e-01  7.87490368e-01  5.91746986e-01 -2.98252463e-01
   3.33623886e-02 -3.98900896e-01  4.73451704e-01  1.50293171e-01
   9.23785329e-01  1.22392178e-02 -1.20966721e+00 -4.86042529e-01
   1.41369730e-01  4.78143990e-02 -2.92315125e-01 -4.51348156e-01
  -7.01596200e-01  4.67902124e-02 -5.75682521e-02  7.74742246e-01
   5.35550058e-01 -1.31515950e-01  8.10900927e-01 -9.02372301e-01
  -4.10093546e-01 -1.34173036e+00  1.83939576e+00 -4.60394263e-01
  -9.33897793e-01 -7.93620795e-02  1.12805867e+00 -4.04253989e-01
  -5.59296757e-02  3.61798346e-01 -7.04691231e-01  5.55806398e-01
   3.90304476e-01  6.73191905e-01 -6.16869092e-01  1.06532574e+00
   1.37244558e+00  5.05020499e-01 -8.67174089e-01  2.38110393e-01
   1.57327926e+00 -1.00741458e+00 -4.13132280e-01 -2.00841665e-01
  -1.01323342e+00 -7.10247874e-01 -3.91059875e-01  6.67601228e-01
   1.15140900e-01  1.01429617e+00 -6.01350069e-01  5.80744445e-02
   1.44380629e-02  6.48906767e-01  1.12830973e+00  6.77504539e-01
   8.55720639e-02  7.21934748e+00  6.77939177e+00  6.71338701e+00
   7.66950369e+00  7.42656517e+00  7.67256355e+00  7.21532202e+00
   8.36937714e+00  7.42302847e+00  7.48442936e+00  7.74549723e+00
   8.91592216e+00  8.26918602e+00  8.41814899e+00  7.34862757e+00
   7.86235523e+00  8.24625015e+00  9.41838264e+00  7.76162004e+00
   6.43347216e+00  7.14934540e+00  8.87645721e+00  7.50563192e+00
   8.25189304e+00  7.10863924e+00  7.74557829e+00  8.81878757e+00
   7.00314856e+00  8.90810871e+00  6.77601290e+00  6.83719301e+00
   7.74033308e+00  8.96193409e+00  8.13315964e+00  7.90172577e+00
   7.52558470e+00  8.14216423e+00  7.63583612e+00  8.56522846e+00
   7.62880421e+00  8.64343452e+00  7.84248447e+00  6.91086197e+00
   7.56073761e+00  6.75110579e+00  8.86372566e+00  7.45724535e+00
   7.95568657e+00  7.99738646e+00  9.09249973e+00  8.42767334e+00
   7.35816813e+00  8.74318027e+00  9.02444363e+00  7.25098324e+00
   8.50630665e+00  6.97853041e+00  8.57389641e+00  7.94364691e+00
   6.05564535e-01  3.90900433e-01  8.79632115e-01  1.19798827e+00
   1.66421521e+00  1.38906050e+00  5.68782210e-01  5.80103457e-01
   5.71254313e-01  9.73939419e-01  5.26751995e-01  5.55998266e-01
   7.99288929e-01  2.01229429e+00  1.59925544e+00  5.84420800e-01
   1.45078135e+00  2.55434394e-01  1.54750788e+00  5.66979527e-01
   6.69770658e-01  1.64811230e+00  2.81957173e+00  6.75306559e-01
   1.26979661e+00  5.35973370e-01  4.70071435e-01  5.24269819e-01
   7.90378809e-01  1.47525859e+00  1.96372867e-01  6.24517858e-01
   4.87314165e-01  7.67944455e-01  1.01057339e+00  1.55037832e+00
   1.63279128e+00  6.78036809e-01  1.29413855e+00  5.02647877e-01
   2.60758781e+00  4.60792124e-01  2.24817812e-01  1.68412066e+00
   1.87601972e+00  4.50696945e-01  3.45476151e-01  4.99001622e-01
   1.47050142e-01  6.58860803e-01  6.79749787e-01  2.54021573e+00
   2.26785469e+00  1.80766749e+00  1.16389537e+00  3.14673424e+00
   1.45675898e-01  1.52639318e+00  1.18153739e+00  1.43745184e+00
   1.38386941e+00  7.54622817e-01  9.21551228e-01  5.01494110e-01
   4.91978645e-01  1.63005424e+00  1.93358386e+00  7.06688702e-01
   2.94650555e-01  1.00009131e+00  1.49419606e+00  1.61868858e+00
   1.08771014e+00  1.75372887e+00  6.39200449e-01  5.10826588e-01
   4.26437855e-01  2.83049774e+00  2.47640419e+00  3.95176291e-01
   5.85089624e-01  2.04017925e+00  9.11532342e-01  2.20723939e+00
   2.13386178e-01  5.78380883e-01  4.09852862e-01  2.15456247e-01
   2.66001749e+00  4.09267604e-01  2.07418501e-01  1.46462059e+00
   6.02133274e-01  1.58308697e+00  1.61845779e+00  1.44925439e+00
   1.23277426e+00  1.53959239e+00  1.05739045e+00  3.12462986e-01
   1.77113986e+00  8.99919093e-01  5.99964082e-01  2.45899975e-01
   5.93379021e-01  2.89959860e+00  3.59320045e-01  1.34644198e+00
   5.46108246e-01  1.81786048e+00  1.49921858e+00  3.43166471e-01
   1.12500954e+00  1.42515957e+00  3.18415165e-01  1.71371460e+00
   2.00715160e+00  7.60242879e-01  8.33638012e-01  6.73552632e-01
   6.41266108e+00 -5.69198990e+00 -6.46173382e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 16:13:30.381351
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6711
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 16:13:30.385085
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    8982.7
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 16:13:30.388232
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.5652
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 16:13:30.391786
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -803.464
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140481965359568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140481023849416
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140481023849920
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140481023850424
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140481023850928
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140481023851432

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fc4a2e8e3c8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.538729
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.516518
grad_step = 000002, loss = 0.500119
grad_step = 000003, loss = 0.485214
grad_step = 000004, loss = 0.470155
grad_step = 000005, loss = 0.456251
grad_step = 000006, loss = 0.445583
grad_step = 000007, loss = 0.437976
grad_step = 000008, loss = 0.426190
grad_step = 000009, loss = 0.413147
grad_step = 000010, loss = 0.402085
grad_step = 000011, loss = 0.392619
grad_step = 000012, loss = 0.383739
grad_step = 000013, loss = 0.374943
grad_step = 000014, loss = 0.366173
grad_step = 000015, loss = 0.356796
grad_step = 000016, loss = 0.346931
grad_step = 000017, loss = 0.337216
grad_step = 000018, loss = 0.327714
grad_step = 000019, loss = 0.317971
grad_step = 000020, loss = 0.308175
grad_step = 000021, loss = 0.298835
grad_step = 000022, loss = 0.289979
grad_step = 000023, loss = 0.281111
grad_step = 000024, loss = 0.272015
grad_step = 000025, loss = 0.263085
grad_step = 000026, loss = 0.254416
grad_step = 000027, loss = 0.245686
grad_step = 000028, loss = 0.236897
grad_step = 000029, loss = 0.228408
grad_step = 000030, loss = 0.220167
grad_step = 000031, loss = 0.212001
grad_step = 000032, loss = 0.204055
grad_step = 000033, loss = 0.196317
grad_step = 000034, loss = 0.188731
grad_step = 000035, loss = 0.181318
grad_step = 000036, loss = 0.174015
grad_step = 000037, loss = 0.166890
grad_step = 000038, loss = 0.160024
grad_step = 000039, loss = 0.153274
grad_step = 000040, loss = 0.146782
grad_step = 000041, loss = 0.140564
grad_step = 000042, loss = 0.134436
grad_step = 000043, loss = 0.128569
grad_step = 000044, loss = 0.122899
grad_step = 000045, loss = 0.117350
grad_step = 000046, loss = 0.112063
grad_step = 000047, loss = 0.106910
grad_step = 000048, loss = 0.101977
grad_step = 000049, loss = 0.097253
grad_step = 000050, loss = 0.092685
grad_step = 000051, loss = 0.088341
grad_step = 000052, loss = 0.084144
grad_step = 000053, loss = 0.080131
grad_step = 000054, loss = 0.076292
grad_step = 000055, loss = 0.072620
grad_step = 000056, loss = 0.069097
grad_step = 000057, loss = 0.065743
grad_step = 000058, loss = 0.062523
grad_step = 000059, loss = 0.059470
grad_step = 000060, loss = 0.056545
grad_step = 000061, loss = 0.053771
grad_step = 000062, loss = 0.051108
grad_step = 000063, loss = 0.048574
grad_step = 000064, loss = 0.046154
grad_step = 000065, loss = 0.043849
grad_step = 000066, loss = 0.041653
grad_step = 000067, loss = 0.039563
grad_step = 000068, loss = 0.037573
grad_step = 000069, loss = 0.035674
grad_step = 000070, loss = 0.033871
grad_step = 000071, loss = 0.032152
grad_step = 000072, loss = 0.030520
grad_step = 000073, loss = 0.028963
grad_step = 000074, loss = 0.027486
grad_step = 000075, loss = 0.026079
grad_step = 000076, loss = 0.024744
grad_step = 000077, loss = 0.023476
grad_step = 000078, loss = 0.022273
grad_step = 000079, loss = 0.021131
grad_step = 000080, loss = 0.020046
grad_step = 000081, loss = 0.019016
grad_step = 000082, loss = 0.018041
grad_step = 000083, loss = 0.017117
grad_step = 000084, loss = 0.016241
grad_step = 000085, loss = 0.015412
grad_step = 000086, loss = 0.014628
grad_step = 000087, loss = 0.013885
grad_step = 000088, loss = 0.013185
grad_step = 000089, loss = 0.012522
grad_step = 000090, loss = 0.011896
grad_step = 000091, loss = 0.011303
grad_step = 000092, loss = 0.010746
grad_step = 000093, loss = 0.010219
grad_step = 000094, loss = 0.009723
grad_step = 000095, loss = 0.009254
grad_step = 000096, loss = 0.008812
grad_step = 000097, loss = 0.008396
grad_step = 000098, loss = 0.008004
grad_step = 000099, loss = 0.007635
grad_step = 000100, loss = 0.007287
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.006959
grad_step = 000102, loss = 0.006652
grad_step = 000103, loss = 0.006364
grad_step = 000104, loss = 0.006097
grad_step = 000105, loss = 0.005857
grad_step = 000106, loss = 0.005653
grad_step = 000107, loss = 0.005466
grad_step = 000108, loss = 0.005275
grad_step = 000109, loss = 0.005012
grad_step = 000110, loss = 0.004764
grad_step = 000111, loss = 0.004594
grad_step = 000112, loss = 0.004476
grad_step = 000113, loss = 0.004336
grad_step = 000114, loss = 0.004143
grad_step = 000115, loss = 0.003974
grad_step = 000116, loss = 0.003864
grad_step = 000117, loss = 0.003772
grad_step = 000118, loss = 0.003657
grad_step = 000119, loss = 0.003518
grad_step = 000120, loss = 0.003404
grad_step = 000121, loss = 0.003327
grad_step = 000122, loss = 0.003258
grad_step = 000123, loss = 0.003177
grad_step = 000124, loss = 0.003080
grad_step = 000125, loss = 0.002992
grad_step = 000126, loss = 0.002927
grad_step = 000127, loss = 0.002877
grad_step = 000128, loss = 0.002829
grad_step = 000129, loss = 0.002772
grad_step = 000130, loss = 0.002710
grad_step = 000131, loss = 0.002650
grad_step = 000132, loss = 0.002598
grad_step = 000133, loss = 0.002557
grad_step = 000134, loss = 0.002523
grad_step = 000135, loss = 0.002495
grad_step = 000136, loss = 0.002470
grad_step = 000137, loss = 0.002448
grad_step = 000138, loss = 0.002429
grad_step = 000139, loss = 0.002418
grad_step = 000140, loss = 0.002404
grad_step = 000141, loss = 0.002387
grad_step = 000142, loss = 0.002353
grad_step = 000143, loss = 0.002308
grad_step = 000144, loss = 0.002257
grad_step = 000145, loss = 0.002216
grad_step = 000146, loss = 0.002193
grad_step = 000147, loss = 0.002185
grad_step = 000148, loss = 0.002187
grad_step = 000149, loss = 0.002192
grad_step = 000150, loss = 0.002198
grad_step = 000151, loss = 0.002196
grad_step = 000152, loss = 0.002187
grad_step = 000153, loss = 0.002163
grad_step = 000154, loss = 0.002128
grad_step = 000155, loss = 0.002092
grad_step = 000156, loss = 0.002064
grad_step = 000157, loss = 0.002051
grad_step = 000158, loss = 0.002049
grad_step = 000159, loss = 0.002055
grad_step = 000160, loss = 0.002067
grad_step = 000161, loss = 0.002082
grad_step = 000162, loss = 0.002102
grad_step = 000163, loss = 0.002116
grad_step = 000164, loss = 0.002120
grad_step = 000165, loss = 0.002082
grad_step = 000166, loss = 0.002032
grad_step = 000167, loss = 0.001993
grad_step = 000168, loss = 0.001986
grad_step = 000169, loss = 0.002005
grad_step = 000170, loss = 0.002021
grad_step = 000171, loss = 0.002022
grad_step = 000172, loss = 0.001987
grad_step = 000173, loss = 0.001958
grad_step = 000174, loss = 0.001957
grad_step = 000175, loss = 0.001977
grad_step = 000176, loss = 0.001995
grad_step = 000177, loss = 0.001982
grad_step = 000178, loss = 0.001970
grad_step = 000179, loss = 0.001979
grad_step = 000180, loss = 0.002032
grad_step = 000181, loss = 0.002109
grad_step = 000182, loss = 0.002196
grad_step = 000183, loss = 0.002195
grad_step = 000184, loss = 0.002115
grad_step = 000185, loss = 0.001956
grad_step = 000186, loss = 0.001929
grad_step = 000187, loss = 0.002023
grad_step = 000188, loss = 0.002053
grad_step = 000189, loss = 0.001979
grad_step = 000190, loss = 0.001916
grad_step = 000191, loss = 0.001952
grad_step = 000192, loss = 0.001989
grad_step = 000193, loss = 0.001938
grad_step = 000194, loss = 0.001915
grad_step = 000195, loss = 0.001960
grad_step = 000196, loss = 0.001968
grad_step = 000197, loss = 0.001933
grad_step = 000198, loss = 0.001916
grad_step = 000199, loss = 0.001936
grad_step = 000200, loss = 0.001930
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001895
grad_step = 000202, loss = 0.001890
grad_step = 000203, loss = 0.001908
grad_step = 000204, loss = 0.001902
grad_step = 000205, loss = 0.001883
grad_step = 000206, loss = 0.001885
grad_step = 000207, loss = 0.001903
grad_step = 000208, loss = 0.001917
grad_step = 000209, loss = 0.001923
grad_step = 000210, loss = 0.001970
grad_step = 000211, loss = 0.001989
grad_step = 000212, loss = 0.001970
grad_step = 000213, loss = 0.001913
grad_step = 000214, loss = 0.001879
grad_step = 000215, loss = 0.001900
grad_step = 000216, loss = 0.001951
grad_step = 000217, loss = 0.001979
grad_step = 000218, loss = 0.001970
grad_step = 000219, loss = 0.001995
grad_step = 000220, loss = 0.001975
grad_step = 000221, loss = 0.001920
grad_step = 000222, loss = 0.001866
grad_step = 000223, loss = 0.001897
grad_step = 000224, loss = 0.001942
grad_step = 000225, loss = 0.001899
grad_step = 000226, loss = 0.001855
grad_step = 000227, loss = 0.001876
grad_step = 000228, loss = 0.001904
grad_step = 000229, loss = 0.001885
grad_step = 000230, loss = 0.001853
grad_step = 000231, loss = 0.001861
grad_step = 000232, loss = 0.001881
grad_step = 000233, loss = 0.001869
grad_step = 000234, loss = 0.001851
grad_step = 000235, loss = 0.001854
grad_step = 000236, loss = 0.001861
grad_step = 000237, loss = 0.001855
grad_step = 000238, loss = 0.001846
grad_step = 000239, loss = 0.001848
grad_step = 000240, loss = 0.001851
grad_step = 000241, loss = 0.001845
grad_step = 000242, loss = 0.001837
grad_step = 000243, loss = 0.001838
grad_step = 000244, loss = 0.001843
grad_step = 000245, loss = 0.001841
grad_step = 000246, loss = 0.001833
grad_step = 000247, loss = 0.001829
grad_step = 000248, loss = 0.001831
grad_step = 000249, loss = 0.001834
grad_step = 000250, loss = 0.001831
grad_step = 000251, loss = 0.001832
grad_step = 000252, loss = 0.001826
grad_step = 000253, loss = 0.001828
grad_step = 000254, loss = 0.001833
grad_step = 000255, loss = 0.001838
grad_step = 000256, loss = 0.001844
grad_step = 000257, loss = 0.001845
grad_step = 000258, loss = 0.001844
grad_step = 000259, loss = 0.001837
grad_step = 000260, loss = 0.001831
grad_step = 000261, loss = 0.001823
grad_step = 000262, loss = 0.001817
grad_step = 000263, loss = 0.001813
grad_step = 000264, loss = 0.001809
grad_step = 000265, loss = 0.001806
grad_step = 000266, loss = 0.001804
grad_step = 000267, loss = 0.001803
grad_step = 000268, loss = 0.001803
grad_step = 000269, loss = 0.001805
grad_step = 000270, loss = 0.001809
grad_step = 000271, loss = 0.001821
grad_step = 000272, loss = 0.001848
grad_step = 000273, loss = 0.001912
grad_step = 000274, loss = 0.002000
grad_step = 000275, loss = 0.002138
grad_step = 000276, loss = 0.002072
grad_step = 000277, loss = 0.001918
grad_step = 000278, loss = 0.001798
grad_step = 000279, loss = 0.001884
grad_step = 000280, loss = 0.001973
grad_step = 000281, loss = 0.001854
grad_step = 000282, loss = 0.001793
grad_step = 000283, loss = 0.001873
grad_step = 000284, loss = 0.001875
grad_step = 000285, loss = 0.001803
grad_step = 000286, loss = 0.001796
grad_step = 000287, loss = 0.001848
grad_step = 000288, loss = 0.001842
grad_step = 000289, loss = 0.001787
grad_step = 000290, loss = 0.001801
grad_step = 000291, loss = 0.001833
grad_step = 000292, loss = 0.001798
grad_step = 000293, loss = 0.001779
grad_step = 000294, loss = 0.001805
grad_step = 000295, loss = 0.001799
grad_step = 000296, loss = 0.001775
grad_step = 000297, loss = 0.001783
grad_step = 000298, loss = 0.001794
grad_step = 000299, loss = 0.001779
grad_step = 000300, loss = 0.001770
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001782
grad_step = 000302, loss = 0.001783
grad_step = 000303, loss = 0.001770
grad_step = 000304, loss = 0.001769
grad_step = 000305, loss = 0.001779
grad_step = 000306, loss = 0.001779
grad_step = 000307, loss = 0.001776
grad_step = 000308, loss = 0.001786
grad_step = 000309, loss = 0.001812
grad_step = 000310, loss = 0.001839
grad_step = 000311, loss = 0.001887
grad_step = 000312, loss = 0.001930
grad_step = 000313, loss = 0.001988
grad_step = 000314, loss = 0.001933
grad_step = 000315, loss = 0.001848
grad_step = 000316, loss = 0.001765
grad_step = 000317, loss = 0.001766
grad_step = 000318, loss = 0.001823
grad_step = 000319, loss = 0.001842
grad_step = 000320, loss = 0.001809
grad_step = 000321, loss = 0.001757
grad_step = 000322, loss = 0.001759
grad_step = 000323, loss = 0.001797
grad_step = 000324, loss = 0.001800
grad_step = 000325, loss = 0.001770
grad_step = 000326, loss = 0.001744
grad_step = 000327, loss = 0.001757
grad_step = 000328, loss = 0.001779
grad_step = 000329, loss = 0.001772
grad_step = 000330, loss = 0.001749
grad_step = 000331, loss = 0.001740
grad_step = 000332, loss = 0.001751
grad_step = 000333, loss = 0.001759
grad_step = 000334, loss = 0.001752
grad_step = 000335, loss = 0.001742
grad_step = 000336, loss = 0.001740
grad_step = 000337, loss = 0.001744
grad_step = 000338, loss = 0.001743
grad_step = 000339, loss = 0.001736
grad_step = 000340, loss = 0.001733
grad_step = 000341, loss = 0.001736
grad_step = 000342, loss = 0.001740
grad_step = 000343, loss = 0.001739
grad_step = 000344, loss = 0.001734
grad_step = 000345, loss = 0.001730
grad_step = 000346, loss = 0.001730
grad_step = 000347, loss = 0.001731
grad_step = 000348, loss = 0.001729
grad_step = 000349, loss = 0.001726
grad_step = 000350, loss = 0.001723
grad_step = 000351, loss = 0.001723
grad_step = 000352, loss = 0.001724
grad_step = 000353, loss = 0.001724
grad_step = 000354, loss = 0.001723
grad_step = 000355, loss = 0.001723
grad_step = 000356, loss = 0.001725
grad_step = 000357, loss = 0.001730
grad_step = 000358, loss = 0.001740
grad_step = 000359, loss = 0.001755
grad_step = 000360, loss = 0.001782
grad_step = 000361, loss = 0.001815
grad_step = 000362, loss = 0.001869
grad_step = 000363, loss = 0.001883
grad_step = 000364, loss = 0.001879
grad_step = 000365, loss = 0.001789
grad_step = 000366, loss = 0.001725
grad_step = 000367, loss = 0.001741
grad_step = 000368, loss = 0.001774
grad_step = 000369, loss = 0.001766
grad_step = 000370, loss = 0.001733
grad_step = 000371, loss = 0.001736
grad_step = 000372, loss = 0.001752
grad_step = 000373, loss = 0.001737
grad_step = 000374, loss = 0.001715
grad_step = 000375, loss = 0.001716
grad_step = 000376, loss = 0.001734
grad_step = 000377, loss = 0.001738
grad_step = 000378, loss = 0.001720
grad_step = 000379, loss = 0.001706
grad_step = 000380, loss = 0.001708
grad_step = 000381, loss = 0.001721
grad_step = 000382, loss = 0.001723
grad_step = 000383, loss = 0.001711
grad_step = 000384, loss = 0.001698
grad_step = 000385, loss = 0.001695
grad_step = 000386, loss = 0.001701
grad_step = 000387, loss = 0.001706
grad_step = 000388, loss = 0.001703
grad_step = 000389, loss = 0.001695
grad_step = 000390, loss = 0.001692
grad_step = 000391, loss = 0.001695
grad_step = 000392, loss = 0.001700
grad_step = 000393, loss = 0.001700
grad_step = 000394, loss = 0.001697
grad_step = 000395, loss = 0.001693
grad_step = 000396, loss = 0.001693
grad_step = 000397, loss = 0.001695
grad_step = 000398, loss = 0.001699
grad_step = 000399, loss = 0.001699
grad_step = 000400, loss = 0.001699
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001699
grad_step = 000402, loss = 0.001703
grad_step = 000403, loss = 0.001710
grad_step = 000404, loss = 0.001722
grad_step = 000405, loss = 0.001734
grad_step = 000406, loss = 0.001754
grad_step = 000407, loss = 0.001772
grad_step = 000408, loss = 0.001805
grad_step = 000409, loss = 0.001830
grad_step = 000410, loss = 0.001857
grad_step = 000411, loss = 0.001820
grad_step = 000412, loss = 0.001764
grad_step = 000413, loss = 0.001696
grad_step = 000414, loss = 0.001672
grad_step = 000415, loss = 0.001697
grad_step = 000416, loss = 0.001729
grad_step = 000417, loss = 0.001742
grad_step = 000418, loss = 0.001716
grad_step = 000419, loss = 0.001686
grad_step = 000420, loss = 0.001673
grad_step = 000421, loss = 0.001679
grad_step = 000422, loss = 0.001689
grad_step = 000423, loss = 0.001693
grad_step = 000424, loss = 0.001691
grad_step = 000425, loss = 0.001682
grad_step = 000426, loss = 0.001673
grad_step = 000427, loss = 0.001665
grad_step = 000428, loss = 0.001663
grad_step = 000429, loss = 0.001668
grad_step = 000430, loss = 0.001674
grad_step = 000431, loss = 0.001678
grad_step = 000432, loss = 0.001673
grad_step = 000433, loss = 0.001665
grad_step = 000434, loss = 0.001657
grad_step = 000435, loss = 0.001653
grad_step = 000436, loss = 0.001653
grad_step = 000437, loss = 0.001654
grad_step = 000438, loss = 0.001656
grad_step = 000439, loss = 0.001656
grad_step = 000440, loss = 0.001655
grad_step = 000441, loss = 0.001655
grad_step = 000442, loss = 0.001654
grad_step = 000443, loss = 0.001652
grad_step = 000444, loss = 0.001650
grad_step = 000445, loss = 0.001646
grad_step = 000446, loss = 0.001643
grad_step = 000447, loss = 0.001641
grad_step = 000448, loss = 0.001641
grad_step = 000449, loss = 0.001640
grad_step = 000450, loss = 0.001639
grad_step = 000451, loss = 0.001638
grad_step = 000452, loss = 0.001637
grad_step = 000453, loss = 0.001636
grad_step = 000454, loss = 0.001636
grad_step = 000455, loss = 0.001637
grad_step = 000456, loss = 0.001639
grad_step = 000457, loss = 0.001642
grad_step = 000458, loss = 0.001648
grad_step = 000459, loss = 0.001661
grad_step = 000460, loss = 0.001683
grad_step = 000461, loss = 0.001727
grad_step = 000462, loss = 0.001779
grad_step = 000463, loss = 0.001860
grad_step = 000464, loss = 0.001859
grad_step = 000465, loss = 0.001798
grad_step = 000466, loss = 0.001698
grad_step = 000467, loss = 0.001666
grad_step = 000468, loss = 0.001687
grad_step = 000469, loss = 0.001694
grad_step = 000470, loss = 0.001684
grad_step = 000471, loss = 0.001673
grad_step = 000472, loss = 0.001681
grad_step = 000473, loss = 0.001677
grad_step = 000474, loss = 0.001647
grad_step = 000475, loss = 0.001632
grad_step = 000476, loss = 0.001647
grad_step = 000477, loss = 0.001665
grad_step = 000478, loss = 0.001655
grad_step = 000479, loss = 0.001626
grad_step = 000480, loss = 0.001614
grad_step = 000481, loss = 0.001628
grad_step = 000482, loss = 0.001643
grad_step = 000483, loss = 0.001639
grad_step = 000484, loss = 0.001621
grad_step = 000485, loss = 0.001609
grad_step = 000486, loss = 0.001612
grad_step = 000487, loss = 0.001622
grad_step = 000488, loss = 0.001625
grad_step = 000489, loss = 0.001616
grad_step = 000490, loss = 0.001605
grad_step = 000491, loss = 0.001602
grad_step = 000492, loss = 0.001606
grad_step = 000493, loss = 0.001611
grad_step = 000494, loss = 0.001609
grad_step = 000495, loss = 0.001603
grad_step = 000496, loss = 0.001597
grad_step = 000497, loss = 0.001597
grad_step = 000498, loss = 0.001599
grad_step = 000499, loss = 0.001600
grad_step = 000500, loss = 0.001598
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001594
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

  date_run                              2020-05-10 16:13:49.466612
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.23439
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 16:13:49.473801
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13356
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 16:13:49.482057
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.144247
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 16:13:49.488258
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.02949
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
0   2020-05-10 16:13:21.853658  ...    mean_absolute_error
1   2020-05-10 16:13:21.857481  ...     mean_squared_error
2   2020-05-10 16:13:21.860513  ...  median_absolute_error
3   2020-05-10 16:13:21.863641  ...               r2_score
4   2020-05-10 16:13:30.381351  ...    mean_absolute_error
5   2020-05-10 16:13:30.385085  ...     mean_squared_error
6   2020-05-10 16:13:30.388232  ...  median_absolute_error
7   2020-05-10 16:13:30.391786  ...               r2_score
8   2020-05-10 16:13:49.466612  ...    mean_absolute_error
9   2020-05-10 16:13:49.473801  ...     mean_squared_error
10  2020-05-10 16:13:49.482057  ...  median_absolute_error
11  2020-05-10 16:13:49.488258  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:20, 123283.32it/s] 12%|█▏        | 1146880/9912422 [00:00<00:50, 175291.15it/s] 30%|██▉       | 2957312/9912422 [00:00<00:27, 249354.28it/s] 73%|███████▎  | 7282688/9912422 [00:00<00:07, 355301.68it/s]9920512it [00:00, 21053921.05it/s]                           
0it [00:00, ?it/s]32768it [00:00, 578281.27it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 483833.22it/s]1654784it [00:00, 11747734.29it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191768.51it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12f1e80780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f128f5c4a58> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12f1e36e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f128f5c4da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12f1e36e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1298ce44a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12f1e80e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12a4832cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12f1e80e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f128f5c5048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f12f1e36e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd2f7a901d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=26fab22f7f5afdafde63a1b00126c8a34198d4c0aae366b9facde0c13df653df
  Stored in directory: /tmp/pip-ephem-wheel-cache-jrh79jay/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd28f675048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2547712/17464789 [===>..........................] - ETA: 0s
 9773056/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 16:15:15.296821: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 16:15:15.303047: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095159999 Hz
2020-05-10 16:15:15.303198: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c87207e110 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 16:15:15.303212: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6027 - accuracy: 0.5042
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5861 - accuracy: 0.5052
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6274 - accuracy: 0.5026
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5976 - accuracy: 0.5045
11000/25000 [============>.................] - ETA: 3s - loss: 7.5788 - accuracy: 0.5057
12000/25000 [=============>................] - ETA: 3s - loss: 7.5861 - accuracy: 0.5052
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5935 - accuracy: 0.5048
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6086 - accuracy: 0.5038
15000/25000 [=================>............] - ETA: 2s - loss: 7.6298 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6484 - accuracy: 0.5012
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6495 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6712 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6771 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 7s 271us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 16:15:28.442128
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 16:15:28.442128  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 16:15:34.289811: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 16:15:34.295226: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095159999 Hz
2020-05-10 16:15:34.295399: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558d45591700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 16:15:34.295413: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f7b323dcd30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4250 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.3645 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7b2983e6a0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4673 - accuracy: 0.5130 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5695 - accuracy: 0.5063
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6475 - accuracy: 0.5013
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7188 - accuracy: 0.4966
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7331 - accuracy: 0.4957
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7652 - accuracy: 0.4936
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7663 - accuracy: 0.4935
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7263 - accuracy: 0.4961
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7096 - accuracy: 0.4972
11000/25000 [============>.................] - ETA: 3s - loss: 7.7307 - accuracy: 0.4958
12000/25000 [=============>................] - ETA: 3s - loss: 7.7152 - accuracy: 0.4968
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7185 - accuracy: 0.4966
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6995 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.7014 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6892 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6777 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6650 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 7s 273us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f7af8c46278> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:19:03, 9.85kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:15:16, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:07:53, 19.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:29:58, 28.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:56:02, 40.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.13M/862M [00:01<4:08:01, 57.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:52:50, 81.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.8M/862M [00:01<2:00:38, 117kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.4M/862M [00:01<1:24:04, 167kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:01<58:42, 238kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.5M/862M [00:01<40:58, 339kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:02<28:38, 483kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.0M/862M [00:02<20:06, 685kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:02<14:03, 972kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<09:51, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:02<06:59, 1.93MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:11, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<06:14, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<08:01, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:32, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<04:48, 2.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<09:04, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<08:04, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:07<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:08<06:50, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<07:46, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<06:11, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<04:29, 2.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<18:19, 722kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<13:57, 947kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<10:05, 1.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<10:09, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<08:26, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<06:13, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<07:27, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<07:53, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<06:07, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<04:24, 2.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<29:28, 442kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<21:57, 593kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<15:40, 829kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<13:58, 926kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<12:25, 1.04MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<09:21, 1.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<08:36, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:20, 1.75MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<05:27, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<06:48, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<06:05, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:22<04:35, 2.78MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<06:11, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<07:03, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<05:29, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:00, 3.17MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:09, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:43, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:43, 2.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:55, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:24, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:49, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:12, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:07:08, 17.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<8:29:49, 24.6kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<5:56:27, 35.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<4:11:45, 49.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<2:57:31, 70.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<2:04:20, 100kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:29:28, 139kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:05:24, 190kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<46:26, 267kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<32:31, 379kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<35:57, 343kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<26:32, 464kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<18:50, 653kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<15:46, 777kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<13:47, 889kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<10:15, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<07:17, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<16:29, 739kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:53, 945kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<09:17, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:06, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:04, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:56, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:02, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:52, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:51, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:11, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:59, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:00, 2.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<13:12, 902kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:33, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:42, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:57, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:13, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:24, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:36, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<10:21, 1.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:33, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<06:18, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:57, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:28, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:53, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<04:14, 2.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<14:03, 828kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<11:07, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<08:03, 1.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<08:07, 1.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<08:15, 1.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:19, 1.83MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:38, 2.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:30, 1.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:51, 1.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:22, 2.62MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<05:32, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:26, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:08, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:43, 3.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<14:44, 771kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<11:34, 981kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<08:21, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:17, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:19, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:20, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:36, 2.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:07, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:13, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:36, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<03:21, 3.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<44:19, 251kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<33:29, 333kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<23:56, 465kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<17:00, 653kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<14:29, 765kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<11:22, 974kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<08:12, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:08, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:55, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<05:09, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:58, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:36, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<03:47, 2.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<15:49, 687kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<12:16, 884kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<08:53, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:32, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:12, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:17, 2.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:01, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:36, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:13, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<03:46, 2.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<14:01, 760kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<10:59, 969kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<07:58, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:53, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:52, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:05, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<04:23, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<15:48, 665kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<12:12, 861kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<08:49, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:26, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:14, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:16, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:31, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:12, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:12, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:35, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:26, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:58, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:43, 2.77MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<04:51, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:30, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:25, 2.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:38, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:30, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:09, 3.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:26, 1.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:36, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:09, 2.43MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:07, 1.96MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:49, 1.72MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:37, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<03:21, 2.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<12:57, 769kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<10:10, 979kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:22, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:18, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<07:18, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:39, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<04:04, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<14:45, 666kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<11:25, 860kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<08:12, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:53, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:37, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:48, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:50<04:09, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<12:53, 752kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<10:02, 964kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:16, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:13, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:14, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:01, 2.38MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:30, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:56, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<03:40, 2.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<02:40, 3.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<1:01:59, 153kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<45:28, 208kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<32:16, 293kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<22:38, 417kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<20:05, 468kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<15:05, 623kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<10:48, 868kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<09:33, 977kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:44, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:37, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:57, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:10, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:51, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:46, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:23, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:16, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:05, 2.96MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<11:56, 765kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<09:24, 969kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:47, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<04:50, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<1:34:15, 96.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<1:08:00, 133kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<47:57, 189kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<33:39, 268kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<25:59, 346kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<19:10, 469kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<13:36, 658kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<11:26, 780kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<10:05, 884kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<07:29, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<05:19, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<10:21, 855kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<08:04, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<06:01, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<04:19, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<10:31, 834kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<09:19, 942kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:56, 1.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<04:56, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<10:20, 843kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<08:11, 1.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:56, 1.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:00, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<06:09, 1.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:47, 1.80MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:27, 2.49MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<12:46, 672kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<09:50, 871kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<07:04, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:50, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<05:42, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:12, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:51, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<05:18, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:10, 2.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:00, 2.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<09:55, 843kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<07:52, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:41, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:46, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:58, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:40, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:22, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:48, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:43, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:43, 3.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:25, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:41, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:27, 2.35MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:13, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:52, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<02:53, 2.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:45, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:25, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:32, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:34, 3.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<11:25, 696kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<08:52, 896kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<06:24, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:11, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:05, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:41, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:22, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<11:49, 661kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<09:08, 854kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:35, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:17, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:15, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:51, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:21, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:45, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:45, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:42, 2.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<09:05, 837kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:12, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:12, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:44, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<29:10, 258kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<22:00, 342kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<15:43, 478kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<11:06, 675kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<09:58, 748kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<07:48, 956kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<05:37, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<16:42, 443kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<13:16, 557kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<09:37, 767kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<06:46, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<11:16, 650kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<08:43, 840kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<06:15, 1.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:55, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<05:40, 1.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:17, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:05, 2.33MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:27, 1.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:37, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:23, 2.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:53, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:14, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:16, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:23, 2.96MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:21, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:42, 1.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:48, 2.51MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<02:02, 3.42MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<06:51, 1.02MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:35, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:03, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:21, 1.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:50, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:52, 2.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:56, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:04, 2.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:15, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:52, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:27, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:36, 2.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:16, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:47, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:58, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:10, 3.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:18, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:27, 2.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:11, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:55, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:12, 2.96MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:02, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:48, 2.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:06, 3.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:55, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:29, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:47, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:01, 3.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<08:13, 773kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:26, 987kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:38, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:39, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:39, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:34, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<09:06, 683kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:03, 880kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<05:06, 1.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:53, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:06, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:00, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:24, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:54, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:06, 2.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:38, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:13, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:23, 2.51MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:57, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:44, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:03, 2.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:41, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:10, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:33, 2.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<01:50, 3.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:30, 774kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<05:53, 984kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:14, 1.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:12, 1.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:12, 1.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:13, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:18, 2.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<07:11, 789kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:37, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:02, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<04:04, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:00, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:02, 1.84MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:12, 2.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<03:34, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:06, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:18, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:47, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:10, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:31, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:48, 2.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<05:18, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<04:20, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:10, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:20, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:31, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:42, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:57, 2.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<04:02, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:24, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:29, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:52, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:06, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:27, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:46, 2.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<12:40, 403kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<09:24, 543kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<06:41, 760kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<05:46, 872kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<05:09, 976kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:50, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:44, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:36, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:46, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:44, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:59, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:36, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:55, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:25, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:50, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:15, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:37, 2.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<11:06, 429kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<08:16, 575kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:52, 805kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:08, 914kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:35, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:24, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:26, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:36, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:02, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:13, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:32, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:44, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:09, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:32, 2.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<10:40, 420kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<07:57, 563kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<05:39, 789kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:52, 906kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<03:54, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:49, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:54, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:57, 1.47MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:15, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:37, 2.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:13, 1.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:43, 1.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:59, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:19, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:33, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:01, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:27, 2.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<04:56, 838kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:53, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:48, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:53, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:53, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:14, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:35, 2.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<12:38, 317kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<09:16, 431kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<06:33, 607kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<05:23, 730kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:38, 846kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:26, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:25, 1.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<05:18, 727kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<04:08, 932kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:58, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:53, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:51, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:12, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:34, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<05:04, 734kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:57, 939kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:51, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:47, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:21, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:43, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:58, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:42, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:17, 2.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:40, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:55, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:29, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:06, 3.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:55, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:43, 1.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:16, 2.67MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:37, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:30, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:08, 2.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:30, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:25, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:04, 3.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:26, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:41, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<00:57, 3.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:53, 1.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:22, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:43, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:52, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:39, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:13, 2.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:30, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:43, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:22, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<00:59, 3.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<04:16, 692kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:19, 891kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:22, 1.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:16, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:13, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:42, 1.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:13, 2.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:12, 672kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:15, 866kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<02:19, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:12, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:09, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:39, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:10, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:41, 727kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:53, 930kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<02:04, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:00, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:41, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:14, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:24, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:12, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:52, 2.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:28, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:18, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:58, 2.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:11, 2.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:22, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:05, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:46, 3.03MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<03:02, 767kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:23, 977kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:43, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:40, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:40, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:17, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:55, 2.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:18, 666kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:33, 860kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:49, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:42, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:38, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:14, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:52, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:11, 940kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:45, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:15, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:18, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:21, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:03, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:44, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:50, 675kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:11, 876kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:33, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:29, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:35, 714kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:49, 996kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:37, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:19, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:57, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:02, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:06, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:51, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:36, 2.73MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<02:11, 752kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:42, 957kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:13, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:10, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:57, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:42, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:49, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:54, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:42, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:30, 2.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:53, 761kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:28, 970kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:03, 1.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:00, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:51, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:37, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:42, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:37, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:28, 2.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:35, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:32, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:22, 3.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:40, 694kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:17, 897kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:54, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:51, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:37, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:26, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<02:45, 370kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<02:01, 499kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:25, 699kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:09, 820kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:01, 927kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:45, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:30, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:11, 737kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:55, 942kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:39, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:36, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:36, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:18, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:00, 736kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:47, 939kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:33, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:30, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:25, 1.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:19, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:21, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:18, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:16, 1.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:11, 2.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:13, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 2.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:10, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 3.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<01:03, 310kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:45, 422kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:30, 593kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:21, 715kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:18, 832kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:13, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:07, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:14, 790kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:10, 1.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.69MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.93MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 863/400000 [00:00<00:46, 8629.62it/s]  0%|          | 1750/400000 [00:00<00:45, 8697.52it/s]  1%|          | 2602/400000 [00:00<00:45, 8641.79it/s]  1%|          | 3436/400000 [00:00<00:46, 8547.20it/s]  1%|          | 4303/400000 [00:00<00:46, 8581.07it/s]  1%|▏         | 5177/400000 [00:00<00:45, 8627.11it/s]  2%|▏         | 6049/400000 [00:00<00:45, 8652.55it/s]  2%|▏         | 6934/400000 [00:00<00:45, 8708.64it/s]  2%|▏         | 7808/400000 [00:00<00:44, 8715.65it/s]  2%|▏         | 8702/400000 [00:01<00:44, 8781.18it/s]  2%|▏         | 9592/400000 [00:01<00:44, 8815.37it/s]  3%|▎         | 10478/400000 [00:01<00:44, 8826.42it/s]  3%|▎         | 11354/400000 [00:01<00:44, 8804.42it/s]  3%|▎         | 12229/400000 [00:01<00:44, 8785.71it/s]  3%|▎         | 13104/400000 [00:01<00:44, 8772.19it/s]  3%|▎         | 13978/400000 [00:01<00:44, 8756.49it/s]  4%|▎         | 14851/400000 [00:01<00:44, 8578.31it/s]  4%|▍         | 15749/400000 [00:01<00:44, 8694.53it/s]  4%|▍         | 16622/400000 [00:01<00:44, 8704.59it/s]  4%|▍         | 17564/400000 [00:02<00:42, 8907.14it/s]  5%|▍         | 18502/400000 [00:02<00:42, 9040.97it/s]  5%|▍         | 19427/400000 [00:02<00:41, 9101.76it/s]  5%|▌         | 20339/400000 [00:02<00:41, 9058.74it/s]  5%|▌         | 21246/400000 [00:02<00:41, 9045.79it/s]  6%|▌         | 22201/400000 [00:02<00:41, 9189.96it/s]  6%|▌         | 23121/400000 [00:02<00:41, 9025.75it/s]  6%|▌         | 24025/400000 [00:02<00:41, 8952.64it/s]  6%|▌         | 24922/400000 [00:02<00:42, 8916.48it/s]  6%|▋         | 25828/400000 [00:02<00:41, 8958.69it/s]  7%|▋         | 26725/400000 [00:03<00:41, 8924.66it/s]  7%|▋         | 27631/400000 [00:03<00:41, 8962.44it/s]  7%|▋         | 28528/400000 [00:03<00:41, 8886.42it/s]  7%|▋         | 29418/400000 [00:03<00:41, 8843.49it/s]  8%|▊         | 30303/400000 [00:03<00:41, 8811.12it/s]  8%|▊         | 31185/400000 [00:03<00:42, 8764.80it/s]  8%|▊         | 32095/400000 [00:03<00:41, 8861.37it/s]  8%|▊         | 32982/400000 [00:03<00:41, 8807.41it/s]  8%|▊         | 33888/400000 [00:03<00:41, 8881.65it/s]  9%|▊         | 34833/400000 [00:03<00:40, 9044.78it/s]  9%|▉         | 35739/400000 [00:04<00:40, 9040.91it/s]  9%|▉         | 36671/400000 [00:04<00:39, 9122.82it/s]  9%|▉         | 37637/400000 [00:04<00:39, 9275.73it/s] 10%|▉         | 38566/400000 [00:04<00:39, 9181.02it/s] 10%|▉         | 39486/400000 [00:04<00:39, 9032.65it/s] 10%|█         | 40439/400000 [00:04<00:39, 9176.14it/s] 10%|█         | 41366/400000 [00:04<00:38, 9203.89it/s] 11%|█         | 42311/400000 [00:04<00:38, 9274.68it/s] 11%|█         | 43254/400000 [00:04<00:38, 9320.17it/s] 11%|█         | 44209/400000 [00:04<00:37, 9387.09it/s] 11%|█▏        | 45149/400000 [00:05<00:38, 9326.24it/s] 12%|█▏        | 46083/400000 [00:05<00:38, 9254.77it/s] 12%|█▏        | 47044/400000 [00:05<00:37, 9357.88it/s] 12%|█▏        | 47981/400000 [00:05<00:37, 9284.78it/s] 12%|█▏        | 48911/400000 [00:05<00:37, 9281.65it/s] 12%|█▏        | 49844/400000 [00:05<00:37, 9294.13it/s] 13%|█▎        | 50820/400000 [00:05<00:37, 9427.11it/s] 13%|█▎        | 51764/400000 [00:05<00:36, 9421.52it/s] 13%|█▎        | 52714/400000 [00:05<00:36, 9443.47it/s] 13%|█▎        | 53659/400000 [00:05<00:37, 9327.91it/s] 14%|█▎        | 54593/400000 [00:06<00:37, 9317.26it/s] 14%|█▍        | 55526/400000 [00:06<00:37, 9272.19it/s] 14%|█▍        | 56454/400000 [00:06<00:37, 9189.68it/s] 14%|█▍        | 57374/400000 [00:06<00:37, 9094.80it/s] 15%|█▍        | 58284/400000 [00:06<00:37, 9005.82it/s] 15%|█▍        | 59186/400000 [00:06<00:38, 8948.16it/s] 15%|█▌        | 60082/400000 [00:06<00:38, 8939.77it/s] 15%|█▌        | 60977/400000 [00:06<00:38, 8878.80it/s] 15%|█▌        | 61866/400000 [00:06<00:38, 8795.75it/s] 16%|█▌        | 62786/400000 [00:06<00:37, 8911.76it/s] 16%|█▌        | 63701/400000 [00:07<00:37, 8980.95it/s] 16%|█▌        | 64622/400000 [00:07<00:37, 9046.79it/s] 16%|█▋        | 65528/400000 [00:07<00:37, 9034.00it/s] 17%|█▋        | 66432/400000 [00:07<00:37, 8921.33it/s] 17%|█▋        | 67325/400000 [00:07<00:37, 8889.52it/s] 17%|█▋        | 68263/400000 [00:07<00:36, 9029.25it/s] 17%|█▋        | 69167/400000 [00:07<00:36, 9026.87it/s] 18%|█▊        | 70081/400000 [00:07<00:36, 9059.46it/s] 18%|█▊        | 70997/400000 [00:07<00:36, 9088.30it/s] 18%|█▊        | 71907/400000 [00:07<00:36, 9043.43it/s] 18%|█▊        | 72812/400000 [00:08<00:36, 9010.47it/s] 18%|█▊        | 73714/400000 [00:08<00:36, 8961.98it/s] 19%|█▊        | 74611/400000 [00:08<00:36, 8942.87it/s] 19%|█▉        | 75506/400000 [00:08<00:36, 8858.09it/s] 19%|█▉        | 76393/400000 [00:08<00:37, 8561.08it/s] 19%|█▉        | 77280/400000 [00:08<00:37, 8651.32it/s] 20%|█▉        | 78148/400000 [00:08<00:37, 8655.72it/s] 20%|█▉        | 79040/400000 [00:08<00:36, 8733.01it/s] 20%|█▉        | 79933/400000 [00:08<00:36, 8790.55it/s] 20%|██        | 80813/400000 [00:09<00:36, 8740.29it/s] 20%|██        | 81710/400000 [00:09<00:36, 8805.05it/s] 21%|██        | 82611/400000 [00:09<00:35, 8862.75it/s] 21%|██        | 83500/400000 [00:09<00:35, 8868.86it/s] 21%|██        | 84388/400000 [00:09<00:35, 8786.73it/s] 21%|██▏       | 85268/400000 [00:09<00:35, 8759.04it/s] 22%|██▏       | 86145/400000 [00:09<00:35, 8748.43it/s] 22%|██▏       | 87021/400000 [00:09<00:35, 8749.84it/s] 22%|██▏       | 87897/400000 [00:09<00:35, 8702.99it/s] 22%|██▏       | 88768/400000 [00:09<00:35, 8692.68it/s] 22%|██▏       | 89638/400000 [00:10<00:36, 8612.83it/s] 23%|██▎       | 90514/400000 [00:10<00:35, 8655.19it/s] 23%|██▎       | 91391/400000 [00:10<00:35, 8687.86it/s] 23%|██▎       | 92261/400000 [00:10<00:35, 8688.67it/s] 23%|██▎       | 93139/400000 [00:10<00:35, 8715.60it/s] 24%|██▎       | 94016/400000 [00:10<00:35, 8729.54it/s] 24%|██▎       | 94901/400000 [00:10<00:34, 8763.80it/s] 24%|██▍       | 95781/400000 [00:10<00:34, 8771.78it/s] 24%|██▍       | 96659/400000 [00:10<00:34, 8729.78it/s] 24%|██▍       | 97545/400000 [00:10<00:34, 8767.51it/s] 25%|██▍       | 98437/400000 [00:11<00:34, 8812.08it/s] 25%|██▍       | 99355/400000 [00:11<00:33, 8918.66it/s] 25%|██▌       | 100250/400000 [00:11<00:33, 8927.45it/s] 25%|██▌       | 101147/400000 [00:11<00:33, 8937.30it/s] 26%|██▌       | 102041/400000 [00:11<00:33, 8904.22it/s] 26%|██▌       | 102932/400000 [00:11<00:33, 8860.94it/s] 26%|██▌       | 103819/400000 [00:11<00:33, 8816.29it/s] 26%|██▌       | 104741/400000 [00:11<00:33, 8933.20it/s] 26%|██▋       | 105635/400000 [00:11<00:33, 8881.03it/s] 27%|██▋       | 106564/400000 [00:11<00:32, 8997.15it/s] 27%|██▋       | 107477/400000 [00:12<00:32, 9036.54it/s] 27%|██▋       | 108382/400000 [00:12<00:32, 8978.35it/s] 27%|██▋       | 109281/400000 [00:12<00:32, 8935.37it/s] 28%|██▊       | 110175/400000 [00:12<00:32, 8854.06it/s] 28%|██▊       | 111061/400000 [00:12<00:32, 8832.09it/s] 28%|██▊       | 111961/400000 [00:12<00:32, 8880.59it/s] 28%|██▊       | 112853/400000 [00:12<00:32, 8890.03it/s] 28%|██▊       | 113772/400000 [00:12<00:31, 8975.39it/s] 29%|██▊       | 114670/400000 [00:12<00:31, 8958.45it/s] 29%|██▉       | 115572/400000 [00:12<00:31, 8974.18it/s] 29%|██▉       | 116470/400000 [00:13<00:31, 8931.96it/s] 29%|██▉       | 117364/400000 [00:13<00:31, 8893.75it/s] 30%|██▉       | 118254/400000 [00:13<00:31, 8866.58it/s] 30%|██▉       | 119141/400000 [00:13<00:31, 8832.60it/s] 30%|███       | 120025/400000 [00:13<00:31, 8807.69it/s] 30%|███       | 120906/400000 [00:13<00:31, 8784.05it/s] 30%|███       | 121802/400000 [00:13<00:31, 8833.83it/s] 31%|███       | 122686/400000 [00:13<00:31, 8821.41it/s] 31%|███       | 123575/400000 [00:13<00:31, 8841.13it/s] 31%|███       | 124465/400000 [00:13<00:31, 8858.34it/s] 31%|███▏      | 125358/400000 [00:14<00:30, 8877.66it/s] 32%|███▏      | 126254/400000 [00:14<00:30, 8900.93it/s] 32%|███▏      | 127145/400000 [00:14<00:30, 8895.08it/s] 32%|███▏      | 128038/400000 [00:14<00:30, 8903.85it/s] 32%|███▏      | 128929/400000 [00:14<00:30, 8834.00it/s] 32%|███▏      | 129813/400000 [00:14<00:30, 8811.97it/s] 33%|███▎      | 130695/400000 [00:14<00:30, 8783.93it/s] 33%|███▎      | 131574/400000 [00:14<00:30, 8780.45it/s] 33%|███▎      | 132453/400000 [00:14<00:30, 8776.13it/s] 33%|███▎      | 133333/400000 [00:14<00:30, 8782.38it/s] 34%|███▎      | 134225/400000 [00:15<00:30, 8823.05it/s] 34%|███▍      | 135118/400000 [00:15<00:29, 8852.78it/s] 34%|███▍      | 136004/400000 [00:15<00:29, 8849.20it/s] 34%|███▍      | 136889/400000 [00:15<00:29, 8811.72it/s] 34%|███▍      | 137771/400000 [00:15<00:29, 8784.45it/s] 35%|███▍      | 138651/400000 [00:15<00:29, 8787.68it/s] 35%|███▍      | 139530/400000 [00:15<00:29, 8778.55it/s] 35%|███▌      | 140419/400000 [00:15<00:29, 8810.04it/s] 35%|███▌      | 141316/400000 [00:15<00:29, 8857.16it/s] 36%|███▌      | 142207/400000 [00:15<00:29, 8870.54it/s] 36%|███▌      | 143117/400000 [00:16<00:28, 8937.81it/s] 36%|███▌      | 144040/400000 [00:16<00:28, 9023.04it/s] 36%|███▌      | 144970/400000 [00:16<00:28, 9102.71it/s] 36%|███▋      | 145882/400000 [00:16<00:27, 9105.13it/s] 37%|███▋      | 146793/400000 [00:16<00:27, 9058.53it/s] 37%|███▋      | 147702/400000 [00:16<00:27, 9065.70it/s] 37%|███▋      | 148626/400000 [00:16<00:27, 9116.83it/s] 37%|███▋      | 149538/400000 [00:16<00:28, 8865.62it/s] 38%|███▊      | 150430/400000 [00:16<00:28, 8879.72it/s] 38%|███▊      | 151349/400000 [00:16<00:27, 8968.27it/s] 38%|███▊      | 152250/400000 [00:17<00:27, 8978.73it/s] 38%|███▊      | 153149/400000 [00:17<00:27, 8924.00it/s] 39%|███▊      | 154042/400000 [00:17<00:27, 8891.35it/s] 39%|███▊      | 154932/400000 [00:17<00:27, 8850.07it/s] 39%|███▉      | 155818/400000 [00:17<00:27, 8778.71it/s] 39%|███▉      | 156697/400000 [00:17<00:28, 8663.71it/s] 39%|███▉      | 157581/400000 [00:17<00:27, 8712.97it/s] 40%|███▉      | 158477/400000 [00:17<00:27, 8784.40it/s] 40%|███▉      | 159376/400000 [00:17<00:27, 8844.46it/s] 40%|████      | 160261/400000 [00:17<00:27, 8840.43it/s] 40%|████      | 161146/400000 [00:18<00:27, 8836.11it/s] 41%|████      | 162034/400000 [00:18<00:26, 8848.55it/s] 41%|████      | 162921/400000 [00:18<00:26, 8852.69it/s] 41%|████      | 163807/400000 [00:18<00:26, 8831.90it/s] 41%|████      | 164691/400000 [00:18<00:26, 8798.49it/s] 41%|████▏     | 165571/400000 [00:18<00:26, 8781.84it/s] 42%|████▏     | 166450/400000 [00:18<00:26, 8751.64it/s] 42%|████▏     | 167326/400000 [00:18<00:26, 8693.08it/s] 42%|████▏     | 168196/400000 [00:18<00:26, 8691.79it/s] 42%|████▏     | 169115/400000 [00:18<00:26, 8833.49it/s] 43%|████▎     | 170005/400000 [00:19<00:25, 8851.60it/s] 43%|████▎     | 170905/400000 [00:19<00:25, 8891.44it/s] 43%|████▎     | 171795/400000 [00:19<00:25, 8869.60it/s] 43%|████▎     | 172683/400000 [00:19<00:25, 8841.39it/s] 43%|████▎     | 173568/400000 [00:19<00:25, 8777.66it/s] 44%|████▎     | 174447/400000 [00:19<00:25, 8779.00it/s] 44%|████▍     | 175326/400000 [00:19<00:25, 8774.83it/s] 44%|████▍     | 176204/400000 [00:19<00:25, 8773.20it/s] 44%|████▍     | 177082/400000 [00:19<00:25, 8770.96it/s] 44%|████▍     | 177987/400000 [00:19<00:25, 8851.76it/s] 45%|████▍     | 178891/400000 [00:20<00:24, 8907.07it/s] 45%|████▍     | 179805/400000 [00:20<00:24, 8973.33it/s] 45%|████▌     | 180718/400000 [00:20<00:24, 9019.45it/s] 45%|████▌     | 181621/400000 [00:20<00:24, 8998.68it/s] 46%|████▌     | 182522/400000 [00:20<00:24, 8911.60it/s] 46%|████▌     | 183414/400000 [00:20<00:24, 8863.26it/s] 46%|████▌     | 184301/400000 [00:20<00:24, 8851.20it/s] 46%|████▋     | 185191/400000 [00:20<00:24, 8865.68it/s] 47%|████▋     | 186078/400000 [00:20<00:24, 8866.56it/s] 47%|████▋     | 186965/400000 [00:20<00:24, 8812.90it/s] 47%|████▋     | 187847/400000 [00:21<00:24, 8800.68it/s] 47%|████▋     | 188729/400000 [00:21<00:23, 8804.43it/s] 47%|████▋     | 189610/400000 [00:21<00:23, 8798.82it/s] 48%|████▊     | 190490/400000 [00:21<00:23, 8772.35it/s] 48%|████▊     | 191368/400000 [00:21<00:23, 8738.72it/s] 48%|████▊     | 192242/400000 [00:21<00:23, 8683.73it/s] 48%|████▊     | 193111/400000 [00:21<00:23, 8636.90it/s] 48%|████▊     | 193979/400000 [00:21<00:23, 8647.71it/s] 49%|████▊     | 194855/400000 [00:21<00:23, 8680.21it/s] 49%|████▉     | 195746/400000 [00:22<00:23, 8745.34it/s] 49%|████▉     | 196621/400000 [00:22<00:23, 8614.96it/s] 49%|████▉     | 197488/400000 [00:22<00:23, 8629.92it/s] 50%|████▉     | 198366/400000 [00:22<00:23, 8672.25it/s] 50%|████▉     | 199240/400000 [00:22<00:23, 8689.75it/s] 50%|█████     | 200110/400000 [00:22<00:22, 8691.38it/s] 50%|█████     | 200983/400000 [00:22<00:22, 8702.38it/s] 50%|█████     | 201858/400000 [00:22<00:22, 8715.96it/s] 51%|█████     | 202736/400000 [00:22<00:22, 8732.88it/s] 51%|█████     | 203619/400000 [00:22<00:22, 8759.67it/s] 51%|█████     | 204496/400000 [00:23<00:22, 8744.46it/s] 51%|█████▏    | 205376/400000 [00:23<00:22, 8758.75it/s] 52%|█████▏    | 206254/400000 [00:23<00:22, 8764.59it/s] 52%|█████▏    | 207132/400000 [00:23<00:22, 8766.64it/s] 52%|█████▏    | 208009/400000 [00:23<00:21, 8767.17it/s] 52%|█████▏    | 208886/400000 [00:23<00:21, 8740.08it/s] 52%|█████▏    | 209761/400000 [00:23<00:21, 8704.26it/s] 53%|█████▎    | 210632/400000 [00:23<00:22, 8605.37it/s] 53%|█████▎    | 211493/400000 [00:23<00:22, 8300.83it/s] 53%|█████▎    | 212361/400000 [00:23<00:22, 8408.53it/s] 53%|█████▎    | 213238/400000 [00:24<00:21, 8511.71it/s] 54%|█████▎    | 214112/400000 [00:24<00:21, 8577.49it/s] 54%|█████▎    | 214986/400000 [00:24<00:21, 8623.11it/s] 54%|█████▍    | 215861/400000 [00:24<00:21, 8660.42it/s] 54%|█████▍    | 216728/400000 [00:24<00:21, 8655.60it/s] 54%|█████▍    | 217595/400000 [00:24<00:21, 8652.35it/s] 55%|█████▍    | 218470/400000 [00:24<00:20, 8679.39it/s] 55%|█████▍    | 219347/400000 [00:24<00:20, 8704.79it/s] 55%|█████▌    | 220222/400000 [00:24<00:20, 8718.10it/s] 55%|█████▌    | 221102/400000 [00:24<00:20, 8740.82it/s] 55%|█████▌    | 221977/400000 [00:25<00:20, 8737.84it/s] 56%|█████▌    | 222883/400000 [00:25<00:20, 8829.77it/s] 56%|█████▌    | 223786/400000 [00:25<00:19, 8888.16it/s] 56%|█████▌    | 224695/400000 [00:25<00:19, 8947.04it/s] 56%|█████▋    | 225599/400000 [00:25<00:19, 8972.54it/s] 57%|█████▋    | 226497/400000 [00:25<00:19, 8887.78it/s] 57%|█████▋    | 227387/400000 [00:25<00:19, 8829.49it/s] 57%|█████▋    | 228271/400000 [00:25<00:19, 8630.06it/s] 57%|█████▋    | 229208/400000 [00:25<00:19, 8837.82it/s] 58%|█████▊    | 230098/400000 [00:25<00:19, 8855.57it/s] 58%|█████▊    | 230986/400000 [00:26<00:19, 8852.36it/s] 58%|█████▊    | 231873/400000 [00:26<00:19, 8825.09it/s] 58%|█████▊    | 232757/400000 [00:26<00:19, 8792.56it/s] 58%|█████▊    | 233637/400000 [00:26<00:18, 8757.04it/s] 59%|█████▊    | 234514/400000 [00:26<00:18, 8737.16it/s] 59%|█████▉    | 235389/400000 [00:26<00:18, 8704.35it/s] 59%|█████▉    | 236260/400000 [00:26<00:18, 8696.00it/s] 59%|█████▉    | 237132/400000 [00:26<00:18, 8702.75it/s] 60%|█████▉    | 238007/400000 [00:26<00:18, 8716.19it/s] 60%|█████▉    | 238879/400000 [00:26<00:18, 8717.07it/s] 60%|█████▉    | 239751/400000 [00:27<00:18, 8700.52it/s] 60%|██████    | 240655/400000 [00:27<00:18, 8797.40it/s] 60%|██████    | 241536/400000 [00:27<00:18, 8785.08it/s] 61%|██████    | 242463/400000 [00:27<00:17, 8922.42it/s] 61%|██████    | 243375/400000 [00:27<00:17, 8978.08it/s] 61%|██████    | 244274/400000 [00:27<00:17, 8893.33it/s] 61%|██████▏   | 245164/400000 [00:27<00:17, 8811.99it/s] 62%|██████▏   | 246046/400000 [00:27<00:17, 8789.87it/s] 62%|██████▏   | 246974/400000 [00:27<00:17, 8930.97it/s] 62%|██████▏   | 247894/400000 [00:27<00:16, 9009.15it/s] 62%|██████▏   | 248799/400000 [00:28<00:16, 9020.13it/s] 62%|██████▏   | 249773/400000 [00:28<00:16, 9222.96it/s] 63%|██████▎   | 250721/400000 [00:28<00:16, 9295.86it/s] 63%|██████▎   | 251652/400000 [00:28<00:16, 9233.15it/s] 63%|██████▎   | 252577/400000 [00:28<00:16, 9101.09it/s] 63%|██████▎   | 253489/400000 [00:28<00:16, 8963.27it/s] 64%|██████▎   | 254387/400000 [00:28<00:16, 8942.73it/s] 64%|██████▍   | 255311/400000 [00:28<00:16, 9028.17it/s] 64%|██████▍   | 256215/400000 [00:28<00:15, 9022.32it/s] 64%|██████▍   | 257118/400000 [00:28<00:15, 8966.74it/s] 65%|██████▍   | 258016/400000 [00:29<00:15, 8891.25it/s] 65%|██████▍   | 258923/400000 [00:29<00:15, 8943.14it/s] 65%|██████▍   | 259838/400000 [00:29<00:15, 9001.95it/s] 65%|██████▌   | 260757/400000 [00:29<00:15, 9055.72it/s] 65%|██████▌   | 261663/400000 [00:29<00:15, 8852.09it/s] 66%|██████▌   | 262550/400000 [00:29<00:15, 8775.24it/s] 66%|██████▌   | 263429/400000 [00:29<00:15, 8763.61it/s] 66%|██████▌   | 264307/400000 [00:29<00:15, 8734.79it/s] 66%|██████▋   | 265228/400000 [00:29<00:15, 8870.90it/s] 67%|██████▋   | 266130/400000 [00:29<00:15, 8912.37it/s] 67%|██████▋   | 267022/400000 [00:30<00:15, 8830.78it/s] 67%|██████▋   | 267909/400000 [00:30<00:14, 8840.69it/s] 67%|██████▋   | 268794/400000 [00:30<00:15, 8685.72it/s] 67%|██████▋   | 269678/400000 [00:30<00:14, 8729.57it/s] 68%|██████▊   | 270552/400000 [00:30<00:14, 8731.97it/s] 68%|██████▊   | 271426/400000 [00:30<00:15, 8499.81it/s] 68%|██████▊   | 272294/400000 [00:30<00:14, 8552.43it/s] 68%|██████▊   | 273169/400000 [00:30<00:14, 8608.21it/s] 69%|██████▊   | 274031/400000 [00:30<00:15, 8354.96it/s] 69%|██████▊   | 274946/400000 [00:31<00:14, 8576.77it/s] 69%|██████▉   | 275877/400000 [00:31<00:14, 8784.23it/s] 69%|██████▉   | 276818/400000 [00:31<00:13, 8961.64it/s] 69%|██████▉   | 277737/400000 [00:31<00:13, 9026.91it/s] 70%|██████▉   | 278661/400000 [00:31<00:13, 9089.59it/s] 70%|██████▉   | 279598/400000 [00:31<00:13, 9169.60it/s] 70%|███████   | 280517/400000 [00:31<00:13, 9038.64it/s] 70%|███████   | 281423/400000 [00:31<00:13, 9016.91it/s] 71%|███████   | 282344/400000 [00:31<00:12, 9072.33it/s] 71%|███████   | 283258/400000 [00:31<00:12, 9091.81it/s] 71%|███████   | 284168/400000 [00:32<00:12, 9075.93it/s] 71%|███████▏  | 285076/400000 [00:32<00:12, 9065.46it/s] 72%|███████▏  | 286012/400000 [00:32<00:12, 9151.77it/s] 72%|███████▏  | 286928/400000 [00:32<00:12, 9075.96it/s] 72%|███████▏  | 287837/400000 [00:32<00:12, 9032.34it/s] 72%|███████▏  | 288744/400000 [00:32<00:12, 9042.31it/s] 72%|███████▏  | 289649/400000 [00:32<00:12, 8928.55it/s] 73%|███████▎  | 290543/400000 [00:32<00:12, 8884.62it/s] 73%|███████▎  | 291432/400000 [00:32<00:12, 8829.45it/s] 73%|███████▎  | 292435/400000 [00:32<00:11, 9157.93it/s] 73%|███████▎  | 293355/400000 [00:33<00:11, 9105.53it/s] 74%|███████▎  | 294289/400000 [00:33<00:11, 9172.88it/s] 74%|███████▍  | 295209/400000 [00:33<00:11, 9179.86it/s] 74%|███████▍  | 296129/400000 [00:33<00:11, 9168.73it/s] 74%|███████▍  | 297047/400000 [00:33<00:11, 9061.90it/s] 74%|███████▍  | 297955/400000 [00:33<00:11, 8990.85it/s] 75%|███████▍  | 298855/400000 [00:33<00:11, 8887.26it/s] 75%|███████▍  | 299745/400000 [00:33<00:11, 8794.98it/s] 75%|███████▌  | 300626/400000 [00:33<00:11, 8769.75it/s] 75%|███████▌  | 301504/400000 [00:33<00:11, 8748.93it/s] 76%|███████▌  | 302395/400000 [00:34<00:11, 8795.59it/s] 76%|███████▌  | 303287/400000 [00:34<00:10, 8831.78it/s] 76%|███████▌  | 304171/400000 [00:34<00:10, 8819.04it/s] 76%|███████▋  | 305054/400000 [00:34<00:10, 8783.38it/s] 76%|███████▋  | 305933/400000 [00:34<00:10, 8742.90it/s] 77%|███████▋  | 306808/400000 [00:34<00:10, 8739.66it/s] 77%|███████▋  | 307683/400000 [00:34<00:10, 8691.45it/s] 77%|███████▋  | 308553/400000 [00:34<00:10, 8468.91it/s] 77%|███████▋  | 309425/400000 [00:34<00:10, 8540.77it/s] 78%|███████▊  | 310301/400000 [00:34<00:10, 8602.76it/s] 78%|███████▊  | 311172/400000 [00:35<00:10, 8631.84it/s] 78%|███████▊  | 312046/400000 [00:35<00:10, 8663.77it/s] 78%|███████▊  | 313003/400000 [00:35<00:09, 8917.04it/s] 78%|███████▊  | 313916/400000 [00:35<00:09, 8977.30it/s] 79%|███████▊  | 314816/400000 [00:35<00:09, 8980.19it/s] 79%|███████▉  | 315716/400000 [00:35<00:09, 8936.33it/s] 79%|███████▉  | 316611/400000 [00:35<00:09, 8853.56it/s] 79%|███████▉  | 317498/400000 [00:35<00:09, 8814.72it/s] 80%|███████▉  | 318381/400000 [00:35<00:09, 8788.53it/s] 80%|███████▉  | 319262/400000 [00:35<00:09, 8793.88it/s] 80%|████████  | 320181/400000 [00:36<00:08, 8907.69it/s] 80%|████████  | 321078/400000 [00:36<00:08, 8923.93it/s] 80%|████████  | 321981/400000 [00:36<00:08, 8954.67it/s] 81%|████████  | 322877/400000 [00:36<00:08, 8886.33it/s] 81%|████████  | 323766/400000 [00:36<00:08, 8879.46it/s] 81%|████████  | 324675/400000 [00:36<00:08, 8938.68it/s] 81%|████████▏ | 325570/400000 [00:36<00:08, 8857.55it/s] 82%|████████▏ | 326457/400000 [00:36<00:08, 8782.84it/s] 82%|████████▏ | 327336/400000 [00:36<00:08, 8756.03it/s] 82%|████████▏ | 328234/400000 [00:36<00:08, 8820.74it/s] 82%|████████▏ | 329130/400000 [00:37<00:07, 8860.18it/s] 83%|████████▎ | 330017/400000 [00:37<00:07, 8765.38it/s] 83%|████████▎ | 330907/400000 [00:37<00:07, 8803.64it/s] 83%|████████▎ | 331834/400000 [00:37<00:07, 8937.20it/s] 83%|████████▎ | 332757/400000 [00:37<00:07, 9021.54it/s] 83%|████████▎ | 333670/400000 [00:37<00:07, 9052.02it/s] 84%|████████▎ | 334576/400000 [00:37<00:07, 8937.57it/s] 84%|████████▍ | 335471/400000 [00:37<00:07, 8861.87it/s] 84%|████████▍ | 336358/400000 [00:37<00:07, 8818.56it/s] 84%|████████▍ | 337241/400000 [00:38<00:07, 8803.93it/s] 85%|████████▍ | 338122/400000 [00:38<00:07, 8775.37it/s] 85%|████████▍ | 339000/400000 [00:38<00:07, 8639.45it/s] 85%|████████▍ | 339913/400000 [00:38<00:06, 8779.20it/s] 85%|████████▌ | 340792/400000 [00:38<00:06, 8629.43it/s] 85%|████████▌ | 341681/400000 [00:38<00:06, 8703.56it/s] 86%|████████▌ | 342565/400000 [00:38<00:06, 8741.77it/s] 86%|████████▌ | 343440/400000 [00:38<00:06, 8703.09it/s] 86%|████████▌ | 344316/400000 [00:38<00:06, 8719.78it/s] 86%|████████▋ | 345189/400000 [00:38<00:06, 8690.53it/s] 87%|████████▋ | 346066/400000 [00:39<00:06, 8713.69it/s] 87%|████████▋ | 346938/400000 [00:39<00:06, 8702.81it/s] 87%|████████▋ | 347809/400000 [00:39<00:06, 8664.92it/s] 87%|████████▋ | 348676/400000 [00:39<00:05, 8595.79it/s] 87%|████████▋ | 349551/400000 [00:39<00:05, 8639.59it/s] 88%|████████▊ | 350433/400000 [00:39<00:05, 8692.12it/s] 88%|████████▊ | 351311/400000 [00:39<00:05, 8717.98it/s] 88%|████████▊ | 352183/400000 [00:39<00:05, 8702.01it/s] 88%|████████▊ | 353058/400000 [00:39<00:05, 8715.20it/s] 88%|████████▊ | 353930/400000 [00:39<00:05, 8715.97it/s] 89%|████████▊ | 354802/400000 [00:40<00:05, 8703.48it/s] 89%|████████▉ | 355676/400000 [00:40<00:05, 8712.50it/s] 89%|████████▉ | 356548/400000 [00:40<00:05, 8667.93it/s] 89%|████████▉ | 357426/400000 [00:40<00:04, 8700.91it/s] 90%|████████▉ | 358302/400000 [00:40<00:04, 8716.01it/s] 90%|████████▉ | 359186/400000 [00:40<00:04, 8752.57it/s] 90%|█████████ | 360062/400000 [00:40<00:04, 8584.50it/s] 90%|█████████ | 360922/400000 [00:40<00:04, 8350.59it/s] 90%|█████████ | 361794/400000 [00:40<00:04, 8455.49it/s] 91%|█████████ | 362676/400000 [00:40<00:04, 8560.23it/s] 91%|█████████ | 363570/400000 [00:41<00:04, 8669.44it/s] 91%|█████████ | 364462/400000 [00:41<00:04, 8742.69it/s] 91%|█████████▏| 365338/400000 [00:41<00:03, 8703.94it/s] 92%|█████████▏| 366223/400000 [00:41<00:03, 8746.21it/s] 92%|█████████▏| 367133/400000 [00:41<00:03, 8846.83it/s] 92%|█████████▏| 368037/400000 [00:41<00:03, 8903.03it/s] 92%|█████████▏| 368928/400000 [00:41<00:03, 8835.79it/s] 92%|█████████▏| 369813/400000 [00:41<00:03, 8767.15it/s] 93%|█████████▎| 370693/400000 [00:41<00:03, 8776.59it/s] 93%|█████████▎| 371604/400000 [00:41<00:03, 8873.46it/s] 93%|█████████▎| 372520/400000 [00:42<00:03, 8956.60it/s] 93%|█████████▎| 373418/400000 [00:42<00:02, 8962.37it/s] 94%|█████████▎| 374315/400000 [00:42<00:02, 8834.96it/s] 94%|█████████▍| 375200/400000 [00:42<00:02, 8810.38it/s] 94%|█████████▍| 376082/400000 [00:42<00:02, 8662.29it/s] 94%|█████████▍| 376950/400000 [00:42<00:02, 8639.07it/s] 94%|█████████▍| 377833/400000 [00:42<00:02, 8692.63it/s] 95%|█████████▍| 378703/400000 [00:42<00:02, 8679.35it/s] 95%|█████████▍| 379573/400000 [00:42<00:02, 8683.88it/s] 95%|█████████▌| 380444/400000 [00:42<00:02, 8688.96it/s] 95%|█████████▌| 381329/400000 [00:43<00:02, 8735.17it/s] 96%|█████████▌| 382203/400000 [00:43<00:02, 8647.97it/s] 96%|█████████▌| 383069/400000 [00:43<00:01, 8635.32it/s] 96%|█████████▌| 383933/400000 [00:43<00:01, 8591.82it/s] 96%|█████████▌| 384811/400000 [00:43<00:01, 8645.34it/s] 96%|█████████▋| 385686/400000 [00:43<00:01, 8676.29it/s] 97%|█████████▋| 386554/400000 [00:43<00:01, 8674.76it/s] 97%|█████████▋| 387422/400000 [00:43<00:01, 8667.16it/s] 97%|█████████▋| 388295/400000 [00:43<00:01, 8685.36it/s] 97%|█████████▋| 389167/400000 [00:43<00:01, 8695.24it/s] 98%|█████████▊| 390052/400000 [00:44<00:01, 8739.88it/s] 98%|█████████▊| 390945/400000 [00:44<00:01, 8793.21it/s] 98%|█████████▊| 391825/400000 [00:44<00:00, 8745.77it/s] 98%|█████████▊| 392741/400000 [00:44<00:00, 8865.82it/s] 98%|█████████▊| 393662/400000 [00:44<00:00, 8965.16it/s] 99%|█████████▊| 394560/400000 [00:44<00:00, 8911.95it/s] 99%|█████████▉| 395472/400000 [00:44<00:00, 8971.32it/s] 99%|█████████▉| 396370/400000 [00:44<00:00, 8653.81it/s] 99%|█████████▉| 397245/400000 [00:44<00:00, 8682.36it/s]100%|█████████▉| 398136/400000 [00:44<00:00, 8747.79it/s]100%|█████████▉| 399025/400000 [00:45<00:00, 8788.92it/s]100%|█████████▉| 399905/400000 [00:45<00:00, 8782.67it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8847.82it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7aec51b128> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01098030780006376 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011069193532235646 	 Accuracy: 53

  model saves at 53% accuracy 

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f948b1a34a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 18:12:04.433907
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 18:12:04.437476
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 18:12:04.440657
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 18:12:04.443780
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f94834f3400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351452.0625
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 252092.9688
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 146297.3750
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 73150.7188
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 36388.5078
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 20080.1777
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 12544.9775
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 8509.2646
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 6224.6460
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 4797.8374

  #### Inference Need return ypred, ytrue ######################### 
[[-1.93646759e-01  1.01954050e+01  8.89366436e+00  1.19419584e+01
   9.29545593e+00  9.50142193e+00  1.03299742e+01  8.68926430e+00
   9.74385643e+00  8.33214951e+00  9.89113140e+00  1.00545321e+01
   8.26745987e+00  9.57858086e+00  1.09579945e+01  1.09530058e+01
   9.71583652e+00  8.79502964e+00  1.12269325e+01  9.06846237e+00
   8.92819977e+00  9.93928051e+00  1.03549051e+01  1.05131388e+01
   7.52213860e+00  9.79977894e+00  8.81339455e+00  1.05311813e+01
   1.16246300e+01  9.99604702e+00  9.82695198e+00  9.46236229e+00
   9.83745098e+00  9.19813824e+00  1.02667255e+01  1.00955143e+01
   1.01252050e+01  9.55993843e+00  1.04638643e+01  9.17954731e+00
   9.12011147e+00  9.76328754e+00  8.83878899e+00  8.60744381e+00
   1.11541843e+01  9.23060131e+00  1.07361202e+01  9.80771637e+00
   9.19329929e+00  1.01147680e+01  1.01699352e+01  9.21189022e+00
   8.00598621e+00  1.04990015e+01  8.49093819e+00  1.10526314e+01
   9.37251759e+00  1.07622395e+01  1.03845348e+01  8.66845894e+00
   4.70976233e-01 -1.00466096e+00  1.39462900e+00 -1.25656033e+00
  -8.91843200e-01 -2.83104748e-01 -1.24424207e+00  1.29198349e+00
  -1.19670928e+00  9.31050539e-01 -2.51910716e-01 -2.72534966e-01
   2.03892159e+00  2.18978643e-01  7.37596989e-01  1.05645192e+00
   1.13066161e+00 -9.81207490e-01  5.12316704e-01  6.84170663e-01
  -2.30796486e-01 -5.46615571e-02 -1.48994601e+00 -2.95083928e+00
  -7.72720873e-01 -2.41358113e+00  4.73997980e-01 -1.97639048e-01
   5.87231219e-02 -1.59139466e+00 -1.03301251e+00 -3.32936585e-01
   1.09504414e+00 -1.15937936e+00 -1.39710271e+00 -1.29136622e+00
  -3.54338586e-02 -2.06250811e+00  6.03523433e-01 -5.83021343e-01
  -2.81459540e-01 -1.33738816e-02 -2.28816247e+00 -1.15369463e+00
  -7.87937522e-01 -1.24914575e+00 -8.75083983e-01  2.87898207e+00
  -2.98597634e-01  4.12113130e-01 -3.56213897e-01 -7.13189006e-01
   2.41886705e-01 -7.77546763e-01 -8.50820482e-01 -1.11315989e+00
  -6.40592635e-01  3.20292488e-02 -1.01258528e+00  2.01052055e-01
  -1.29009402e+00 -1.40603557e-01  1.45587468e+00  4.32045579e-01
  -4.60260600e-01 -1.69788551e+00  1.34524179e+00 -8.30886483e-01
   6.94244981e-01  1.94127524e+00 -9.08050179e-01 -7.28758037e-01
  -1.23482692e+00  1.58531594e+00  2.12328434e-02  1.53913796e-01
  -7.87670910e-01  5.75394154e-01 -3.41208935e-01  9.12494063e-01
  -6.55556917e-01 -1.59475911e+00  8.66732776e-01  2.70985007e-01
   1.51724005e+00  9.84476924e-01  9.86464500e-01  7.58171916e-01
   5.53988874e-01  5.85630536e-02 -1.22386193e+00  1.39851201e+00
  -6.05421960e-01  1.32650602e+00  3.38391513e-02  1.02497935e-02
  -1.38128591e+00 -2.76637673e-02 -4.39764053e-01  5.45477629e-01
   1.76886308e+00 -8.12039256e-01 -2.26067662e+00 -7.25831866e-01
  -9.53588128e-01  6.56452060e-01 -7.74505317e-01 -2.15272355e+00
   1.66495955e+00 -2.53691375e-01 -4.20971394e-01  1.03681076e+00
   8.32211137e-01 -1.60598171e+00  1.91205740e+00  3.30309451e-01
   1.13704121e+00 -6.62472785e-01 -5.99865079e-01 -8.29014182e-03
   1.19216919e-01  9.96484947e+00  9.47195625e+00  1.05376692e+01
   9.66901875e+00  8.96850300e+00  1.04607153e+01  1.03217707e+01
   1.05921202e+01  8.86224842e+00  1.02490377e+01  1.10046263e+01
   1.03037939e+01  1.08140812e+01  1.06419020e+01  1.03999863e+01
   8.78884983e+00  9.74235630e+00  9.93688965e+00  9.57951069e+00
   1.03537464e+01  9.49380589e+00  9.77470875e+00  1.12288208e+01
   9.31978893e+00  9.79195118e+00  1.01040468e+01  9.88591671e+00
   1.07793331e+01  9.90864086e+00  9.97999859e+00  1.04008102e+01
   9.96156311e+00  1.00231171e+01  1.03710451e+01  1.05720434e+01
   1.03938103e+01  1.04774828e+01  9.71849155e+00  1.08947363e+01
   1.02003727e+01  1.03852224e+01  9.15958977e+00  1.04600191e+01
   9.03582764e+00  1.05868587e+01  1.06793652e+01  9.70108700e+00
   1.01490498e+01  9.69367790e+00  8.66985703e+00  9.84745216e+00
   1.02833605e+01  8.51985073e+00  1.04057198e+01  9.53009892e+00
   9.78925800e+00  9.92280674e+00  1.18136578e+01  1.10736561e+01
   1.87452114e+00  5.01922786e-01  2.38396704e-01  2.30990767e-01
   1.86752653e+00  9.69996333e-01  1.33285832e+00  2.71153975e+00
   1.51847529e+00  8.56086969e-01  4.15527463e-01  9.99352753e-01
   9.23997164e-02  8.20390821e-01  3.73538077e-01  5.37614882e-01
   8.87812376e-01  2.24573016e-01  1.19527364e+00  1.40202343e+00
   5.52530766e-01  1.09655058e+00  1.02956939e+00  2.91462922e+00
   6.53430045e-01  1.24761915e+00  1.98168921e+00  2.63055229e+00
   5.24250805e-01  1.62935960e+00  1.02426529e+00  1.12182033e+00
   3.77492845e-01  6.29599750e-01  8.48891199e-01  4.57787335e-01
   4.80641723e-01  1.40140295e-01  6.15073919e-01  1.64685702e+00
   1.48138416e+00  1.62799132e+00  2.57475185e+00  3.16645741e-01
   1.10740650e+00  1.51101768e+00  1.41791582e-01  9.15008605e-01
   6.09009743e-01  6.89432800e-01  1.27835298e+00  2.17813730e+00
   4.92236793e-01  2.64813137e+00  2.66070843e-01  1.53949118e+00
   3.21102798e-01  1.61412001e+00  1.30061054e+00  1.44350708e-01
   5.17493963e-01  1.22632957e+00  1.11985159e+00  8.88876557e-01
   8.45990658e-01  3.08749080e-01  4.61219788e-01  1.99838305e+00
   4.67563272e-01  3.38997602e-01  1.74254644e+00  1.66006780e+00
   6.57038569e-01  1.90365314e-01  1.39015150e+00  3.13863802e+00
   1.30110860e-01  1.05892301e+00  2.58426237e+00  1.98848486e+00
   3.57468188e-01  2.59938240e+00  7.10643291e-01  1.49212706e+00
   1.66310608e+00  3.71068478e-01  2.18955708e+00  2.80175757e+00
   7.05299556e-01  1.90978348e-01  1.92695057e+00  1.58628690e+00
   5.49808919e-01  1.39726305e+00  3.09702277e-01  1.57819605e+00
   6.96515560e-01  2.49149656e+00  1.63551331e+00  8.00609231e-01
   5.82845330e-01  9.35709536e-01  8.20599735e-01  7.10113049e-02
   4.49183702e-01  3.18560362e+00  2.63380766e-01  1.17108774e+00
   1.53343678e-01  5.43992519e-02  3.24741542e-01  2.05651236e+00
   2.54147410e-01  1.38747132e+00  3.13074410e-01  6.33925259e-01
   2.11485410e+00  2.23856831e+00  1.96878910e+00  2.53606319e+00
   1.08874216e+01 -7.86365414e+00 -5.87181950e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 18:12:12.802239
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6439
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 18:12:12.806019
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8606.93
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 18:12:12.809252
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7097
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 18:12:12.812499
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.811
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140275293916464
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140274066906248
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140274066906752
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140274066505912
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140274066506416
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140274066506920

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f947f374f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.590343
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.564461
grad_step = 000002, loss = 0.545032
grad_step = 000003, loss = 0.526362
grad_step = 000004, loss = 0.507437
grad_step = 000005, loss = 0.490683
grad_step = 000006, loss = 0.475983
grad_step = 000007, loss = 0.461431
grad_step = 000008, loss = 0.444044
grad_step = 000009, loss = 0.428312
grad_step = 000010, loss = 0.415285
grad_step = 000011, loss = 0.403858
grad_step = 000012, loss = 0.392791
grad_step = 000013, loss = 0.381163
grad_step = 000014, loss = 0.369121
grad_step = 000015, loss = 0.357137
grad_step = 000016, loss = 0.345245
grad_step = 000017, loss = 0.333412
grad_step = 000018, loss = 0.321506
grad_step = 000019, loss = 0.309709
grad_step = 000020, loss = 0.298308
grad_step = 000021, loss = 0.287196
grad_step = 000022, loss = 0.276244
grad_step = 000023, loss = 0.265400
grad_step = 000024, loss = 0.254849
grad_step = 000025, loss = 0.244655
grad_step = 000026, loss = 0.232844
grad_step = 000027, loss = 0.220701
grad_step = 000028, loss = 0.208791
grad_step = 000029, loss = 0.197304
grad_step = 000030, loss = 0.186472
grad_step = 000031, loss = 0.176672
grad_step = 000032, loss = 0.167930
grad_step = 000033, loss = 0.159348
grad_step = 000034, loss = 0.151086
grad_step = 000035, loss = 0.143076
grad_step = 000036, loss = 0.135311
grad_step = 000037, loss = 0.127648
grad_step = 000038, loss = 0.120052
grad_step = 000039, loss = 0.112601
grad_step = 000040, loss = 0.105313
grad_step = 000041, loss = 0.098343
grad_step = 000042, loss = 0.091945
grad_step = 000043, loss = 0.086045
grad_step = 000044, loss = 0.080453
grad_step = 000045, loss = 0.075079
grad_step = 000046, loss = 0.069883
grad_step = 000047, loss = 0.064884
grad_step = 000048, loss = 0.060113
grad_step = 000049, loss = 0.055587
grad_step = 000050, loss = 0.051353
grad_step = 000051, loss = 0.047382
grad_step = 000052, loss = 0.043664
grad_step = 000053, loss = 0.040208
grad_step = 000054, loss = 0.036987
grad_step = 000055, loss = 0.034004
grad_step = 000056, loss = 0.031281
grad_step = 000057, loss = 0.028877
grad_step = 000058, loss = 0.026694
grad_step = 000059, loss = 0.024098
grad_step = 000060, loss = 0.021738
grad_step = 000061, loss = 0.020038
grad_step = 000062, loss = 0.018242
grad_step = 000063, loss = 0.016359
grad_step = 000064, loss = 0.014932
grad_step = 000065, loss = 0.013683
grad_step = 000066, loss = 0.012239
grad_step = 000067, loss = 0.011048
grad_step = 000068, loss = 0.010140
grad_step = 000069, loss = 0.009126
grad_step = 000070, loss = 0.008175
grad_step = 000071, loss = 0.007513
grad_step = 000072, loss = 0.006852
grad_step = 000073, loss = 0.006134
grad_step = 000074, loss = 0.005609
grad_step = 000075, loss = 0.005232
grad_step = 000076, loss = 0.004785
grad_step = 000077, loss = 0.004359
grad_step = 000078, loss = 0.004073
grad_step = 000079, loss = 0.003839
grad_step = 000080, loss = 0.003567
grad_step = 000081, loss = 0.003308
grad_step = 000082, loss = 0.003142
grad_step = 000083, loss = 0.003025
grad_step = 000084, loss = 0.002885
grad_step = 000085, loss = 0.002731
grad_step = 000086, loss = 0.002612
grad_step = 000087, loss = 0.002540
grad_step = 000088, loss = 0.002480
grad_step = 000089, loss = 0.002405
grad_step = 000090, loss = 0.002321
grad_step = 000091, loss = 0.002249
grad_step = 000092, loss = 0.002200
grad_step = 000093, loss = 0.002171
grad_step = 000094, loss = 0.002151
grad_step = 000095, loss = 0.002134
grad_step = 000096, loss = 0.002113
grad_step = 000097, loss = 0.002092
grad_step = 000098, loss = 0.002068
grad_step = 000099, loss = 0.002046
grad_step = 000100, loss = 0.002024
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002004
grad_step = 000102, loss = 0.001986
grad_step = 000103, loss = 0.001971
grad_step = 000104, loss = 0.001958
grad_step = 000105, loss = 0.001948
grad_step = 000106, loss = 0.001941
grad_step = 000107, loss = 0.001935
grad_step = 000108, loss = 0.001930
grad_step = 000109, loss = 0.001927
grad_step = 000110, loss = 0.001926
grad_step = 000111, loss = 0.001930
grad_step = 000112, loss = 0.001951
grad_step = 000113, loss = 0.002019
grad_step = 000114, loss = 0.002201
grad_step = 000115, loss = 0.002630
grad_step = 000116, loss = 0.002964
grad_step = 000117, loss = 0.002712
grad_step = 000118, loss = 0.001965
grad_step = 000119, loss = 0.002111
grad_step = 000120, loss = 0.002597
grad_step = 000121, loss = 0.002178
grad_step = 000122, loss = 0.001898
grad_step = 000123, loss = 0.002277
grad_step = 000124, loss = 0.002147
grad_step = 000125, loss = 0.001860
grad_step = 000126, loss = 0.002091
grad_step = 000127, loss = 0.002082
grad_step = 000128, loss = 0.001854
grad_step = 000129, loss = 0.001982
grad_step = 000130, loss = 0.002017
grad_step = 000131, loss = 0.001849
grad_step = 000132, loss = 0.001918
grad_step = 000133, loss = 0.001959
grad_step = 000134, loss = 0.001841
grad_step = 000135, loss = 0.001886
grad_step = 000136, loss = 0.001919
grad_step = 000137, loss = 0.001834
grad_step = 000138, loss = 0.001861
grad_step = 000139, loss = 0.001890
grad_step = 000140, loss = 0.001828
grad_step = 000141, loss = 0.001838
grad_step = 000142, loss = 0.001871
grad_step = 000143, loss = 0.001832
grad_step = 000144, loss = 0.001818
grad_step = 000145, loss = 0.001850
grad_step = 000146, loss = 0.001834
grad_step = 000147, loss = 0.001808
grad_step = 000148, loss = 0.001828
grad_step = 000149, loss = 0.001831
grad_step = 000150, loss = 0.001806
grad_step = 000151, loss = 0.001809
grad_step = 000152, loss = 0.001821
grad_step = 000153, loss = 0.001806
grad_step = 000154, loss = 0.001798
grad_step = 000155, loss = 0.001808
grad_step = 000156, loss = 0.001806
grad_step = 000157, loss = 0.001793
grad_step = 000158, loss = 0.001795
grad_step = 000159, loss = 0.001800
grad_step = 000160, loss = 0.001793
grad_step = 000161, loss = 0.001786
grad_step = 000162, loss = 0.001789
grad_step = 000163, loss = 0.001790
grad_step = 000164, loss = 0.001784
grad_step = 000165, loss = 0.001779
grad_step = 000166, loss = 0.001781
grad_step = 000167, loss = 0.001782
grad_step = 000168, loss = 0.001777
grad_step = 000169, loss = 0.001773
grad_step = 000170, loss = 0.001774
grad_step = 000171, loss = 0.001774
grad_step = 000172, loss = 0.001771
grad_step = 000173, loss = 0.001767
grad_step = 000174, loss = 0.001766
grad_step = 000175, loss = 0.001766
grad_step = 000176, loss = 0.001765
grad_step = 000177, loss = 0.001762
grad_step = 000178, loss = 0.001760
grad_step = 000179, loss = 0.001759
grad_step = 000180, loss = 0.001758
grad_step = 000181, loss = 0.001757
grad_step = 000182, loss = 0.001755
grad_step = 000183, loss = 0.001752
grad_step = 000184, loss = 0.001751
grad_step = 000185, loss = 0.001750
grad_step = 000186, loss = 0.001749
grad_step = 000187, loss = 0.001747
grad_step = 000188, loss = 0.001745
grad_step = 000189, loss = 0.001743
grad_step = 000190, loss = 0.001742
grad_step = 000191, loss = 0.001740
grad_step = 000192, loss = 0.001739
grad_step = 000193, loss = 0.001738
grad_step = 000194, loss = 0.001736
grad_step = 000195, loss = 0.001735
grad_step = 000196, loss = 0.001733
grad_step = 000197, loss = 0.001731
grad_step = 000198, loss = 0.001730
grad_step = 000199, loss = 0.001728
grad_step = 000200, loss = 0.001726
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001725
grad_step = 000202, loss = 0.001724
grad_step = 000203, loss = 0.001722
grad_step = 000204, loss = 0.001721
grad_step = 000205, loss = 0.001719
grad_step = 000206, loss = 0.001718
grad_step = 000207, loss = 0.001717
grad_step = 000208, loss = 0.001716
grad_step = 000209, loss = 0.001715
grad_step = 000210, loss = 0.001716
grad_step = 000211, loss = 0.001718
grad_step = 000212, loss = 0.001724
grad_step = 000213, loss = 0.001739
grad_step = 000214, loss = 0.001770
grad_step = 000215, loss = 0.001833
grad_step = 000216, loss = 0.001966
grad_step = 000217, loss = 0.002180
grad_step = 000218, loss = 0.002496
grad_step = 000219, loss = 0.002638
grad_step = 000220, loss = 0.002448
grad_step = 000221, loss = 0.001918
grad_step = 000222, loss = 0.001700
grad_step = 000223, loss = 0.001953
grad_step = 000224, loss = 0.002189
grad_step = 000225, loss = 0.002052
grad_step = 000226, loss = 0.001756
grad_step = 000227, loss = 0.001708
grad_step = 000228, loss = 0.001798
grad_step = 000229, loss = 0.001832
grad_step = 000230, loss = 0.001745
grad_step = 000231, loss = 0.001689
grad_step = 000232, loss = 0.001751
grad_step = 000233, loss = 0.001791
grad_step = 000234, loss = 0.001730
grad_step = 000235, loss = 0.001683
grad_step = 000236, loss = 0.001703
grad_step = 000237, loss = 0.001745
grad_step = 000238, loss = 0.001734
grad_step = 000239, loss = 0.001686
grad_step = 000240, loss = 0.001676
grad_step = 000241, loss = 0.001702
grad_step = 000242, loss = 0.001712
grad_step = 000243, loss = 0.001685
grad_step = 000244, loss = 0.001665
grad_step = 000245, loss = 0.001675
grad_step = 000246, loss = 0.001691
grad_step = 000247, loss = 0.001683
grad_step = 000248, loss = 0.001663
grad_step = 000249, loss = 0.001657
grad_step = 000250, loss = 0.001664
grad_step = 000251, loss = 0.001672
grad_step = 000252, loss = 0.001666
grad_step = 000253, loss = 0.001654
grad_step = 000254, loss = 0.001647
grad_step = 000255, loss = 0.001651
grad_step = 000256, loss = 0.001655
grad_step = 000257, loss = 0.001652
grad_step = 000258, loss = 0.001644
grad_step = 000259, loss = 0.001638
grad_step = 000260, loss = 0.001638
grad_step = 000261, loss = 0.001640
grad_step = 000262, loss = 0.001642
grad_step = 000263, loss = 0.001639
grad_step = 000264, loss = 0.001633
grad_step = 000265, loss = 0.001628
grad_step = 000266, loss = 0.001625
grad_step = 000267, loss = 0.001625
grad_step = 000268, loss = 0.001626
grad_step = 000269, loss = 0.001627
grad_step = 000270, loss = 0.001626
grad_step = 000271, loss = 0.001624
grad_step = 000272, loss = 0.001620
grad_step = 000273, loss = 0.001615
grad_step = 000274, loss = 0.001612
grad_step = 000275, loss = 0.001609
grad_step = 000276, loss = 0.001608
grad_step = 000277, loss = 0.001607
grad_step = 000278, loss = 0.001607
grad_step = 000279, loss = 0.001606
grad_step = 000280, loss = 0.001606
grad_step = 000281, loss = 0.001606
grad_step = 000282, loss = 0.001606
grad_step = 000283, loss = 0.001607
grad_step = 000284, loss = 0.001609
grad_step = 000285, loss = 0.001613
grad_step = 000286, loss = 0.001619
grad_step = 000287, loss = 0.001631
grad_step = 000288, loss = 0.001651
grad_step = 000289, loss = 0.001689
grad_step = 000290, loss = 0.001742
grad_step = 000291, loss = 0.001831
grad_step = 000292, loss = 0.001913
grad_step = 000293, loss = 0.002000
grad_step = 000294, loss = 0.001967
grad_step = 000295, loss = 0.001841
grad_step = 000296, loss = 0.001673
grad_step = 000297, loss = 0.001582
grad_step = 000298, loss = 0.001613
grad_step = 000299, loss = 0.001710
grad_step = 000300, loss = 0.001784
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001758
grad_step = 000302, loss = 0.001671
grad_step = 000303, loss = 0.001587
grad_step = 000304, loss = 0.001575
grad_step = 000305, loss = 0.001626
grad_step = 000306, loss = 0.001675
grad_step = 000307, loss = 0.001675
grad_step = 000308, loss = 0.001625
grad_step = 000309, loss = 0.001574
grad_step = 000310, loss = 0.001558
grad_step = 000311, loss = 0.001578
grad_step = 000312, loss = 0.001615
grad_step = 000313, loss = 0.001643
grad_step = 000314, loss = 0.001651
grad_step = 000315, loss = 0.001625
grad_step = 000316, loss = 0.001586
grad_step = 000317, loss = 0.001556
grad_step = 000318, loss = 0.001550
grad_step = 000319, loss = 0.001564
grad_step = 000320, loss = 0.001583
grad_step = 000321, loss = 0.001594
grad_step = 000322, loss = 0.001591
grad_step = 000323, loss = 0.001575
grad_step = 000324, loss = 0.001557
grad_step = 000325, loss = 0.001543
grad_step = 000326, loss = 0.001538
grad_step = 000327, loss = 0.001539
grad_step = 000328, loss = 0.001543
grad_step = 000329, loss = 0.001548
grad_step = 000330, loss = 0.001554
grad_step = 000331, loss = 0.001556
grad_step = 000332, loss = 0.001554
grad_step = 000333, loss = 0.001550
grad_step = 000334, loss = 0.001545
grad_step = 000335, loss = 0.001539
grad_step = 000336, loss = 0.001533
grad_step = 000337, loss = 0.001528
grad_step = 000338, loss = 0.001525
grad_step = 000339, loss = 0.001522
grad_step = 000340, loss = 0.001520
grad_step = 000341, loss = 0.001518
grad_step = 000342, loss = 0.001516
grad_step = 000343, loss = 0.001515
grad_step = 000344, loss = 0.001514
grad_step = 000345, loss = 0.001513
grad_step = 000346, loss = 0.001511
grad_step = 000347, loss = 0.001510
grad_step = 000348, loss = 0.001510
grad_step = 000349, loss = 0.001510
grad_step = 000350, loss = 0.001511
grad_step = 000351, loss = 0.001514
grad_step = 000352, loss = 0.001523
grad_step = 000353, loss = 0.001543
grad_step = 000354, loss = 0.001587
grad_step = 000355, loss = 0.001685
grad_step = 000356, loss = 0.001874
grad_step = 000357, loss = 0.002246
grad_step = 000358, loss = 0.002659
grad_step = 000359, loss = 0.002952
grad_step = 000360, loss = 0.002496
grad_step = 000361, loss = 0.001763
grad_step = 000362, loss = 0.001587
grad_step = 000363, loss = 0.001964
grad_step = 000364, loss = 0.002170
grad_step = 000365, loss = 0.001825
grad_step = 000366, loss = 0.001527
grad_step = 000367, loss = 0.001859
grad_step = 000368, loss = 0.002026
grad_step = 000369, loss = 0.001673
grad_step = 000370, loss = 0.001561
grad_step = 000371, loss = 0.001800
grad_step = 000372, loss = 0.001784
grad_step = 000373, loss = 0.001571
grad_step = 000374, loss = 0.001578
grad_step = 000375, loss = 0.001684
grad_step = 000376, loss = 0.001646
grad_step = 000377, loss = 0.001526
grad_step = 000378, loss = 0.001548
grad_step = 000379, loss = 0.001637
grad_step = 000380, loss = 0.001560
grad_step = 000381, loss = 0.001486
grad_step = 000382, loss = 0.001576
grad_step = 000383, loss = 0.001595
grad_step = 000384, loss = 0.001492
grad_step = 000385, loss = 0.001511
grad_step = 000386, loss = 0.001570
grad_step = 000387, loss = 0.001519
grad_step = 000388, loss = 0.001483
grad_step = 000389, loss = 0.001513
grad_step = 000390, loss = 0.001520
grad_step = 000391, loss = 0.001493
grad_step = 000392, loss = 0.001474
grad_step = 000393, loss = 0.001493
grad_step = 000394, loss = 0.001502
grad_step = 000395, loss = 0.001472
grad_step = 000396, loss = 0.001466
grad_step = 000397, loss = 0.001484
grad_step = 000398, loss = 0.001480
grad_step = 000399, loss = 0.001458
grad_step = 000400, loss = 0.001462
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001473
grad_step = 000402, loss = 0.001462
grad_step = 000403, loss = 0.001454
grad_step = 000404, loss = 0.001457
grad_step = 000405, loss = 0.001459
grad_step = 000406, loss = 0.001454
grad_step = 000407, loss = 0.001448
grad_step = 000408, loss = 0.001448
grad_step = 000409, loss = 0.001449
grad_step = 000410, loss = 0.001448
grad_step = 000411, loss = 0.001442
grad_step = 000412, loss = 0.001440
grad_step = 000413, loss = 0.001443
grad_step = 000414, loss = 0.001441
grad_step = 000415, loss = 0.001437
grad_step = 000416, loss = 0.001435
grad_step = 000417, loss = 0.001435
grad_step = 000418, loss = 0.001435
grad_step = 000419, loss = 0.001432
grad_step = 000420, loss = 0.001430
grad_step = 000421, loss = 0.001429
grad_step = 000422, loss = 0.001428
grad_step = 000423, loss = 0.001427
grad_step = 000424, loss = 0.001425
grad_step = 000425, loss = 0.001423
grad_step = 000426, loss = 0.001422
grad_step = 000427, loss = 0.001421
grad_step = 000428, loss = 0.001420
grad_step = 000429, loss = 0.001418
grad_step = 000430, loss = 0.001416
grad_step = 000431, loss = 0.001415
grad_step = 000432, loss = 0.001414
grad_step = 000433, loss = 0.001413
grad_step = 000434, loss = 0.001412
grad_step = 000435, loss = 0.001410
grad_step = 000436, loss = 0.001409
grad_step = 000437, loss = 0.001408
grad_step = 000438, loss = 0.001407
grad_step = 000439, loss = 0.001405
grad_step = 000440, loss = 0.001404
grad_step = 000441, loss = 0.001402
grad_step = 000442, loss = 0.001401
grad_step = 000443, loss = 0.001400
grad_step = 000444, loss = 0.001399
grad_step = 000445, loss = 0.001397
grad_step = 000446, loss = 0.001396
grad_step = 000447, loss = 0.001395
grad_step = 000448, loss = 0.001393
grad_step = 000449, loss = 0.001392
grad_step = 000450, loss = 0.001391
grad_step = 000451, loss = 0.001390
grad_step = 000452, loss = 0.001388
grad_step = 000453, loss = 0.001387
grad_step = 000454, loss = 0.001386
grad_step = 000455, loss = 0.001385
grad_step = 000456, loss = 0.001383
grad_step = 000457, loss = 0.001382
grad_step = 000458, loss = 0.001381
grad_step = 000459, loss = 0.001379
grad_step = 000460, loss = 0.001378
grad_step = 000461, loss = 0.001377
grad_step = 000462, loss = 0.001376
grad_step = 000463, loss = 0.001374
grad_step = 000464, loss = 0.001373
grad_step = 000465, loss = 0.001372
grad_step = 000466, loss = 0.001371
grad_step = 000467, loss = 0.001369
grad_step = 000468, loss = 0.001368
grad_step = 000469, loss = 0.001367
grad_step = 000470, loss = 0.001366
grad_step = 000471, loss = 0.001365
grad_step = 000472, loss = 0.001364
grad_step = 000473, loss = 0.001363
grad_step = 000474, loss = 0.001363
grad_step = 000475, loss = 0.001363
grad_step = 000476, loss = 0.001365
grad_step = 000477, loss = 0.001370
grad_step = 000478, loss = 0.001379
grad_step = 000479, loss = 0.001400
grad_step = 000480, loss = 0.001437
grad_step = 000481, loss = 0.001511
grad_step = 000482, loss = 0.001632
grad_step = 000483, loss = 0.001844
grad_step = 000484, loss = 0.002072
grad_step = 000485, loss = 0.002277
grad_step = 000486, loss = 0.002166
grad_step = 000487, loss = 0.001794
grad_step = 000488, loss = 0.001430
grad_step = 000489, loss = 0.001400
grad_step = 000490, loss = 0.001630
grad_step = 000491, loss = 0.001746
grad_step = 000492, loss = 0.001583
grad_step = 000493, loss = 0.001374
grad_step = 000494, loss = 0.001397
grad_step = 000495, loss = 0.001552
grad_step = 000496, loss = 0.001576
grad_step = 000497, loss = 0.001424
grad_step = 000498, loss = 0.001338
grad_step = 000499, loss = 0.001414
grad_step = 000500, loss = 0.001503
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001467
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

  date_run                              2020-05-10 18:12:30.664441
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.256439
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 18:12:30.670369
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.167652
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 18:12:30.677520
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.150645
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 18:12:30.682537
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.54754
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
0   2020-05-10 18:12:04.433907  ...    mean_absolute_error
1   2020-05-10 18:12:04.437476  ...     mean_squared_error
2   2020-05-10 18:12:04.440657  ...  median_absolute_error
3   2020-05-10 18:12:04.443780  ...               r2_score
4   2020-05-10 18:12:12.802239  ...    mean_absolute_error
5   2020-05-10 18:12:12.806019  ...     mean_squared_error
6   2020-05-10 18:12:12.809252  ...  median_absolute_error
7   2020-05-10 18:12:12.812499  ...               r2_score
8   2020-05-10 18:12:30.664441  ...    mean_absolute_error
9   2020-05-10 18:12:30.670369  ...     mean_squared_error
10  2020-05-10 18:12:30.677520  ...  median_absolute_error
11  2020-05-10 18:12:30.682537  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▌      | 3489792/9912422 [00:00<00:00, 34893627.25it/s]9920512it [00:00, 36054227.52it/s]                             
0it [00:00, ?it/s]32768it [00:00, 714781.77it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 974560.04it/s]1654784it [00:00, 12704887.16it/s]                         
0it [00:00, ?it/s]8192it [00:00, 248095.50it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e4da97710> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8deb1dbbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e4da4bdd8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8deb1dbd68> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e4da4bdd8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e4da97e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e4da97710> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e0044bc88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8deb1dbd68> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e0044bc88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8e4da4bdd8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f745f26e208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c8a24e32f92a483a2743e2aca08a85b29720e58d4878aaa18683d26a3672e8d1
  Stored in directory: /tmp/pip-ephem-wheel-cache-px7cxzo3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f74553dd048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1777664/17464789 [==>...........................] - ETA: 0s
 7176192/17464789 [===========>..................] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 18:13:56.000015: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 18:13:56.004716: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-10 18:13:56.004842: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5645c5be00a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 18:13:56.004854: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4366 - accuracy: 0.5150 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5133 - accuracy: 0.5100
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4213 - accuracy: 0.5160
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.4857 - accuracy: 0.5118
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5695 - accuracy: 0.5063
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5571 - accuracy: 0.5071
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5785 - accuracy: 0.5058
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5968 - accuracy: 0.5046
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5930 - accuracy: 0.5048
11000/25000 [============>.................] - ETA: 3s - loss: 7.6332 - accuracy: 0.5022
12000/25000 [=============>................] - ETA: 3s - loss: 7.6130 - accuracy: 0.5035
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5994 - accuracy: 0.5044
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5834 - accuracy: 0.5054
15000/25000 [=================>............] - ETA: 2s - loss: 7.6032 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6321 - accuracy: 0.5023
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6414 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6910 - accuracy: 0.4984
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 7s 279us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 18:14:09.611796
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 18:14:09.611796  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 18:14:15.535700: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 18:14:15.540430: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-10 18:14:15.540619: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5644f36f3770 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 18:14:15.540632: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f2fd6a66d30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3848 - crf_viterbi_accuracy: 0.2800 - val_loss: 1.2764 - val_crf_viterbi_accuracy: 0.3333

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2fcbe0df60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9733 - accuracy: 0.4800
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6590 - accuracy: 0.5005
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6452 - accuracy: 0.5014
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5934 - accuracy: 0.5048
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6406 - accuracy: 0.5017
11000/25000 [============>.................] - ETA: 3s - loss: 7.6164 - accuracy: 0.5033
12000/25000 [=============>................] - ETA: 3s - loss: 7.5976 - accuracy: 0.5045
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5982 - accuracy: 0.5045
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6075 - accuracy: 0.5039
15000/25000 [=================>............] - ETA: 2s - loss: 7.6247 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6197 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6296 - accuracy: 0.5024
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6384 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6475 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6535 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f2f87bca080> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:26:03, 9.42kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:02:15, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:40:52, 18.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 852k/862M [00:01<8:53:08, 26.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.44M/862M [00:01<6:12:16, 38.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.43M/862M [00:01<4:19:29, 54.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<3:00:37, 78.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.8M/862M [00:01<2:06:06, 112kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:27:47, 160kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.3M/862M [00:01<1:01:07, 228kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:02<42:34, 325kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:02<29:49, 462kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.7M/862M [00:02<20:50, 656kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.4M/862M [00:02<14:35, 931kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<10:26, 1.29MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<07:43, 1.74MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<11:12:48, 20.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<7:50:44, 28.5kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<5:30:48, 40.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<3:54:30, 57.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<2:44:38, 81.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:06<1:55:06, 116kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:26:40, 154kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<1:02:00, 215kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<43:40, 304kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<33:35, 394kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<24:52, 532kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:10<17:40, 747kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<15:28, 851kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<12:10, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<08:50, 1.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<09:16, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<07:49, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:14<05:48, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<07:08, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<07:40, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<05:54, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<04:23, 2.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<06:57, 1.86MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<06:15, 2.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<04:42, 2.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<06:16, 2.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<07:02, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<05:30, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:20<04:00, 3.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<10:25, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<08:38, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<06:18, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<07:24, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<07:47, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<06:06, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:18, 2.01MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:44, 2.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:20, 2.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:58, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:28, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:05, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:51, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:23, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:05, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:16, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:41, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<03:56, 3.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:39, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<06:23, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:06, 2.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<03:42, 3.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<11:48:43, 17.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<8:17:08, 24.7kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<5:47:32, 35.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<4:05:26, 49.8kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<2:52:57, 70.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:01:07, 101kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<1:27:24, 139kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<1:02:23, 194kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<43:51, 276kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<31:28, 384kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<8:18:02, 24.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<5:48:27, 34.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<4:05:11, 49.0kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<2:54:16, 68.9kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<2:02:31, 97.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<1:25:34, 140kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<1:18:38, 152kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<56:14, 212kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<39:35, 301kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<30:27, 390kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<23:46, 499kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<17:13, 689kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<13:55, 848kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<10:58, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<07:55, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<08:17, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<07:00, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:11, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<06:23, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<07:20, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:47, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:11, 2.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:29, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:47, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:43, 1.72MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:53, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<07:09, 1.61MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<07:51, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:10, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<04:29, 2.56MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<11:08, 1.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<10:29, 1.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:54, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<05:47, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:51, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<07:36, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:00, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<04:20, 2.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<10:18, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<09:46, 1.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:27, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<05:21, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<14:08, 796kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<12:18, 914kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<09:07, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<06:30, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<11:08, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<10:12, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:39, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:31, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:59, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:59, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:09, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<04:25, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<11:55, 927kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<10:55, 1.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:12, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:55, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<07:27, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:28, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:47, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:10, 2.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<18:18, 596kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<14:57, 729kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:53, 1.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<07:45, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:20, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:28, 1.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:09, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<05:08, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<19:45, 545kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<15:57, 675kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:40, 921kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<08:17, 1.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<24:19, 440kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<19:09, 559kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<13:54, 768kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<09:51, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<25:51, 411kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<20:35, 516kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<15:00, 708kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<10:35, 999kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<12:50, 823kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<11:27, 922kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:37, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<06:10, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<11:39, 900kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<10:37, 988kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<07:56, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:46, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:37, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<06:37, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:08, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:42, 2.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<2:31:03, 68.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<1:47:44, 96.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:15:45, 137kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<53:02, 194kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<41:04, 251kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<30:46, 334kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<22:00, 467kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<15:54, 643kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<6:11:49, 27.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<4:20:39, 39.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<3:01:50, 56.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<2:12:49, 76.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:36:50, 105kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:08:33, 148kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<48:19, 210kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<33:46, 299kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<2:45:02, 61.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:57:44, 85.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:22:53, 122kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<57:54, 173kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<48:26, 207kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<36:14, 276kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<25:57, 386kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<18:14, 546kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<19:23, 513kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<15:48, 630kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:37, 855kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<08:14, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<13:59, 707kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<11:59, 824kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:53, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<06:20, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:27, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:51, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:57, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<04:15, 2.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<1:23:26, 117kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:00:14, 162kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<42:34, 229kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<29:47, 325kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<40:41, 238kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<30:20, 319kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<21:37, 447kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<15:10, 634kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<19:37, 490kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<15:58, 601kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<11:44, 817kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<08:19, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<12:29, 764kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<10:58, 869kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<08:08, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<05:52, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:34, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:23, 1.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:52, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:33, 2.65MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:05, 1.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:21, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:58, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:35, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:33, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:21, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:49, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:30, 2.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<06:07, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:02, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:38, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:20, 2.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:39, 863kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<09:15, 993kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:54, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<04:56, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<22:45, 401kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<17:31, 520kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<12:39, 719kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<10:20, 876kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<09:15, 977kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:55, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:58, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:47, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:51, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:19, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:49, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<09:37, 927kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<08:21, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:11, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:27, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<07:54, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<07:12, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:26, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:54, 2.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<1:14:54, 117kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<54:28, 161kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<38:32, 227kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<26:56, 324kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<24:17, 359kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<18:31, 470kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<13:20, 652kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<10:45, 803kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<09:33, 904kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<07:12, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:08, 1.67MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<09:49, 873kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<08:52, 966kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<06:42, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:46, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<10:00, 850kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<08:54, 955kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:41, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<04:46, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<12:17, 686kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<10:05, 836kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:24, 1.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:36, 1.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:59, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:29, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:15, 2.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:12, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:12, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:43, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:33, 2.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:13, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:18, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:20, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:46, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:22, 1.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:26, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:34, 3.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:27, 3.30MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<5:02:05, 26.8kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<3:31:35, 38.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<2:27:18, 54.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<1:52:14, 71.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<1:21:50, 98.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<58:03, 138kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<40:41, 196kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<30:09, 264kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<22:48, 349kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<16:21, 485kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<11:29, 687kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<30:32, 258kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<23:03, 342kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<16:31, 477kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<11:35, 675kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<37:52, 207kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<28:10, 278kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<20:02, 389kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<14:07, 551kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<12:16, 632kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<09:44, 795kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:05, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:21, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:57, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:29, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<03:13, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:25, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:41, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:02, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:44, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:48, 1.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:43, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:49, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:08, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:15, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<03:29, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:54, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:05, 2.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:21, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:47, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:00, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:17, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:43, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<02:57, 2.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:14, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:42, 1.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<02:56, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:13, 2.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:54, 2.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:10, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:38, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:53, 2.44MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:09, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<03:35, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:51, 2.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:07, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:33, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:46, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:03, 3.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:52, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:00, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:05, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:14, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<06:35, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:56, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:27, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:10, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<07:57, 845kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:53, 976kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:06, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:38, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:55, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<4:10:44, 26.6kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<2:55:29, 37.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<2:01:54, 54.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<2:48:09, 39.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<2:00:10, 54.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<1:24:37, 77.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<59:11, 111kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<42:33, 153kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<31:00, 210kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<21:59, 296kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<16:19, 396kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<12:41, 508kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<09:10, 702kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<07:25, 860kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:28, 987kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:50, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:23, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:19, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:17, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:22, 2.63MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:33, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:24, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:22, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:34, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:45, 2.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:02, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:20, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:32, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:46, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:55, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:14, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:33, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:45, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:06, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:27, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:40, 2.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:59, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:22, 2.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:36, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:58, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:19, 2.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<01:45, 3.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:42, 2.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:03, 1.88MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:25, 2.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:45, 3.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<05:16, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:48, 1.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:38, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:27, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<07:59, 704kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:47, 970kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:55, 1.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<04:32, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:24, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:27, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:08, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:58, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:02, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:10, 2.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<15:56, 341kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<12:12, 444kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<08:47, 616kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<06:58, 768kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:56, 902kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:21, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:07, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:26, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:08, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:09, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:27, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<3:13:24, 27.1kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<2:15:17, 38.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<1:33:45, 55.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<2:26:31, 35.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<1:43:51, 49.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<1:12:51, 70.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<50:45, 101kB/s]   .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<37:01, 138kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<26:47, 190kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<18:55, 268kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<13:58, 360kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<10:25, 481kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<07:25, 674kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<06:07, 810kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<05:09, 961kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:14, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:05, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:23, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:52, 2.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:16, 778kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:49, 716kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:22, 908kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:58, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:58, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:17, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:47, 2.69MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<1:10:10, 68.7kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<51:15, 94.0kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<36:17, 133kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<25:34, 188kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<18:02, 265kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<12:50, 372kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<09:07, 522kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<06:35, 720kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<14:43, 322kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<12:25, 382kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<09:13, 514kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<06:36, 715kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<04:48, 981kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:32, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:58, 940kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<05:38, 829kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:27, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:18, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:29, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:54, 2.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:59, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<04:51, 948kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:54, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:55, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:42, 2.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:55, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:47, 947kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:49, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:51, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:10, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:40, 2.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:57, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:46, 938kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:48, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:51, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:40, 2.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:19, 3.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<11:33, 381kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<10:03, 438kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<07:30, 586kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<05:23, 813kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:54, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:57, 1.47MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:12, 1.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<15:56, 272kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<13:06, 331kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<09:37, 450kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<06:54, 625kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:57, 866kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<03:37, 1.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:58, 857kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<05:19, 801kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:09, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<03:03, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:17, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:45, 2.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:01, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:32, 924kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:36, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:41, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:01, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:34, 2.63MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<04:07, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:41, 879kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:42, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:45, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:04, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:34, 2.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:27, 911kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:40, 869kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:40, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:41, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:02, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:34, 2.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:51, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:31, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:50, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:08, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:36, 2.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:14, 3.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:49, 813kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:46, 822kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:41, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:43, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:01, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<1:02:17, 61.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<44:55, 85.7kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<31:43, 121kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<22:14, 172kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<15:35, 244kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<12:20, 307kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<10:57, 345kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<08:18, 455kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<05:56, 634kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:14, 884kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:06, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:49, 972kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<03:45, 989kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:53, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:07, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:33, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:39, 999kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<04:32, 802kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:41, 987kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:42, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:58, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:42, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:51, 928kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:09, 1.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:18, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:40, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:28, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<03:17, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:42, 1.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:59, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:27, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:40, 934kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:06, 836kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<03:14, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:20, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:41, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:51, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:22, 996kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:58, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:37, 2.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<1:47:50, 30.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<1:15:37, 43.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<52:35, 62.3kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<36:29, 88.9kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<40:15, 80.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<29:20, 110kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<20:49, 155kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<14:39, 220kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<10:15, 312kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<08:03, 394kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<06:40, 475kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:55, 643kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<03:28, 901kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:12, 969kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:11, 971kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:25, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:46, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:16, 2.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<05:15, 577kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<04:30, 673kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:19, 908kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:23, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:18, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:25, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:53, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:22, 2.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:54, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:04, 1.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:38, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:11, 2.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:07, 1.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:10, 1.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:41, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:13, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:54, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:08, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:39, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:12, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:35, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:46, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:23, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:00, 2.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:54, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:52, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:26, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:01, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:23, 751kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:01, 840kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:15, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:35, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:01, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:03, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:34, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:07, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:39, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:33, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:10, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:50, 2.81MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:53, 809kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:34, 908kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:54, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<01:20, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:56, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:50, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:23, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:59, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:28, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:31, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:09, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:49, 2.59MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:34, 1.35MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:33, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:15, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:55, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:43, 2.91MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:37, 3.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:52, 1.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:09, 957kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:42, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:14, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:55, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:44, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:32, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:46, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:25, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:03, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:47, 2.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:35, 3.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<1:34:55, 20.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<1:07:03, 28.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<46:55, 40.9kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<32:34, 58.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<22:35, 83.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<16:24, 113kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<12:47, 145kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<09:14, 200kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<06:30, 282kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<04:33, 399kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<03:12, 561kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<03:18, 540kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:51, 624kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:08, 833kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:32, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:06, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:39, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:41, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:17, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:56, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:42, 2.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:21, 1.22MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:57, 842kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:37, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:11, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:52, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:38, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<15:29, 102kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<11:17, 140kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<07:58, 197kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<05:33, 279kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<03:51, 395kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<03:33, 425kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<03:18, 456kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:32, 594kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:48, 823kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:17, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:56, 1.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:30, 961kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:56, 744kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:33, 921kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:07, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:49, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:36, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:16, 1.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:17, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:58, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:42, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:32, 2.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:24, 3.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<05:31, 236kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<04:40, 280kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<03:26, 378kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:26, 527kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<01:42, 740kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:32, 800kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:50, 671kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:27, 842kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:03, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:45, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:42, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<33:16, 35.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<23:11, 50.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<15:58, 71.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<10:58, 102kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<08:12, 135kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<06:25, 173kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<04:38, 238kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<03:14, 335kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<02:14, 473kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:52, 554kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<10:15, 101kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<06:57, 144kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<04:46, 205kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<03:54, 249kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:59, 324kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<02:08, 448kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<01:29, 628kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<01:02, 876kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:22, 653kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:31, 593kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<01:11, 754kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:51, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:36, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:40, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:55, 901kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:45, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:23, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:21, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:17, 2.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:54, 846kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:56, 808kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:42, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:30, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:23, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:17, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:36, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:40, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:31, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:16, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:12, 2.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:41, 896kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:55, 674kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:45, 826kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:32, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:22, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:16, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:06, 505kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:56, 591kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:40, 800kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:29, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:20, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:14, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:38, 298kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:26, 338kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:03, 453kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:44, 633kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:31, 871kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:21, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:15, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<06:55, 60.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<04:56, 84.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<03:24, 120kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<02:16, 170kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<01:30, 241kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<01:13, 285kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<01:02, 333kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:46, 444kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:31, 619kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:20, 864kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:19, 871kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:22, 739kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:17, 922kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:12, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:08, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:09, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:13, 924kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:10, 809kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:12, 709kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 889kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:03, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:03, 1.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 884kB/s] .vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 860/400000 [00:00<00:46, 8591.24it/s]  0%|          | 1738/400000 [00:00<00:46, 8645.20it/s]  1%|          | 2615/400000 [00:00<00:45, 8681.04it/s]  1%|          | 3496/400000 [00:00<00:45, 8716.85it/s]  1%|          | 4374/400000 [00:00<00:45, 8735.13it/s]  1%|▏         | 5236/400000 [00:00<00:45, 8699.29it/s]  2%|▏         | 6105/400000 [00:00<00:45, 8695.71it/s]  2%|▏         | 6966/400000 [00:00<00:45, 8667.11it/s]  2%|▏         | 7838/400000 [00:00<00:45, 8682.37it/s]  2%|▏         | 8722/400000 [00:01<00:44, 8726.88it/s]  2%|▏         | 9589/400000 [00:01<00:44, 8708.80it/s]  3%|▎         | 10469/400000 [00:01<00:44, 8734.18it/s]  3%|▎         | 11354/400000 [00:01<00:44, 8768.21it/s]  3%|▎         | 12240/400000 [00:01<00:44, 8792.68it/s]  3%|▎         | 13126/400000 [00:01<00:43, 8810.25it/s]  4%|▎         | 14004/400000 [00:01<00:43, 8790.15it/s]  4%|▎         | 14888/400000 [00:01<00:43, 8801.92it/s]  4%|▍         | 15774/400000 [00:01<00:43, 8818.18it/s]  4%|▍         | 16655/400000 [00:01<00:43, 8730.00it/s]  4%|▍         | 17528/400000 [00:02<00:44, 8518.09it/s]  5%|▍         | 18395/400000 [00:02<00:44, 8561.36it/s]  5%|▍         | 19281/400000 [00:02<00:44, 8647.78it/s]  5%|▌         | 20155/400000 [00:02<00:43, 8674.89it/s]  5%|▌         | 21033/400000 [00:02<00:43, 8704.94it/s]  5%|▌         | 21906/400000 [00:02<00:43, 8710.35it/s]  6%|▌         | 22779/400000 [00:02<00:43, 8713.63it/s]  6%|▌         | 23660/400000 [00:02<00:43, 8739.42it/s]  6%|▌         | 24540/400000 [00:02<00:42, 8756.89it/s]  6%|▋         | 25424/400000 [00:02<00:42, 8779.26it/s]  7%|▋         | 26307/400000 [00:03<00:42, 8791.52it/s]  7%|▋         | 27187/400000 [00:03<00:42, 8779.64it/s]  7%|▋         | 28067/400000 [00:03<00:42, 8783.99it/s]  7%|▋         | 28946/400000 [00:03<00:42, 8780.85it/s]  7%|▋         | 29826/400000 [00:03<00:42, 8784.68it/s]  8%|▊         | 30709/400000 [00:03<00:41, 8797.97it/s]  8%|▊         | 31589/400000 [00:03<00:43, 8539.00it/s]  8%|▊         | 32474/400000 [00:03<00:42, 8627.70it/s]  8%|▊         | 33360/400000 [00:03<00:42, 8693.32it/s]  9%|▊         | 34242/400000 [00:03<00:41, 8730.27it/s]  9%|▉         | 35116/400000 [00:04<00:41, 8726.33it/s]  9%|▉         | 35990/400000 [00:04<00:41, 8701.05it/s]  9%|▉         | 36873/400000 [00:04<00:41, 8736.56it/s]  9%|▉         | 37760/400000 [00:04<00:41, 8775.96it/s] 10%|▉         | 38638/400000 [00:04<00:41, 8775.73it/s] 10%|▉         | 39526/400000 [00:04<00:40, 8804.20it/s] 10%|█         | 40407/400000 [00:04<00:40, 8772.52it/s] 10%|█         | 41285/400000 [00:04<00:41, 8698.47it/s] 11%|█         | 42163/400000 [00:04<00:41, 8720.68it/s] 11%|█         | 43045/400000 [00:04<00:40, 8749.42it/s] 11%|█         | 43922/400000 [00:05<00:40, 8753.83it/s] 11%|█         | 44798/400000 [00:05<00:40, 8748.61it/s] 11%|█▏        | 45673/400000 [00:05<00:40, 8739.88it/s] 12%|█▏        | 46559/400000 [00:05<00:40, 8773.14it/s] 12%|█▏        | 47443/400000 [00:05<00:40, 8791.67it/s] 12%|█▏        | 48330/400000 [00:05<00:39, 8814.80it/s] 12%|█▏        | 49212/400000 [00:05<00:39, 8771.17it/s] 13%|█▎        | 50098/400000 [00:05<00:39, 8795.99it/s] 13%|█▎        | 50979/400000 [00:05<00:39, 8798.52it/s] 13%|█▎        | 51867/400000 [00:05<00:39, 8820.66it/s] 13%|█▎        | 52750/400000 [00:06<00:39, 8808.50it/s] 13%|█▎        | 53631/400000 [00:06<00:39, 8742.86it/s] 14%|█▎        | 54514/400000 [00:06<00:39, 8766.61it/s] 14%|█▍        | 55397/400000 [00:06<00:39, 8783.36it/s] 14%|█▍        | 56281/400000 [00:06<00:39, 8798.37it/s] 14%|█▍        | 57161/400000 [00:06<00:39, 8776.43it/s] 15%|█▍        | 58039/400000 [00:06<00:38, 8777.12it/s] 15%|█▍        | 58917/400000 [00:06<00:38, 8768.76it/s] 15%|█▍        | 59794/400000 [00:06<00:38, 8746.67it/s] 15%|█▌        | 60677/400000 [00:06<00:38, 8769.74it/s] 15%|█▌        | 61555/400000 [00:07<00:38, 8709.15it/s] 16%|█▌        | 62427/400000 [00:07<00:38, 8665.83it/s] 16%|█▌        | 63314/400000 [00:07<00:38, 8724.94it/s] 16%|█▌        | 64190/400000 [00:07<00:38, 8732.68it/s] 16%|█▋        | 65067/400000 [00:07<00:38, 8741.96it/s] 16%|█▋        | 65944/400000 [00:07<00:38, 8747.49it/s] 17%|█▋        | 66819/400000 [00:07<00:38, 8735.70it/s] 17%|█▋        | 67704/400000 [00:07<00:37, 8769.31it/s] 17%|█▋        | 68588/400000 [00:07<00:37, 8789.57it/s] 17%|█▋        | 69468/400000 [00:07<00:37, 8761.91it/s] 18%|█▊        | 70350/400000 [00:08<00:37, 8778.08it/s] 18%|█▊        | 71228/400000 [00:08<00:37, 8744.36it/s] 18%|█▊        | 72109/400000 [00:08<00:37, 8761.74it/s] 18%|█▊        | 72993/400000 [00:08<00:37, 8784.15it/s] 18%|█▊        | 73872/400000 [00:08<00:37, 8783.60it/s] 19%|█▊        | 74753/400000 [00:08<00:36, 8790.95it/s] 19%|█▉        | 75633/400000 [00:08<00:37, 8749.23it/s] 19%|█▉        | 76511/400000 [00:08<00:36, 8758.28it/s] 19%|█▉        | 77391/400000 [00:08<00:36, 8769.19it/s] 20%|█▉        | 78268/400000 [00:08<00:36, 8760.24it/s] 20%|█▉        | 79145/400000 [00:09<00:36, 8757.19it/s] 20%|██        | 80021/400000 [00:09<00:37, 8460.28it/s] 20%|██        | 80870/400000 [00:09<00:37, 8440.07it/s] 20%|██        | 81753/400000 [00:09<00:37, 8551.35it/s] 21%|██        | 82633/400000 [00:09<00:36, 8622.15it/s] 21%|██        | 83519/400000 [00:09<00:36, 8690.58it/s] 21%|██        | 84389/400000 [00:09<00:36, 8687.17it/s] 21%|██▏       | 85274/400000 [00:09<00:36, 8735.16it/s] 22%|██▏       | 86159/400000 [00:09<00:35, 8767.79it/s] 22%|██▏       | 87044/400000 [00:09<00:35, 8789.56it/s] 22%|██▏       | 87926/400000 [00:10<00:35, 8797.60it/s] 22%|██▏       | 88806/400000 [00:10<00:35, 8757.84it/s] 22%|██▏       | 89684/400000 [00:10<00:35, 8763.54it/s] 23%|██▎       | 90571/400000 [00:10<00:35, 8794.62it/s] 23%|██▎       | 91455/400000 [00:10<00:35, 8806.38it/s] 23%|██▎       | 92336/400000 [00:10<00:35, 8782.53it/s] 23%|██▎       | 93215/400000 [00:10<00:34, 8766.50it/s] 24%|██▎       | 94092/400000 [00:10<00:34, 8754.77it/s] 24%|██▎       | 94968/400000 [00:10<00:35, 8699.51it/s] 24%|██▍       | 95849/400000 [00:10<00:34, 8730.85it/s] 24%|██▍       | 96729/400000 [00:11<00:34, 8750.99it/s] 24%|██▍       | 97605/400000 [00:11<00:34, 8746.20it/s] 25%|██▍       | 98485/400000 [00:11<00:34, 8760.21it/s] 25%|██▍       | 99371/400000 [00:11<00:34, 8789.88it/s] 25%|██▌       | 100260/400000 [00:11<00:33, 8816.58it/s] 25%|██▌       | 101146/400000 [00:11<00:33, 8826.84it/s] 26%|██▌       | 102029/400000 [00:11<00:33, 8811.68it/s] 26%|██▌       | 102911/400000 [00:11<00:33, 8783.38it/s] 26%|██▌       | 103797/400000 [00:11<00:33, 8803.78it/s] 26%|██▌       | 104679/400000 [00:11<00:33, 8807.81it/s] 26%|██▋       | 105564/400000 [00:12<00:33, 8820.09it/s] 27%|██▋       | 106449/400000 [00:12<00:33, 8826.73it/s] 27%|██▋       | 107332/400000 [00:12<00:33, 8781.84it/s] 27%|██▋       | 108212/400000 [00:12<00:33, 8784.44it/s] 27%|██▋       | 109091/400000 [00:12<00:33, 8767.04it/s] 27%|██▋       | 109974/400000 [00:12<00:33, 8785.50it/s] 28%|██▊       | 110857/400000 [00:12<00:32, 8796.43it/s] 28%|██▊       | 111737/400000 [00:12<00:32, 8772.37it/s] 28%|██▊       | 112620/400000 [00:12<00:32, 8788.48it/s] 28%|██▊       | 113503/400000 [00:12<00:32, 8798.29it/s] 29%|██▊       | 114383/400000 [00:13<00:32, 8764.08it/s] 29%|██▉       | 115265/400000 [00:13<00:32, 8780.42it/s] 29%|██▉       | 116144/400000 [00:13<00:32, 8757.29it/s] 29%|██▉       | 117024/400000 [00:13<00:32, 8769.05it/s] 29%|██▉       | 117910/400000 [00:13<00:32, 8796.08it/s] 30%|██▉       | 118792/400000 [00:13<00:31, 8801.25it/s] 30%|██▉       | 119673/400000 [00:13<00:32, 8758.71it/s] 30%|███       | 120549/400000 [00:13<00:32, 8726.19it/s] 30%|███       | 121436/400000 [00:13<00:31, 8766.57it/s] 31%|███       | 122322/400000 [00:13<00:31, 8793.86it/s] 31%|███       | 123208/400000 [00:14<00:31, 8813.44it/s] 31%|███       | 124097/400000 [00:14<00:31, 8836.05it/s] 31%|███       | 124981/400000 [00:14<00:31, 8692.99it/s] 31%|███▏      | 125851/400000 [00:14<00:32, 8543.88it/s] 32%|███▏      | 126735/400000 [00:14<00:31, 8630.43it/s] 32%|███▏      | 127623/400000 [00:14<00:31, 8703.64it/s] 32%|███▏      | 128502/400000 [00:14<00:31, 8726.87it/s] 32%|███▏      | 129376/400000 [00:14<00:31, 8718.63it/s] 33%|███▎      | 130262/400000 [00:14<00:30, 8759.18it/s] 33%|███▎      | 131148/400000 [00:14<00:30, 8788.34it/s] 33%|███▎      | 132036/400000 [00:15<00:30, 8815.51it/s] 33%|███▎      | 132923/400000 [00:15<00:30, 8830.62it/s] 33%|███▎      | 133807/400000 [00:15<00:30, 8739.99it/s] 34%|███▎      | 134689/400000 [00:15<00:30, 8762.30it/s] 34%|███▍      | 135574/400000 [00:15<00:30, 8786.86it/s] 34%|███▍      | 136460/400000 [00:15<00:29, 8806.55it/s] 34%|███▍      | 137341/400000 [00:15<00:29, 8806.42it/s] 35%|███▍      | 138222/400000 [00:15<00:29, 8791.85it/s] 35%|███▍      | 139102/400000 [00:15<00:29, 8786.80it/s] 35%|███▍      | 139987/400000 [00:15<00:29, 8802.78it/s] 35%|███▌      | 140868/400000 [00:16<00:29, 8782.28it/s] 35%|███▌      | 141747/400000 [00:16<00:29, 8777.00it/s] 36%|███▌      | 142625/400000 [00:16<00:29, 8762.72it/s] 36%|███▌      | 143504/400000 [00:16<00:29, 8769.93it/s] 36%|███▌      | 144388/400000 [00:16<00:29, 8788.87it/s] 36%|███▋      | 145267/400000 [00:16<00:30, 8466.61it/s] 37%|███▋      | 146154/400000 [00:16<00:29, 8583.14it/s] 37%|███▋      | 147022/400000 [00:16<00:29, 8610.47it/s] 37%|███▋      | 147908/400000 [00:16<00:29, 8682.14it/s] 37%|███▋      | 148778/400000 [00:17<00:29, 8478.27it/s] 37%|███▋      | 149660/400000 [00:17<00:29, 8577.72it/s] 38%|███▊      | 150544/400000 [00:17<00:28, 8653.63it/s] 38%|███▊      | 151411/400000 [00:17<00:28, 8656.61it/s] 38%|███▊      | 152283/400000 [00:17<00:28, 8673.47it/s] 38%|███▊      | 153151/400000 [00:17<00:28, 8662.39it/s] 39%|███▊      | 154037/400000 [00:17<00:28, 8718.13it/s] 39%|███▊      | 154918/400000 [00:17<00:28, 8743.67it/s] 39%|███▉      | 155793/400000 [00:17<00:28, 8643.47it/s] 39%|███▉      | 156660/400000 [00:17<00:28, 8649.29it/s] 39%|███▉      | 157545/400000 [00:18<00:27, 8706.17it/s] 40%|███▉      | 158431/400000 [00:18<00:27, 8748.79it/s] 40%|███▉      | 159318/400000 [00:18<00:27, 8783.30it/s] 40%|████      | 160197/400000 [00:18<00:27, 8773.77it/s] 40%|████      | 161075/400000 [00:18<00:27, 8659.65it/s] 40%|████      | 161962/400000 [00:18<00:27, 8720.77it/s] 41%|████      | 162836/400000 [00:18<00:27, 8725.20it/s] 41%|████      | 163713/400000 [00:18<00:27, 8736.42it/s] 41%|████      | 164587/400000 [00:18<00:26, 8722.64it/s] 41%|████▏     | 165460/400000 [00:18<00:27, 8553.82it/s] 42%|████▏     | 166317/400000 [00:19<00:28, 8323.37it/s] 42%|████▏     | 167198/400000 [00:19<00:27, 8461.60it/s] 42%|████▏     | 168066/400000 [00:19<00:27, 8523.79it/s] 42%|████▏     | 168924/400000 [00:19<00:27, 8538.78it/s] 42%|████▏     | 169781/400000 [00:19<00:26, 8546.12it/s] 43%|████▎     | 170646/400000 [00:19<00:26, 8574.69it/s] 43%|████▎     | 171504/400000 [00:19<00:27, 8288.54it/s] 43%|████▎     | 172384/400000 [00:19<00:26, 8434.51it/s] 43%|████▎     | 173259/400000 [00:19<00:26, 8526.01it/s] 44%|████▎     | 174129/400000 [00:19<00:26, 8576.51it/s] 44%|████▍     | 175016/400000 [00:20<00:25, 8660.37it/s] 44%|████▍     | 175902/400000 [00:20<00:25, 8718.88it/s] 44%|████▍     | 176786/400000 [00:20<00:25, 8751.92it/s] 44%|████▍     | 177665/400000 [00:20<00:25, 8762.87it/s] 45%|████▍     | 178550/400000 [00:20<00:25, 8787.17it/s] 45%|████▍     | 179437/400000 [00:20<00:25, 8809.85it/s] 45%|████▌     | 180319/400000 [00:20<00:24, 8803.93it/s] 45%|████▌     | 181203/400000 [00:20<00:24, 8813.94it/s] 46%|████▌     | 182085/400000 [00:20<00:24, 8750.79it/s] 46%|████▌     | 182972/400000 [00:20<00:24, 8784.62it/s] 46%|████▌     | 183851/400000 [00:21<00:24, 8757.58it/s] 46%|████▌     | 184737/400000 [00:21<00:24, 8786.11it/s] 46%|████▋     | 185617/400000 [00:21<00:24, 8789.00it/s] 47%|████▋     | 186496/400000 [00:21<00:24, 8765.62it/s] 47%|████▋     | 187376/400000 [00:21<00:24, 8774.61it/s] 47%|████▋     | 188259/400000 [00:21<00:24, 8788.85it/s] 47%|████▋     | 189142/400000 [00:21<00:23, 8799.39it/s] 48%|████▊     | 190022/400000 [00:21<00:23, 8796.26it/s] 48%|████▊     | 190902/400000 [00:21<00:23, 8781.95it/s] 48%|████▊     | 191781/400000 [00:21<00:23, 8726.27it/s] 48%|████▊     | 192654/400000 [00:22<00:23, 8719.25it/s] 48%|████▊     | 193535/400000 [00:22<00:23, 8744.99it/s] 49%|████▊     | 194422/400000 [00:22<00:23, 8781.93it/s] 49%|████▉     | 195301/400000 [00:22<00:23, 8739.54it/s] 49%|████▉     | 196176/400000 [00:22<00:23, 8742.33it/s] 49%|████▉     | 197056/400000 [00:22<00:23, 8758.23it/s] 49%|████▉     | 197938/400000 [00:22<00:23, 8775.74it/s] 50%|████▉     | 198821/400000 [00:22<00:22, 8789.24it/s] 50%|████▉     | 199700/400000 [00:22<00:22, 8731.84it/s] 50%|█████     | 200583/400000 [00:22<00:22, 8760.87it/s] 50%|█████     | 201461/400000 [00:23<00:22, 8766.41it/s] 51%|█████     | 202346/400000 [00:23<00:22, 8789.15it/s] 51%|█████     | 203229/400000 [00:23<00:22, 8800.01it/s] 51%|█████     | 204110/400000 [00:23<00:22, 8623.02it/s] 51%|█████     | 204990/400000 [00:23<00:22, 8673.91it/s] 51%|█████▏    | 205859/400000 [00:23<00:22, 8625.82it/s] 52%|█████▏    | 206738/400000 [00:23<00:22, 8671.59it/s] 52%|█████▏    | 207621/400000 [00:23<00:22, 8716.22it/s] 52%|█████▏    | 208493/400000 [00:23<00:22, 8695.43it/s] 52%|█████▏    | 209377/400000 [00:23<00:21, 8737.24it/s] 53%|█████▎    | 210265/400000 [00:24<00:21, 8776.93it/s] 53%|█████▎    | 211154/400000 [00:24<00:21, 8809.41it/s] 53%|█████▎    | 212040/400000 [00:24<00:21, 8820.32it/s] 53%|█████▎    | 212923/400000 [00:24<00:21, 8788.00it/s] 53%|█████▎    | 213813/400000 [00:24<00:21, 8819.09it/s] 54%|█████▎    | 214698/400000 [00:24<00:20, 8828.33it/s] 54%|█████▍    | 215581/400000 [00:24<00:21, 8775.05it/s] 54%|█████▍    | 216461/400000 [00:24<00:20, 8781.35it/s] 54%|█████▍    | 217340/400000 [00:24<00:20, 8777.95it/s] 55%|█████▍    | 218218/400000 [00:24<00:20, 8712.26it/s] 55%|█████▍    | 219090/400000 [00:25<00:20, 8676.58it/s] 55%|█████▍    | 219973/400000 [00:25<00:20, 8720.19it/s] 55%|█████▌    | 220852/400000 [00:25<00:20, 8738.23it/s] 55%|█████▌    | 221726/400000 [00:25<00:20, 8720.44it/s] 56%|█████▌    | 222607/400000 [00:25<00:20, 8744.53it/s] 56%|█████▌    | 223482/400000 [00:25<00:20, 8726.12it/s] 56%|█████▌    | 224360/400000 [00:25<00:20, 8739.51it/s] 56%|█████▋    | 225247/400000 [00:25<00:19, 8775.43it/s] 57%|█████▋    | 226125/400000 [00:25<00:19, 8776.73it/s] 57%|█████▋    | 227005/400000 [00:25<00:19, 8783.61it/s] 57%|█████▋    | 227892/400000 [00:26<00:19, 8808.64it/s] 57%|█████▋    | 228780/400000 [00:26<00:19, 8827.42it/s] 57%|█████▋    | 229668/400000 [00:26<00:19, 8841.84it/s] 58%|█████▊    | 230553/400000 [00:26<00:19, 8832.82it/s] 58%|█████▊    | 231437/400000 [00:26<00:19, 8749.43it/s] 58%|█████▊    | 232319/400000 [00:26<00:19, 8768.17it/s] 58%|█████▊    | 233202/400000 [00:26<00:18, 8784.68it/s] 59%|█████▊    | 234085/400000 [00:26<00:18, 8796.21it/s] 59%|█████▊    | 234965/400000 [00:26<00:18, 8781.82it/s] 59%|█████▉    | 235844/400000 [00:27<00:18, 8725.79it/s] 59%|█████▉    | 236728/400000 [00:27<00:18, 8758.15it/s] 59%|█████▉    | 237605/400000 [00:27<00:18, 8761.41it/s] 60%|█████▉    | 238491/400000 [00:27<00:18, 8788.79it/s] 60%|█████▉    | 239370/400000 [00:27<00:18, 8787.83it/s] 60%|██████    | 240249/400000 [00:27<00:18, 8760.18it/s] 60%|██████    | 241134/400000 [00:27<00:18, 8786.58it/s] 61%|██████    | 242014/400000 [00:27<00:17, 8789.68it/s] 61%|██████    | 242897/400000 [00:27<00:17, 8801.60it/s] 61%|██████    | 243782/400000 [00:27<00:17, 8814.54it/s] 61%|██████    | 244664/400000 [00:28<00:17, 8762.23it/s] 61%|██████▏   | 245545/400000 [00:28<00:17, 8774.85it/s] 62%|██████▏   | 246423/400000 [00:28<00:17, 8744.50it/s] 62%|██████▏   | 247306/400000 [00:28<00:17, 8768.96it/s] 62%|██████▏   | 248183/400000 [00:28<00:17, 8746.95it/s] 62%|██████▏   | 249058/400000 [00:28<00:17, 8598.23it/s] 62%|██████▏   | 249919/400000 [00:28<00:17, 8577.26it/s] 63%|██████▎   | 250778/400000 [00:28<00:17, 8439.11it/s] 63%|██████▎   | 251663/400000 [00:28<00:17, 8556.39it/s] 63%|██████▎   | 252546/400000 [00:28<00:17, 8634.67it/s] 63%|██████▎   | 253413/400000 [00:29<00:16, 8643.52it/s] 64%|██████▎   | 254288/400000 [00:29<00:16, 8674.16it/s] 64%|██████▍   | 255175/400000 [00:29<00:16, 8731.14it/s] 64%|██████▍   | 256056/400000 [00:29<00:16, 8753.09it/s] 64%|██████▍   | 256932/400000 [00:29<00:16, 8754.82it/s] 64%|██████▍   | 257808/400000 [00:29<00:16, 8685.15it/s] 65%|██████▍   | 258681/400000 [00:29<00:16, 8698.00it/s] 65%|██████▍   | 259565/400000 [00:29<00:16, 8738.24it/s] 65%|██████▌   | 260453/400000 [00:29<00:15, 8777.59it/s] 65%|██████▌   | 261338/400000 [00:29<00:15, 8797.06it/s] 66%|██████▌   | 262218/400000 [00:30<00:15, 8779.29it/s] 66%|██████▌   | 263097/400000 [00:30<00:15, 8767.05it/s] 66%|██████▌   | 263984/400000 [00:30<00:15, 8796.95it/s] 66%|██████▌   | 264864/400000 [00:30<00:15, 8789.05it/s] 66%|██████▋   | 265749/400000 [00:30<00:15, 8805.66it/s] 67%|██████▋   | 266630/400000 [00:30<00:15, 8789.68it/s] 67%|██████▋   | 267517/400000 [00:30<00:15, 8810.74it/s] 67%|██████▋   | 268403/400000 [00:30<00:14, 8824.73it/s] 67%|██████▋   | 269286/400000 [00:30<00:14, 8823.34it/s] 68%|██████▊   | 270169/400000 [00:30<00:14, 8811.97it/s] 68%|██████▊   | 271051/400000 [00:31<00:14, 8716.97it/s] 68%|██████▊   | 271930/400000 [00:31<00:14, 8737.76it/s] 68%|██████▊   | 272819/400000 [00:31<00:14, 8780.74it/s] 68%|██████▊   | 273702/400000 [00:31<00:14, 8792.60it/s] 69%|██████▊   | 274589/400000 [00:31<00:14, 8814.27it/s] 69%|██████▉   | 275471/400000 [00:31<00:14, 8781.08it/s] 69%|██████▉   | 276350/400000 [00:31<00:14, 8762.64it/s] 69%|██████▉   | 277227/400000 [00:31<00:14, 8738.78it/s] 70%|██████▉   | 278113/400000 [00:31<00:13, 8773.68it/s] 70%|██████▉   | 279001/400000 [00:31<00:13, 8802.84it/s] 70%|██████▉   | 279882/400000 [00:32<00:13, 8765.46it/s] 70%|███████   | 280760/400000 [00:32<00:13, 8767.90it/s] 70%|███████   | 281645/400000 [00:32<00:13, 8790.09it/s] 71%|███████   | 282529/400000 [00:32<00:13, 8804.61it/s] 71%|███████   | 283410/400000 [00:32<00:13, 8748.07it/s] 71%|███████   | 284288/400000 [00:32<00:13, 8754.97it/s] 71%|███████▏  | 285170/400000 [00:32<00:13, 8773.36it/s] 72%|███████▏  | 286054/400000 [00:32<00:12, 8791.34it/s] 72%|███████▏  | 286937/400000 [00:32<00:12, 8801.85it/s] 72%|███████▏  | 287818/400000 [00:32<00:12, 8783.46it/s] 72%|███████▏  | 288697/400000 [00:33<00:12, 8770.15it/s] 72%|███████▏  | 289575/400000 [00:33<00:12, 8747.51it/s] 73%|███████▎  | 290450/400000 [00:33<00:12, 8722.29it/s] 73%|███████▎  | 291331/400000 [00:33<00:12, 8747.32it/s] 73%|███████▎  | 292218/400000 [00:33<00:12, 8783.16it/s] 73%|███████▎  | 293101/400000 [00:33<00:12, 8794.43it/s] 73%|███████▎  | 293981/400000 [00:33<00:12, 8631.41it/s] 74%|███████▎  | 294846/400000 [00:33<00:12, 8635.54it/s] 74%|███████▍  | 295711/400000 [00:33<00:12, 8639.35it/s] 74%|███████▍  | 296577/400000 [00:33<00:11, 8645.22it/s] 74%|███████▍  | 297448/400000 [00:34<00:11, 8664.19it/s] 75%|███████▍  | 298315/400000 [00:34<00:11, 8613.72it/s] 75%|███████▍  | 299177/400000 [00:34<00:11, 8584.55it/s] 75%|███████▌  | 300043/400000 [00:34<00:11, 8605.10it/s] 75%|███████▌  | 300919/400000 [00:34<00:11, 8650.68it/s] 75%|███████▌  | 301790/400000 [00:34<00:11, 8668.15it/s] 76%|███████▌  | 302657/400000 [00:34<00:11, 8566.86it/s] 76%|███████▌  | 303536/400000 [00:34<00:11, 8631.54it/s] 76%|███████▌  | 304416/400000 [00:34<00:11, 8676.98it/s] 76%|███████▋  | 305297/400000 [00:34<00:10, 8713.97it/s] 77%|███████▋  | 306180/400000 [00:35<00:10, 8746.18it/s] 77%|███████▋  | 307055/400000 [00:35<00:10, 8707.87it/s] 77%|███████▋  | 307941/400000 [00:35<00:10, 8750.27it/s] 77%|███████▋  | 308828/400000 [00:35<00:10, 8783.97it/s] 77%|███████▋  | 309711/400000 [00:35<00:10, 8797.33it/s] 78%|███████▊  | 310593/400000 [00:35<00:10, 8803.66it/s] 78%|███████▊  | 311474/400000 [00:35<00:10, 8732.62it/s] 78%|███████▊  | 312361/400000 [00:35<00:09, 8770.93it/s] 78%|███████▊  | 313239/400000 [00:35<00:09, 8682.27it/s] 79%|███████▊  | 314111/400000 [00:35<00:09, 8691.95it/s] 79%|███████▊  | 314981/400000 [00:36<00:09, 8677.82it/s] 79%|███████▉  | 315849/400000 [00:36<00:09, 8663.59it/s] 79%|███████▉  | 316716/400000 [00:36<00:09, 8503.33it/s] 79%|███████▉  | 317568/400000 [00:36<00:09, 8378.64it/s] 80%|███████▉  | 318452/400000 [00:36<00:09, 8509.16it/s] 80%|███████▉  | 319310/400000 [00:36<00:09, 8529.31it/s] 80%|████████  | 320164/400000 [00:36<00:09, 8498.70it/s] 80%|████████  | 321040/400000 [00:36<00:09, 8573.61it/s] 80%|████████  | 321906/400000 [00:36<00:09, 8597.95it/s] 81%|████████  | 322773/400000 [00:36<00:08, 8616.87it/s] 81%|████████  | 323644/400000 [00:37<00:08, 8643.11it/s] 81%|████████  | 324509/400000 [00:37<00:08, 8427.48it/s] 81%|████████▏ | 325354/400000 [00:37<00:08, 8430.58it/s] 82%|████████▏ | 326228/400000 [00:37<00:08, 8520.97it/s] 82%|████████▏ | 327103/400000 [00:37<00:08, 8587.10it/s] 82%|████████▏ | 327978/400000 [00:37<00:08, 8634.67it/s] 82%|████████▏ | 328843/400000 [00:37<00:08, 8377.37it/s] 82%|████████▏ | 329712/400000 [00:37<00:08, 8466.37it/s] 83%|████████▎ | 330596/400000 [00:37<00:08, 8573.18it/s] 83%|████████▎ | 331481/400000 [00:37<00:07, 8651.89it/s] 83%|████████▎ | 332365/400000 [00:38<00:07, 8706.91it/s] 83%|████████▎ | 333237/400000 [00:38<00:07, 8697.10it/s] 84%|████████▎ | 334123/400000 [00:38<00:07, 8743.10it/s] 84%|████████▍ | 335000/400000 [00:38<00:07, 8750.42it/s] 84%|████████▍ | 335884/400000 [00:38<00:07, 8777.04it/s] 84%|████████▍ | 336765/400000 [00:38<00:07, 8784.16it/s] 84%|████████▍ | 337644/400000 [00:38<00:07, 8723.78it/s] 85%|████████▍ | 338519/400000 [00:38<00:07, 8729.56it/s] 85%|████████▍ | 339399/400000 [00:38<00:06, 8749.24it/s] 85%|████████▌ | 340280/400000 [00:38<00:06, 8765.07it/s] 85%|████████▌ | 341157/400000 [00:39<00:06, 8616.25it/s] 86%|████████▌ | 342020/400000 [00:39<00:06, 8610.94it/s] 86%|████████▌ | 342882/400000 [00:39<00:06, 8604.98it/s] 86%|████████▌ | 343746/400000 [00:39<00:06, 8613.95it/s] 86%|████████▌ | 344608/400000 [00:39<00:06, 8610.61it/s] 86%|████████▋ | 345470/400000 [00:39<00:06, 8596.77it/s] 87%|████████▋ | 346332/400000 [00:39<00:06, 8603.55it/s] 87%|████████▋ | 347217/400000 [00:39<00:06, 8674.45it/s] 87%|████████▋ | 348103/400000 [00:39<00:05, 8727.47it/s] 87%|████████▋ | 348976/400000 [00:40<00:05, 8727.81it/s] 87%|████████▋ | 349849/400000 [00:40<00:05, 8711.36it/s] 88%|████████▊ | 350721/400000 [00:40<00:05, 8611.16it/s] 88%|████████▊ | 351593/400000 [00:40<00:05, 8643.23it/s] 88%|████████▊ | 352466/400000 [00:40<00:05, 8668.27it/s] 88%|████████▊ | 353343/400000 [00:40<00:05, 8698.38it/s] 89%|████████▊ | 354214/400000 [00:40<00:05, 8677.99it/s] 89%|████████▉ | 355082/400000 [00:40<00:05, 8559.85it/s] 89%|████████▉ | 355967/400000 [00:40<00:05, 8643.01it/s] 89%|████████▉ | 356850/400000 [00:40<00:04, 8697.60it/s] 89%|████████▉ | 357735/400000 [00:41<00:04, 8741.46it/s] 90%|████████▉ | 358618/400000 [00:41<00:04, 8767.06it/s] 90%|████████▉ | 359495/400000 [00:41<00:04, 8746.49it/s] 90%|█████████ | 360380/400000 [00:41<00:04, 8774.57it/s] 90%|█████████ | 361268/400000 [00:41<00:04, 8804.16it/s] 91%|█████████ | 362155/400000 [00:41<00:04, 8820.90it/s] 91%|█████████ | 363043/400000 [00:41<00:04, 8837.32it/s] 91%|█████████ | 363927/400000 [00:41<00:04, 8793.37it/s] 91%|█████████ | 364809/400000 [00:41<00:03, 8799.37it/s] 91%|█████████▏| 365690/400000 [00:41<00:03, 8748.15it/s] 92%|█████████▏| 366565/400000 [00:42<00:03, 8721.77it/s] 92%|█████████▏| 367438/400000 [00:42<00:03, 8721.14it/s] 92%|█████████▏| 368311/400000 [00:42<00:03, 8710.86it/s] 92%|█████████▏| 369193/400000 [00:42<00:03, 8741.81it/s] 93%|█████████▎| 370068/400000 [00:42<00:03, 8734.30it/s] 93%|█████████▎| 370942/400000 [00:42<00:03, 8617.62it/s] 93%|█████████▎| 371805/400000 [00:42<00:03, 8613.21it/s] 93%|█████████▎| 372676/400000 [00:42<00:03, 8641.63it/s] 93%|█████████▎| 373542/400000 [00:42<00:03, 8645.61it/s] 94%|█████████▎| 374421/400000 [00:42<00:02, 8685.79it/s] 94%|█████████▍| 375294/400000 [00:43<00:02, 8698.95it/s] 94%|█████████▍| 376165/400000 [00:43<00:02, 8700.76it/s] 94%|█████████▍| 377036/400000 [00:43<00:02, 8687.93it/s] 94%|█████████▍| 377911/400000 [00:43<00:02, 8703.67it/s] 95%|█████████▍| 378785/400000 [00:43<00:02, 8714.37it/s] 95%|█████████▍| 379657/400000 [00:43<00:02, 8598.26it/s] 95%|█████████▌| 380518/400000 [00:43<00:02, 8207.66it/s] 95%|█████████▌| 381381/400000 [00:43<00:02, 8329.91it/s] 96%|█████████▌| 382262/400000 [00:43<00:02, 8468.10it/s] 96%|█████████▌| 383137/400000 [00:43<00:01, 8550.61it/s] 96%|█████████▌| 384020/400000 [00:44<00:01, 8632.04it/s] 96%|█████████▌| 384905/400000 [00:44<00:01, 8696.28it/s] 96%|█████████▋| 385776/400000 [00:44<00:01, 8665.13it/s] 97%|█████████▋| 386644/400000 [00:44<00:01, 8379.52it/s] 97%|█████████▋| 387505/400000 [00:44<00:01, 8445.16it/s] 97%|█████████▋| 388375/400000 [00:44<00:01, 8517.14it/s] 97%|█████████▋| 389229/400000 [00:44<00:01, 8254.17it/s] 98%|█████████▊| 390058/400000 [00:44<00:01, 8255.84it/s] 98%|█████████▊| 390943/400000 [00:44<00:01, 8424.13it/s] 98%|█████████▊| 391829/400000 [00:44<00:00, 8549.09it/s] 98%|█████████▊| 392711/400000 [00:45<00:00, 8627.96it/s] 98%|█████████▊| 393598/400000 [00:45<00:00, 8698.62it/s] 99%|█████████▊| 394470/400000 [00:45<00:00, 8691.36it/s] 99%|█████████▉| 395345/400000 [00:45<00:00, 8708.39it/s] 99%|█████████▉| 396229/400000 [00:45<00:00, 8745.16it/s] 99%|█████████▉| 397117/400000 [00:45<00:00, 8782.23it/s] 99%|█████████▉| 397998/400000 [00:45<00:00, 8790.22it/s]100%|█████████▉| 398878/400000 [00:45<00:00, 8784.82it/s]100%|█████████▉| 399757/400000 [00:45<00:00, 8764.35it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8713.61it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2f90bdb518> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011307227168130943 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.01093591674913132 	 Accuracy: 56

  model saves at 56% accuracy 

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

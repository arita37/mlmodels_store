
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fe1e430f470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 19:12:37.299008
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 19:12:37.302716
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 19:12:37.305798
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 19:12:37.309036
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fe1dc65f438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351147.0312
Epoch 2/10

1/1 [==============================] - 0s 101ms/step - loss: 245741.5312
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 148146.0469
Epoch 4/10

1/1 [==============================] - 0s 107ms/step - loss: 80608.0547
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 43437.3828
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 25032.4805
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 15601.4492
Epoch 8/10

1/1 [==============================] - 0s 98ms/step - loss: 10470.0918
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 7500.7832
Epoch 10/10

1/1 [==============================] - 0s 106ms/step - loss: 5693.0312

  #### Inference Need return ypred, ytrue ######################### 
[[-2.13229251e+00 -1.35236585e+00  2.25620672e-01  8.96794021e-01
   1.51021433e+00 -3.64532441e-01 -1.55551970e+00 -1.84207034e+00
   3.21361385e-02  1.12669563e+00  1.27345991e+00 -5.60566127e-01
   3.99333417e-01 -8.03675115e-01  5.66709280e-01  1.46738768e-01
   5.43785453e-01 -2.57492363e-01 -4.15146410e-01 -4.44469512e-01
   4.01453912e-01  6.06185913e-01  8.39780033e-01 -2.99611956e-01
  -6.06630206e-01  6.50220394e-01 -2.28698850e-02 -1.29801714e+00
   1.21638310e+00 -1.03911519e-01  2.25567031e+00 -1.45148039e-01
   6.54628932e-01  1.07290137e+00 -1.42213035e+00  3.71256471e-01
   7.88267851e-01 -8.92852187e-01  1.81030810e-01 -1.18956208e+00
   2.66234189e-01  7.77151436e-02  9.09711123e-02 -2.16620490e-02
  -9.32937682e-01  1.28812611e+00 -9.89025176e-01 -7.71986485e-01
   2.03729892e+00 -8.22670400e-01 -1.97375804e-01  1.86539710e+00
   7.40147412e-01 -1.58315039e+00 -5.68257630e-01  8.09429526e-01
   1.44097853e+00 -1.11178607e-01 -3.52698267e-01 -8.81312251e-01
   1.90335107e+00  4.06630695e-01 -4.89470422e-01 -1.18095100e+00
  -6.86730683e-01 -6.33876562e-01 -2.57050920e+00 -9.17146325e-01
   1.22516620e+00 -2.33835077e+00 -4.97633576e-01 -1.90530449e-01
  -1.11578631e+00 -4.46715355e-02 -5.91655374e-01  4.34022367e-01
   5.23949981e-01 -6.20583177e-01 -1.63658082e-01 -3.16622615e-01
  -2.03527975e+00  1.23787498e+00  5.47608376e-01 -4.60448176e-01
  -4.87587392e-01 -8.52550805e-01  9.48422968e-01 -1.59589720e+00
  -1.04276395e+00  7.05338866e-02  6.72933459e-02 -1.76978469e-01
  -8.40348005e-02  7.93518960e-01 -1.09835076e+00  4.60091680e-01
   9.68449056e-01  8.46803665e-01 -9.41399336e-02 -3.63070995e-01
  -5.66384315e-01 -1.21272850e+00 -2.65126407e-01  3.32122773e-01
  -8.17227840e-01 -8.49324048e-01 -6.02156281e-01 -5.95704377e-01
   8.13260078e-02 -9.62303102e-01  4.93437082e-01 -1.58958220e+00
  -1.09922737e-02 -5.31153977e-01  9.66225713e-02  1.76767492e+00
  -1.24047720e+00 -7.16006517e-01 -1.10277200e+00 -2.15616250e+00
  -2.55643845e-01  9.71960068e+00  7.77614975e+00  1.04417000e+01
   7.80642748e+00  9.33583641e+00  8.91716862e+00  7.88953590e+00
   8.55456448e+00  8.93476772e+00  9.51436043e+00  8.57856846e+00
   9.68439484e+00  9.50980854e+00  9.91245842e+00  8.80205822e+00
   1.03561811e+01  1.01334181e+01  8.63773823e+00  8.74278164e+00
   9.32670593e+00  7.75294924e+00  8.93791962e+00  9.54829788e+00
   9.10474300e+00  1.00629187e+01  1.03401327e+01  8.39265537e+00
   9.72383499e+00  7.58480453e+00  7.55701494e+00  9.32889557e+00
   6.64593554e+00  9.25920963e+00  7.43435192e+00  9.51821232e+00
   6.70939779e+00  9.12878704e+00  8.61653709e+00  8.99563026e+00
   1.01136208e+01  7.51802540e+00  8.08473301e+00  8.90070248e+00
   8.87159920e+00  9.14020729e+00  9.10304546e+00  9.40044308e+00
   8.55950356e+00  9.31016254e+00  1.10157099e+01  1.07500877e+01
   7.25740910e+00  9.95485783e+00  8.91835499e+00  1.12346525e+01
   8.41141701e+00  9.49693108e+00  8.48645210e+00  1.05201254e+01
   2.61368036e-01  1.17471814e+00  9.70970988e-01  3.77996385e-01
   5.50975621e-01  1.03017211e+00  1.65664995e+00  1.72301483e+00
   2.61731386e-01  1.45388651e+00  8.38431895e-01  5.99477351e-01
   2.01746988e+00  2.48565853e-01  1.53201461e+00  6.86688542e-01
   9.01158094e-01  1.54327989e+00  1.46941984e+00  2.21422505e+00
   1.68972754e+00  2.70486951e-01  2.24098802e+00  6.20435238e-01
   3.20274830e+00  3.54302883e-01  1.79966259e+00  3.17238808e-01
   7.57032871e-01  3.24962795e-01  1.67274642e+00  5.30407608e-01
   1.22413445e+00  3.95402312e-01  1.61783159e-01  2.50004172e-01
   1.71516669e+00  1.80858374e-01  1.28879309e+00  4.34206784e-01
   2.44978619e+00  1.91903985e+00  1.97891116e-01  4.28267419e-01
   2.42735922e-01  2.62507200e+00  3.06681252e+00  9.11025405e-01
   5.31540751e-01  4.21076417e-01  1.87250400e+00  8.04445505e-01
   2.59552598e-01  1.56281948e+00  1.25079262e+00  8.01867723e-01
   4.81583893e-01  1.63075113e+00  1.82395244e+00  1.68958449e+00
   1.03685021e+00  4.45692062e-01  4.32657659e-01  2.17732906e+00
   3.19750023e+00  8.24791551e-01  1.81242442e+00  1.29573309e+00
   1.64928567e+00  2.07160258e+00  1.81068897e-01  1.75974524e+00
   4.39998567e-01  3.52311182e+00  1.72167373e+00  1.67398131e+00
   6.84012711e-01  2.42766476e+00  1.48578095e+00  7.25641131e-01
   7.33417153e-01  3.62809300e-01  4.15051877e-01  3.38192463e-01
   5.44937670e-01  2.75550604e-01  1.99242270e+00  9.13300157e-01
   3.29396129e-01  6.04530811e-01  1.84950888e-01  4.80871797e-01
   4.67768490e-01  2.63157725e+00  5.12221634e-01  2.52259135e-01
   1.03219104e+00  2.06431925e-01  1.83468437e+00  5.15062034e-01
   1.50878012e+00  2.51991367e+00  4.54324722e-01  8.37026119e-01
   2.13055325e+00  1.05318725e-01  2.32072949e-01  1.59953582e+00
   1.06190634e+00  9.24147367e-01  2.12713480e+00  4.28255975e-01
   1.20066631e+00  5.21627128e-01  4.82347429e-01  1.99859369e+00
   2.01642084e+00  7.45365918e-01  5.09411752e-01  1.64317906e+00
   1.24301910e-01  9.37880325e+00  1.06455746e+01  9.40382767e+00
   7.45142412e+00  8.32367039e+00  9.94996738e+00  7.97670269e+00
   9.84100819e+00  1.04916582e+01  8.76794338e+00  8.78910732e+00
   8.77378082e+00  9.04649067e+00  8.66615295e+00  9.97695541e+00
   1.07442961e+01  8.63306713e+00  8.13001919e+00  1.04036436e+01
   8.04736805e+00  8.53949451e+00  1.08237562e+01  1.20141296e+01
   8.82459354e+00  9.35714436e+00  8.92136478e+00  1.06885366e+01
   1.03939486e+01  9.10299778e+00  9.92572117e+00  1.01131430e+01
   9.14656544e+00  1.00106602e+01  8.24292755e+00  9.84319210e+00
   9.02605820e+00  1.01490393e+01  9.31967545e+00  7.42992210e+00
   9.30392742e+00  7.89402866e+00  1.06329155e+01  8.18601704e+00
   1.08943729e+01  8.07838440e+00  9.05075359e+00  1.04922686e+01
   8.33697033e+00  9.25290108e+00  8.43973637e+00  9.84146690e+00
   8.13259411e+00  9.36449051e+00  1.00580502e+01  1.04274778e+01
   6.95504999e+00  9.85764217e+00  7.54887152e+00  9.61489010e+00
  -9.46468163e+00 -8.94353199e+00  7.54022837e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 19:12:46.090192
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2083
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 19:12:46.094321
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8707.03
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 19:12:46.097447
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.5644
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 19:12:46.100527
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -778.776
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140607501061760
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140606273907848
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140606273908352
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140606273507512
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140606273508016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140606273508520

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fe1d84e2ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.562379
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.530286
grad_step = 000002, loss = 0.510870
grad_step = 000003, loss = 0.491129
grad_step = 000004, loss = 0.470821
grad_step = 000005, loss = 0.451124
grad_step = 000006, loss = 0.428803
grad_step = 000007, loss = 0.405298
grad_step = 000008, loss = 0.380027
grad_step = 000009, loss = 0.355459
grad_step = 000010, loss = 0.336335
grad_step = 000011, loss = 0.322063
grad_step = 000012, loss = 0.310027
grad_step = 000013, loss = 0.298453
grad_step = 000014, loss = 0.287996
grad_step = 000015, loss = 0.277564
grad_step = 000016, loss = 0.266330
grad_step = 000017, loss = 0.255028
grad_step = 000018, loss = 0.244829
grad_step = 000019, loss = 0.235448
grad_step = 000020, loss = 0.225338
grad_step = 000021, loss = 0.214250
grad_step = 000022, loss = 0.203680
grad_step = 000023, loss = 0.194690
grad_step = 000024, loss = 0.186717
grad_step = 000025, loss = 0.178566
grad_step = 000026, loss = 0.169861
grad_step = 000027, loss = 0.161309
grad_step = 000028, loss = 0.153498
grad_step = 000029, loss = 0.146178
grad_step = 000030, loss = 0.138897
grad_step = 000031, loss = 0.131495
grad_step = 000032, loss = 0.124108
grad_step = 000033, loss = 0.117089
grad_step = 000034, loss = 0.110661
grad_step = 000035, loss = 0.104710
grad_step = 000036, loss = 0.098899
grad_step = 000037, loss = 0.093120
grad_step = 000038, loss = 0.087692
grad_step = 000039, loss = 0.082630
grad_step = 000040, loss = 0.077635
grad_step = 000041, loss = 0.072774
grad_step = 000042, loss = 0.068134
grad_step = 000043, loss = 0.063708
grad_step = 000044, loss = 0.059551
grad_step = 000045, loss = 0.055684
grad_step = 000046, loss = 0.052046
grad_step = 000047, loss = 0.048562
grad_step = 000048, loss = 0.045247
grad_step = 000049, loss = 0.042168
grad_step = 000050, loss = 0.039326
grad_step = 000051, loss = 0.036622
grad_step = 000052, loss = 0.034013
grad_step = 000053, loss = 0.031603
grad_step = 000054, loss = 0.029419
grad_step = 000055, loss = 0.027412
grad_step = 000056, loss = 0.025522
grad_step = 000057, loss = 0.023737
grad_step = 000058, loss = 0.022101
grad_step = 000059, loss = 0.020582
grad_step = 000060, loss = 0.019134
grad_step = 000061, loss = 0.017782
grad_step = 000062, loss = 0.016530
grad_step = 000063, loss = 0.015367
grad_step = 000064, loss = 0.014290
grad_step = 000065, loss = 0.013286
grad_step = 000066, loss = 0.012338
grad_step = 000067, loss = 0.011461
grad_step = 000068, loss = 0.010663
grad_step = 000069, loss = 0.009906
grad_step = 000070, loss = 0.009192
grad_step = 000071, loss = 0.008547
grad_step = 000072, loss = 0.007959
grad_step = 000073, loss = 0.007409
grad_step = 000074, loss = 0.006903
grad_step = 000075, loss = 0.006441
grad_step = 000076, loss = 0.006018
grad_step = 000077, loss = 0.005631
grad_step = 000078, loss = 0.005278
grad_step = 000079, loss = 0.004956
grad_step = 000080, loss = 0.004666
grad_step = 000081, loss = 0.004404
grad_step = 000082, loss = 0.004164
grad_step = 000083, loss = 0.003948
grad_step = 000084, loss = 0.003755
grad_step = 000085, loss = 0.003581
grad_step = 000086, loss = 0.003423
grad_step = 000087, loss = 0.003284
grad_step = 000088, loss = 0.003161
grad_step = 000089, loss = 0.003048
grad_step = 000090, loss = 0.002947
grad_step = 000091, loss = 0.002859
grad_step = 000092, loss = 0.002780
grad_step = 000093, loss = 0.002709
grad_step = 000094, loss = 0.002647
grad_step = 000095, loss = 0.002591
grad_step = 000096, loss = 0.002543
grad_step = 000097, loss = 0.002499
grad_step = 000098, loss = 0.002461
grad_step = 000099, loss = 0.002427
grad_step = 000100, loss = 0.002397
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002370
grad_step = 000102, loss = 0.002348
grad_step = 000103, loss = 0.002328
grad_step = 000104, loss = 0.002310
grad_step = 000105, loss = 0.002295
grad_step = 000106, loss = 0.002281
grad_step = 000107, loss = 0.002268
grad_step = 000108, loss = 0.002258
grad_step = 000109, loss = 0.002248
grad_step = 000110, loss = 0.002240
grad_step = 000111, loss = 0.002232
grad_step = 000112, loss = 0.002226
grad_step = 000113, loss = 0.002219
grad_step = 000114, loss = 0.002214
grad_step = 000115, loss = 0.002210
grad_step = 000116, loss = 0.002207
grad_step = 000117, loss = 0.002202
grad_step = 000118, loss = 0.002196
grad_step = 000119, loss = 0.002191
grad_step = 000120, loss = 0.002189
grad_step = 000121, loss = 0.002186
grad_step = 000122, loss = 0.002183
grad_step = 000123, loss = 0.002178
grad_step = 000124, loss = 0.002173
grad_step = 000125, loss = 0.002170
grad_step = 000126, loss = 0.002168
grad_step = 000127, loss = 0.002166
grad_step = 000128, loss = 0.002163
grad_step = 000129, loss = 0.002159
grad_step = 000130, loss = 0.002155
grad_step = 000131, loss = 0.002150
grad_step = 000132, loss = 0.002147
grad_step = 000133, loss = 0.002143
grad_step = 000134, loss = 0.002141
grad_step = 000135, loss = 0.002138
grad_step = 000136, loss = 0.002137
grad_step = 000137, loss = 0.002139
grad_step = 000138, loss = 0.002145
grad_step = 000139, loss = 0.002154
grad_step = 000140, loss = 0.002155
grad_step = 000141, loss = 0.002140
grad_step = 000142, loss = 0.002121
grad_step = 000143, loss = 0.002115
grad_step = 000144, loss = 0.002124
grad_step = 000145, loss = 0.002130
grad_step = 000146, loss = 0.002124
grad_step = 000147, loss = 0.002110
grad_step = 000148, loss = 0.002102
grad_step = 000149, loss = 0.002104
grad_step = 000150, loss = 0.002110
grad_step = 000151, loss = 0.002110
grad_step = 000152, loss = 0.002102
grad_step = 000153, loss = 0.002093
grad_step = 000154, loss = 0.002088
grad_step = 000155, loss = 0.002088
grad_step = 000156, loss = 0.002090
grad_step = 000157, loss = 0.002091
grad_step = 000158, loss = 0.002089
grad_step = 000159, loss = 0.002084
grad_step = 000160, loss = 0.002078
grad_step = 000161, loss = 0.002072
grad_step = 000162, loss = 0.002069
grad_step = 000163, loss = 0.002067
grad_step = 000164, loss = 0.002066
grad_step = 000165, loss = 0.002067
grad_step = 000166, loss = 0.002068
grad_step = 000167, loss = 0.002070
grad_step = 000168, loss = 0.002074
grad_step = 000169, loss = 0.002080
grad_step = 000170, loss = 0.002087
grad_step = 000171, loss = 0.002093
grad_step = 000172, loss = 0.002091
grad_step = 000173, loss = 0.002081
grad_step = 000174, loss = 0.002061
grad_step = 000175, loss = 0.002043
grad_step = 000176, loss = 0.002034
grad_step = 000177, loss = 0.002034
grad_step = 000178, loss = 0.002041
grad_step = 000179, loss = 0.002047
grad_step = 000180, loss = 0.002049
grad_step = 000181, loss = 0.002044
grad_step = 000182, loss = 0.002034
grad_step = 000183, loss = 0.002022
grad_step = 000184, loss = 0.002012
grad_step = 000185, loss = 0.002006
grad_step = 000186, loss = 0.002005
grad_step = 000187, loss = 0.002005
grad_step = 000188, loss = 0.002007
grad_step = 000189, loss = 0.002009
grad_step = 000190, loss = 0.002011
grad_step = 000191, loss = 0.002014
grad_step = 000192, loss = 0.002016
grad_step = 000193, loss = 0.002012
grad_step = 000194, loss = 0.002002
grad_step = 000195, loss = 0.001994
grad_step = 000196, loss = 0.001987
grad_step = 000197, loss = 0.001979
grad_step = 000198, loss = 0.001968
grad_step = 000199, loss = 0.001957
grad_step = 000200, loss = 0.001950
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001947
grad_step = 000202, loss = 0.001947
grad_step = 000203, loss = 0.001949
grad_step = 000204, loss = 0.001954
grad_step = 000205, loss = 0.001961
grad_step = 000206, loss = 0.001969
grad_step = 000207, loss = 0.001975
grad_step = 000208, loss = 0.001990
grad_step = 000209, loss = 0.002012
grad_step = 000210, loss = 0.002048
grad_step = 000211, loss = 0.002063
grad_step = 000212, loss = 0.002049
grad_step = 000213, loss = 0.002010
grad_step = 000214, loss = 0.001944
grad_step = 000215, loss = 0.001912
grad_step = 000216, loss = 0.001923
grad_step = 000217, loss = 0.001964
grad_step = 000218, loss = 0.001976
grad_step = 000219, loss = 0.001927
grad_step = 000220, loss = 0.001882
grad_step = 000221, loss = 0.001880
grad_step = 000222, loss = 0.001906
grad_step = 000223, loss = 0.001931
grad_step = 000224, loss = 0.001918
grad_step = 000225, loss = 0.001889
grad_step = 000226, loss = 0.001865
grad_step = 000227, loss = 0.001865
grad_step = 000228, loss = 0.001880
grad_step = 000229, loss = 0.001881
grad_step = 000230, loss = 0.001871
grad_step = 000231, loss = 0.001858
grad_step = 000232, loss = 0.001857
grad_step = 000233, loss = 0.001866
grad_step = 000234, loss = 0.001865
grad_step = 000235, loss = 0.001861
grad_step = 000236, loss = 0.001844
grad_step = 000237, loss = 0.001834
grad_step = 000238, loss = 0.001832
grad_step = 000239, loss = 0.001836
grad_step = 000240, loss = 0.001842
grad_step = 000241, loss = 0.001845
grad_step = 000242, loss = 0.001849
grad_step = 000243, loss = 0.001846
grad_step = 000244, loss = 0.001846
grad_step = 000245, loss = 0.001841
grad_step = 000246, loss = 0.001840
grad_step = 000247, loss = 0.001842
grad_step = 000248, loss = 0.001856
grad_step = 000249, loss = 0.001884
grad_step = 000250, loss = 0.001930
grad_step = 000251, loss = 0.002013
grad_step = 000252, loss = 0.002047
grad_step = 000253, loss = 0.002044
grad_step = 000254, loss = 0.001922
grad_step = 000255, loss = 0.001837
grad_step = 000256, loss = 0.001853
grad_step = 000257, loss = 0.001877
grad_step = 000258, loss = 0.001854
grad_step = 000259, loss = 0.001816
grad_step = 000260, loss = 0.001843
grad_step = 000261, loss = 0.001898
grad_step = 000262, loss = 0.001841
grad_step = 000263, loss = 0.001785
grad_step = 000264, loss = 0.001797
grad_step = 000265, loss = 0.001828
grad_step = 000266, loss = 0.001818
grad_step = 000267, loss = 0.001781
grad_step = 000268, loss = 0.001791
grad_step = 000269, loss = 0.001813
grad_step = 000270, loss = 0.001791
grad_step = 000271, loss = 0.001770
grad_step = 000272, loss = 0.001778
grad_step = 000273, loss = 0.001790
grad_step = 000274, loss = 0.001784
grad_step = 000275, loss = 0.001764
grad_step = 000276, loss = 0.001759
grad_step = 000277, loss = 0.001770
grad_step = 000278, loss = 0.001775
grad_step = 000279, loss = 0.001766
grad_step = 000280, loss = 0.001754
grad_step = 000281, loss = 0.001753
grad_step = 000282, loss = 0.001762
grad_step = 000283, loss = 0.001769
grad_step = 000284, loss = 0.001773
grad_step = 000285, loss = 0.001779
grad_step = 000286, loss = 0.001801
grad_step = 000287, loss = 0.001829
grad_step = 000288, loss = 0.001869
grad_step = 000289, loss = 0.001874
grad_step = 000290, loss = 0.001853
grad_step = 000291, loss = 0.001804
grad_step = 000292, loss = 0.001755
grad_step = 000293, loss = 0.001734
grad_step = 000294, loss = 0.001744
grad_step = 000295, loss = 0.001769
grad_step = 000296, loss = 0.001785
grad_step = 000297, loss = 0.001782
grad_step = 000298, loss = 0.001764
grad_step = 000299, loss = 0.001739
grad_step = 000300, loss = 0.001725
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001725
grad_step = 000302, loss = 0.001735
grad_step = 000303, loss = 0.001741
grad_step = 000304, loss = 0.001741
grad_step = 000305, loss = 0.001731
grad_step = 000306, loss = 0.001721
grad_step = 000307, loss = 0.001716
grad_step = 000308, loss = 0.001716
grad_step = 000309, loss = 0.001723
grad_step = 000310, loss = 0.001722
grad_step = 000311, loss = 0.001723
grad_step = 000312, loss = 0.001713
grad_step = 000313, loss = 0.001705
grad_step = 000314, loss = 0.001697
grad_step = 000315, loss = 0.001693
grad_step = 000316, loss = 0.001691
grad_step = 000317, loss = 0.001692
grad_step = 000318, loss = 0.001695
grad_step = 000319, loss = 0.001695
grad_step = 000320, loss = 0.001698
grad_step = 000321, loss = 0.001701
grad_step = 000322, loss = 0.001714
grad_step = 000323, loss = 0.001723
grad_step = 000324, loss = 0.001756
grad_step = 000325, loss = 0.001774
grad_step = 000326, loss = 0.001825
grad_step = 000327, loss = 0.001849
grad_step = 000328, loss = 0.001892
grad_step = 000329, loss = 0.001866
grad_step = 000330, loss = 0.001848
grad_step = 000331, loss = 0.001858
grad_step = 000332, loss = 0.001713
grad_step = 000333, loss = 0.001702
grad_step = 000334, loss = 0.001834
grad_step = 000335, loss = 0.001835
grad_step = 000336, loss = 0.001761
grad_step = 000337, loss = 0.001725
grad_step = 000338, loss = 0.001756
grad_step = 000339, loss = 0.001723
grad_step = 000340, loss = 0.001717
grad_step = 000341, loss = 0.001716
grad_step = 000342, loss = 0.001705
grad_step = 000343, loss = 0.001719
grad_step = 000344, loss = 0.001695
grad_step = 000345, loss = 0.001683
grad_step = 000346, loss = 0.001700
grad_step = 000347, loss = 0.001701
grad_step = 000348, loss = 0.001718
grad_step = 000349, loss = 0.001686
grad_step = 000350, loss = 0.001648
grad_step = 000351, loss = 0.001663
grad_step = 000352, loss = 0.001666
grad_step = 000353, loss = 0.001677
grad_step = 000354, loss = 0.001662
grad_step = 000355, loss = 0.001627
grad_step = 000356, loss = 0.001651
grad_step = 000357, loss = 0.001662
grad_step = 000358, loss = 0.001670
grad_step = 000359, loss = 0.001652
grad_step = 000360, loss = 0.001619
grad_step = 000361, loss = 0.001634
grad_step = 000362, loss = 0.001631
grad_step = 000363, loss = 0.001640
grad_step = 000364, loss = 0.001641
grad_step = 000365, loss = 0.001606
grad_step = 000366, loss = 0.001623
grad_step = 000367, loss = 0.001637
grad_step = 000368, loss = 0.001633
grad_step = 000369, loss = 0.001620
grad_step = 000370, loss = 0.001590
grad_step = 000371, loss = 0.001599
grad_step = 000372, loss = 0.001597
grad_step = 000373, loss = 0.001600
grad_step = 000374, loss = 0.001621
grad_step = 000375, loss = 0.001585
grad_step = 000376, loss = 0.001586
grad_step = 000377, loss = 0.001580
grad_step = 000378, loss = 0.001583
grad_step = 000379, loss = 0.001600
grad_step = 000380, loss = 0.001595
grad_step = 000381, loss = 0.001606
grad_step = 000382, loss = 0.001598
grad_step = 000383, loss = 0.001611
grad_step = 000384, loss = 0.001653
grad_step = 000385, loss = 0.001734
grad_step = 000386, loss = 0.001863
grad_step = 000387, loss = 0.002010
grad_step = 000388, loss = 0.002018
grad_step = 000389, loss = 0.001818
grad_step = 000390, loss = 0.001620
grad_step = 000391, loss = 0.001618
grad_step = 000392, loss = 0.001726
grad_step = 000393, loss = 0.001768
grad_step = 000394, loss = 0.001656
grad_step = 000395, loss = 0.001553
grad_step = 000396, loss = 0.001621
grad_step = 000397, loss = 0.001697
grad_step = 000398, loss = 0.001642
grad_step = 000399, loss = 0.001556
grad_step = 000400, loss = 0.001562
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001606
grad_step = 000402, loss = 0.001621
grad_step = 000403, loss = 0.001590
grad_step = 000404, loss = 0.001544
grad_step = 000405, loss = 0.001547
grad_step = 000406, loss = 0.001586
grad_step = 000407, loss = 0.001591
grad_step = 000408, loss = 0.001551
grad_step = 000409, loss = 0.001555
grad_step = 000410, loss = 0.001573
grad_step = 000411, loss = 0.001581
grad_step = 000412, loss = 0.001540
grad_step = 000413, loss = 0.001524
grad_step = 000414, loss = 0.001523
grad_step = 000415, loss = 0.001531
grad_step = 000416, loss = 0.001540
grad_step = 000417, loss = 0.001529
grad_step = 000418, loss = 0.001533
grad_step = 000419, loss = 0.001544
grad_step = 000420, loss = 0.001553
grad_step = 000421, loss = 0.001538
grad_step = 000422, loss = 0.001531
grad_step = 000423, loss = 0.001519
grad_step = 000424, loss = 0.001512
grad_step = 000425, loss = 0.001501
grad_step = 000426, loss = 0.001495
grad_step = 000427, loss = 0.001488
grad_step = 000428, loss = 0.001488
grad_step = 000429, loss = 0.001499
grad_step = 000430, loss = 0.001504
grad_step = 000431, loss = 0.001519
grad_step = 000432, loss = 0.001522
grad_step = 000433, loss = 0.001546
grad_step = 000434, loss = 0.001554
grad_step = 000435, loss = 0.001581
grad_step = 000436, loss = 0.001558
grad_step = 000437, loss = 0.001533
grad_step = 000438, loss = 0.001487
grad_step = 000439, loss = 0.001467
grad_step = 000440, loss = 0.001477
grad_step = 000441, loss = 0.001495
grad_step = 000442, loss = 0.001506
grad_step = 000443, loss = 0.001484
grad_step = 000444, loss = 0.001468
grad_step = 000445, loss = 0.001463
grad_step = 000446, loss = 0.001467
grad_step = 000447, loss = 0.001469
grad_step = 000448, loss = 0.001463
grad_step = 000449, loss = 0.001459
grad_step = 000450, loss = 0.001460
grad_step = 000451, loss = 0.001474
grad_step = 000452, loss = 0.001481
grad_step = 000453, loss = 0.001499
grad_step = 000454, loss = 0.001479
grad_step = 000455, loss = 0.001474
grad_step = 000456, loss = 0.001458
grad_step = 000457, loss = 0.001463
grad_step = 000458, loss = 0.001472
grad_step = 000459, loss = 0.001491
grad_step = 000460, loss = 0.001500
grad_step = 000461, loss = 0.001510
grad_step = 000462, loss = 0.001496
grad_step = 000463, loss = 0.001486
grad_step = 000464, loss = 0.001455
grad_step = 000465, loss = 0.001442
grad_step = 000466, loss = 0.001430
grad_step = 000467, loss = 0.001433
grad_step = 000468, loss = 0.001434
grad_step = 000469, loss = 0.001431
grad_step = 000470, loss = 0.001425
grad_step = 000471, loss = 0.001418
grad_step = 000472, loss = 0.001421
grad_step = 000473, loss = 0.001430
grad_step = 000474, loss = 0.001459
grad_step = 000475, loss = 0.001457
grad_step = 000476, loss = 0.001475
grad_step = 000477, loss = 0.001436
grad_step = 000478, loss = 0.001416
grad_step = 000479, loss = 0.001399
grad_step = 000480, loss = 0.001396
grad_step = 000481, loss = 0.001404
grad_step = 000482, loss = 0.001418
grad_step = 000483, loss = 0.001435
grad_step = 000484, loss = 0.001439
grad_step = 000485, loss = 0.001441
grad_step = 000486, loss = 0.001442
grad_step = 000487, loss = 0.001440
grad_step = 000488, loss = 0.001447
grad_step = 000489, loss = 0.001462
grad_step = 000490, loss = 0.001508
grad_step = 000491, loss = 0.001510
grad_step = 000492, loss = 0.001537
grad_step = 000493, loss = 0.001428
grad_step = 000494, loss = 0.001376
grad_step = 000495, loss = 0.001386
grad_step = 000496, loss = 0.001422
grad_step = 000497, loss = 0.001451
grad_step = 000498, loss = 0.001390
grad_step = 000499, loss = 0.001362
grad_step = 000500, loss = 0.001384
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001400
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

  date_run                              2020-05-10 19:13:06.152280
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.185222
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 19:13:06.160772
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0717899
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 19:13:06.168488
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.125862
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 19:13:06.173949
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                -0.0908736
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
0   2020-05-10 19:12:37.299008  ...    mean_absolute_error
1   2020-05-10 19:12:37.302716  ...     mean_squared_error
2   2020-05-10 19:12:37.305798  ...  median_absolute_error
3   2020-05-10 19:12:37.309036  ...               r2_score
4   2020-05-10 19:12:46.090192  ...    mean_absolute_error
5   2020-05-10 19:12:46.094321  ...     mean_squared_error
6   2020-05-10 19:12:46.097447  ...  median_absolute_error
7   2020-05-10 19:12:46.100527  ...               r2_score
8   2020-05-10 19:13:06.152280  ...    mean_absolute_error
9   2020-05-10 19:13:06.160772  ...     mean_squared_error
10  2020-05-10 19:13:06.168488  ...  median_absolute_error
11  2020-05-10 19:13:06.173949  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:17, 127709.53it/s] 37%|███▋      | 3645440/9912422 [00:00<00:34, 182156.31it/s]9920512it [00:00, 31224686.97it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1088082.41it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 139187.43it/s]1654784it [00:00, 10182754.29it/s]                         
0it [00:00, ?it/s]8192it [00:00, 205669.35it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0af68dd780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a940219b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0af6894e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a94021eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0af6894e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0aa928dcf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0af68dd780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a94022048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0aa929ff28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0af68ddf98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a940219b0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9290a37208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=bd764172683ed1fe18266de2f34dccb90ebf924bcc6d48fe9ca70414edb21e61
  Stored in directory: /tmp/pip-ephem-wheel-cache-gicr3dun/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9286dc1048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2285568/17464789 [==>...........................] - ETA: 0s
 8429568/17464789 [=============>................] - ETA: 0s
16556032/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 19:14:31.220252: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 19:14:31.224617: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095159999 Hz
2020-05-10 19:14:31.224755: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558283fa39e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 19:14:31.224767: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8148 - accuracy: 0.4903
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6776 - accuracy: 0.4993
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7234 - accuracy: 0.4963
11000/25000 [============>.................] - ETA: 3s - loss: 7.6917 - accuracy: 0.4984
12000/25000 [=============>................] - ETA: 3s - loss: 7.6832 - accuracy: 0.4989
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6228 - accuracy: 0.5029
15000/25000 [=================>............] - ETA: 2s - loss: 7.6227 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6005 - accuracy: 0.5043
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5927 - accuracy: 0.5048
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6189 - accuracy: 0.5031
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6416 - accuracy: 0.5016
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6490 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6476 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 266us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 19:14:44.140592
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 19:14:44.140592  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 19:14:49.657754: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 19:14:49.662624: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095159999 Hz
2020-05-10 19:14:49.662775: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56515e983800 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 19:14:49.662788: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ff6b00d63c8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.8447 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.7711 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff68b00ef60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6390 - accuracy: 0.5018
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6871 - accuracy: 0.4987
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7455 - accuracy: 0.4949
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7050 - accuracy: 0.4975
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7024 - accuracy: 0.4977
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7402 - accuracy: 0.4952
11000/25000 [============>.................] - ETA: 3s - loss: 7.6847 - accuracy: 0.4988
12000/25000 [=============>................] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6631 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6392 - accuracy: 0.5018
15000/25000 [=================>............] - ETA: 2s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6700 - accuracy: 0.4998
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6881 - accuracy: 0.4986
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6914 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6910 - accuracy: 0.4984
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 7s 263us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ff668502080> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<26:56:03, 8.89kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<19:05:17, 12.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<13:24:58, 17.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<9:23:52, 25.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:33:40, 36.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.90M/862M [00:01<4:33:39, 51.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.8M/862M [00:01<3:10:17, 74.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.8M/862M [00:01<2:12:53, 106kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.5M/862M [00:01<1:32:29, 151kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.4M/862M [00:02<1:04:33, 215kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:02<45:01, 307kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<31:27, 437kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:02<21:59, 622kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<15:25, 882kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<10:49, 1.25MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<08:41, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<06:31, 2.06MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<10:43:20, 20.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<7:30:08, 29.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<5:16:26, 42.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<3:44:37, 59.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<2:37:45, 84.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:06<1:50:22, 121kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:21:56, 162kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<58:49, 226kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<41:29, 320kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<31:49, 416kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<23:38, 560kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:10<16:51, 784kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<14:52, 885kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<11:45, 1.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<08:33, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:14<09:04, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<07:40, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:14<05:38, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:16<07:03, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<07:36, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<05:58, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<06:16, 2.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<07:08, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<05:36, 2.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:18<04:04, 3.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<1:10:19, 183kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<50:30, 255kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<35:37, 361kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<27:52, 460kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<20:49, 616kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<14:49, 863kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<13:22, 953kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<11:57, 1.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:55, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:24<06:23, 1.99MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<11:53, 1.07MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:39, 1.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:04, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:02, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:18, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:27, 1.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:40, 2.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<1:28:12, 142kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<1:02:59, 199kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<44:19, 282kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<33:52, 369kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<24:58, 499kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<17:46, 701kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<15:20, 809kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<13:16, 935kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:55, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<08:55, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:31, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:31, 2.23MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<06:43, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:59, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:27, 2.75MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:57, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:48, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:52, 3.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:59, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:51, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:05, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:23, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:57, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:28, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:23, 2.73MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<7:35:51, 26.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<5:18:59, 37.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<3:44:36, 53.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<2:39:58, 74.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:52:29, 106kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<1:18:34, 151kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<1:04:51, 183kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<46:38, 254kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<32:53, 360kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<25:39, 460kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<20:21, 580kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<14:50, 794kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<12:14, 958kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<09:47, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<07:06, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<07:41, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<07:46, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:58, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<04:17, 2.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<18:54, 614kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<14:24, 805kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<10:21, 1.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<09:56, 1.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<08:07, 1.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<06:50, 1.68MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:58, 1.92MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:25, 2.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<05:46, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:22, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:02, 2.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:20, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<04:55, 2.30MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:43, 3.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:13, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<04:47, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:38, 3.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<04:45, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:36, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:09, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<04:44, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:35, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<05:07, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<05:50, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:38, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<05:01, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:38, 2.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<03:31, 3.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:00, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:37, 2.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:30, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<05:00, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:45, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:35, 2.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:20, 3.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<2:05:21, 86.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<1:28:47, 121kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<1:02:13, 173kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<43:34, 246kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<1:34:16, 114kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<1:08:11, 157kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<48:12, 222kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<35:19, 301kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<25:51, 411kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<18:17, 580kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<13:28, 785kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<7:16:18, 24.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<5:05:41, 34.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<3:33:11, 49.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<2:37:39, 66.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<1:52:28, 93.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<1:19:10, 133kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<56:50, 184kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<40:40, 257kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<28:40, 363kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<22:25, 463kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<17:53, 579kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<12:59, 797kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<09:11, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<13:24, 769kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<10:26, 986kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<07:33, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:40, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:27, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:44, 1.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:38, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:59, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:42, 2.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:58, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:32, 1.82MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:23, 2.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:40, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:19, 2.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:16, 3.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:37, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:16, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:14, 3.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:34, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:14, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<04:10, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:02, 3.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<1:12:11, 136kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<51:32, 191kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<36:14, 270kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<27:32, 354kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<21:16, 458kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<15:18, 636kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:50<10:46, 900kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<16:12, 598kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<12:19, 785kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:51, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<08:25, 1.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:51, 1.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:58, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:42, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:00, 1.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:44, 2.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:48, 1.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<05:18, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:12, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:27, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:04, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:02, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:19, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:01, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:17, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:01, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:04, 3.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:46, 3.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<5:55:32, 25.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<4:08:32, 37.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<2:54:55, 52.3kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<2:04:26, 73.5kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<1:27:30, 104kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<1:01:03, 149kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<53:42, 169kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<38:35, 235kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<27:09, 333kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<20:52, 431kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<16:36, 542kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<12:03, 746kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<08:31, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<09:38, 928kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:44, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<05:38, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:51, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:04, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:39, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:22, 2.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<05:47, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:01, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:45, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:31, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:08, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:07, 2.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:03, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:45, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:49, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:45, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<11:06, 773kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<08:43, 983kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<06:19, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:16, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:16, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:48, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:27, 2.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:32, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:31, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:03, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:39, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:07, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:03, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:55, 2.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<10:54, 762kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<08:33, 971kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:12, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:07, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:06, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:43, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:24, 2.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<12:04, 677kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:23, 870kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:44, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:28, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:19, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:48, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:26, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:58, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:15, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:41, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:04, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:59, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<02:52, 2.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<10:24, 759kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<08:09, 967kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:54, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:49, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:48, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:30, 1.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<03:14, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<11:38, 666kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<09:01, 859kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<06:30, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<06:12, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:11, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:49, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:18, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:42, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:43, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:41, 2.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<11:01, 685kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<08:32, 882kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<06:09, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<05:54, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:47, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:28, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:12, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<10:11, 727kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<07:57, 930kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:44, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:35, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:32, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:17, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<03:04, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<10:04, 722kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:51, 925kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<05:39, 1.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:30, 1.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:27, 1.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:12, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<03:01, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<10:45, 663kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:19, 856kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:58, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:41, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:33, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:13, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:02, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:46, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:06, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:01, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:37, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:19, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:31, 2.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:14, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:48, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:02, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:12, 3.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<09:44, 696kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<07:34, 894kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:28, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:14, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:11, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:00, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:51, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<08:18, 800kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:31, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:43, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:46, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:04, 1.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:31, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:54, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:03, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:13, 2.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:39, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:58, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:57, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:26, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:50, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:02, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:11, 2.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<09:09, 688kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<07:06, 885kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:07, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<04:55, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:47, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:40, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:37, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<09:00, 683kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:59, 880kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:02, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:50, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:40, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:33, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:34, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:59, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:35, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:05, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:25, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:42, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:57, 3.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<16:06, 365kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<11:55, 493kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:28, 691kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<07:09, 812kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<06:17, 923kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:40, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:20, 1.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<04:26, 1.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:42, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:44, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:11, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:29, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:45, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:59, 2.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<08:10, 686kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<06:18, 887kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:32, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:24, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<04:15, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:13, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:19, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:50, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:16, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:25, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:01, 2.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<3:39:31, 24.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<2:33:41, 35.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<1:46:47, 50.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<1:17:45, 68.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<54:51, 97.2kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<38:22, 138kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<27:48, 189kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<20:00, 263kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<14:03, 373kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<10:56, 475kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:48, 590kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:26, 805kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<04:32, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<09:23, 547kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:08, 718kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<05:06, 997kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:39, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:23, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:18, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:06, 1.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:24, 1.47MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<02:30, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:50, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:02, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:22, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:42, 2.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<15:47, 307kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<11:34, 419kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<08:11, 588kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<06:44, 710kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:45, 831kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:16, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<03:01, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<17:04, 276kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<12:28, 377kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<08:49, 531kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<07:07, 651kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<06:01, 771kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:27, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<03:08, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<07:12, 635kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<05:32, 824kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:59, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:45, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:37, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:45, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:58, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:15, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:46, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:03, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:22, 1.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:38, 1.66MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:02, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:29, 2.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:31, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:14, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:40, 2.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:05, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:21, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:52, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:20, 3.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<13:26, 309kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<09:52, 421kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<06:59, 591kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<05:43, 713kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<04:53, 835kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:37, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<02:33, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<14:31, 277kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<10:34, 379kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<07:27, 535kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<06:03, 653kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<05:07, 769kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:45, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:40, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:11, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:39, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:57, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:11, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:50, 2.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:19, 2.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<12:09, 308kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<08:55, 419kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<06:17, 590kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<05:09, 713kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<04:01, 912kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:53, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:47, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:45, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:07, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<05:20, 662kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<04:07, 855kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:58, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:48, 1.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:44, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:06, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:29, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<05:09, 659kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:57, 855kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:49, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:42, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:38, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:01, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:26, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<04:28, 727kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<03:29, 932kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:30, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:25, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:22, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:48, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:17, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:30, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:05, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:32, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:44, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:54, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:28, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:07, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:24, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:18, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<00:59, 2.96MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:19, 2.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:34, 1.85MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:15, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<00:54, 3.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<03:17, 864kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:35, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:51, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:55, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:57, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:29, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:04, 2.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:59, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:40, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:13, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:26, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:34, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:14, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:52, 2.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<08:18, 309kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<06:04, 421kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<04:16, 592kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:30, 711kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<03:00, 829kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:14, 1.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:33, 1.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:41, 658kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:50, 850kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:01, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:54, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:50, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:23, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:59, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:03, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:41, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:14, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:59, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<1:30:48, 24.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<1:03:22, 35.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<43:29, 49.9kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<31:43, 68.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<22:39, 95.1kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<15:53, 135kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<10:54, 192kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<09:50, 212kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<07:05, 293kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<04:57, 415kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<03:50, 525kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<03:06, 647kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:15, 884kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:34, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<06:14, 313kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<04:47, 406kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<03:25, 565kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:22, 798kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:41, 698kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:05, 898kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:29, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:25, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:23, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<01:04, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:45, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:37, 661kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:01, 854kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:27, 1.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:21, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:18, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:59, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:41, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<05:17, 303kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<03:51, 413kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<02:42, 581kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:11, 700kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:52, 818kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:23, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:57, 1.53MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:18, 632kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:46, 824kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:14, 1.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:10, 1.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:06, 1.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:50, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:35, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<04:23, 302kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<03:12, 411kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<02:14, 578kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:47, 700kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:32, 810kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:08, 1.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:46, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:48, 656kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:23, 848kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:59, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:54, 1.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:52, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:39, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:27, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:44, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:37, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:31, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:34, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:27, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:18, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:19, 689kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:01, 886kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:42, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:39, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:38, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:28, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:19, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:36, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:30, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:21, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:23, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:21, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:15, 2.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:18, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:21, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:22, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:13, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:08, 2.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:36, 690kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:28, 892kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:19, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:16, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:16, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:07, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:41, 409kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:30, 548kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:19, 767kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:14, 884kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:12, 986kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:09, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:05, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:10, 872kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:07, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.85MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 2.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:01, 305kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 415kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 903/400000 [00:00<00:44, 9027.04it/s]  0%|          | 1773/400000 [00:00<00:44, 8925.35it/s]  1%|          | 2683/400000 [00:00<00:44, 8976.81it/s]  1%|          | 3614/400000 [00:00<00:43, 9072.70it/s]  1%|          | 4489/400000 [00:00<00:44, 8972.88it/s]  1%|▏         | 5308/400000 [00:00<00:45, 8721.76it/s]  2%|▏         | 6167/400000 [00:00<00:45, 8680.09it/s]  2%|▏         | 7127/400000 [00:00<00:43, 8936.32it/s]  2%|▏         | 8095/400000 [00:00<00:42, 9145.72it/s]  2%|▏         | 8985/400000 [00:01<00:43, 9067.55it/s]  2%|▏         | 9947/400000 [00:01<00:42, 9226.01it/s]  3%|▎         | 10875/400000 [00:01<00:42, 9239.77it/s]  3%|▎         | 11828/400000 [00:01<00:41, 9323.54it/s]  3%|▎         | 12753/400000 [00:01<00:42, 9103.08it/s]  3%|▎         | 13660/400000 [00:01<00:42, 9042.23it/s]  4%|▎         | 14562/400000 [00:01<00:43, 8918.32it/s]  4%|▍         | 15453/400000 [00:01<00:43, 8779.24it/s]  4%|▍         | 16331/400000 [00:01<00:44, 8696.13it/s]  4%|▍         | 17201/400000 [00:01<00:44, 8621.30it/s]  5%|▍         | 18106/400000 [00:02<00:43, 8744.98it/s]  5%|▍         | 19091/400000 [00:02<00:42, 9049.23it/s]  5%|▌         | 20042/400000 [00:02<00:41, 9181.06it/s]  5%|▌         | 20963/400000 [00:02<00:41, 9043.78it/s]  5%|▌         | 21870/400000 [00:02<00:41, 9041.01it/s]  6%|▌         | 22825/400000 [00:02<00:41, 9185.63it/s]  6%|▌         | 23751/400000 [00:02<00:40, 9207.13it/s]  6%|▌         | 24739/400000 [00:02<00:39, 9397.73it/s]  6%|▋         | 25681/400000 [00:02<00:40, 9310.72it/s]  7%|▋         | 26678/400000 [00:02<00:39, 9497.57it/s]  7%|▋         | 27630/400000 [00:03<00:39, 9355.29it/s]  7%|▋         | 28635/400000 [00:03<00:38, 9551.76it/s]  7%|▋         | 29593/400000 [00:03<00:38, 9511.82it/s]  8%|▊         | 30546/400000 [00:03<00:38, 9483.67it/s]  8%|▊         | 31496/400000 [00:03<00:39, 9373.51it/s]  8%|▊         | 32435/400000 [00:03<00:39, 9214.03it/s]  8%|▊         | 33358/400000 [00:03<00:40, 9020.63it/s]  9%|▊         | 34262/400000 [00:03<00:40, 9009.07it/s]  9%|▉         | 35165/400000 [00:03<00:40, 9004.81it/s]  9%|▉         | 36067/400000 [00:03<00:41, 8862.74it/s]  9%|▉         | 36955/400000 [00:04<00:41, 8728.98it/s]  9%|▉         | 37837/400000 [00:04<00:41, 8755.10it/s] 10%|▉         | 38741/400000 [00:04<00:40, 8837.54it/s] 10%|▉         | 39626/400000 [00:04<00:42, 8458.71it/s] 10%|█         | 40529/400000 [00:04<00:41, 8615.89it/s] 10%|█         | 41401/400000 [00:04<00:41, 8645.32it/s] 11%|█         | 42350/400000 [00:04<00:40, 8879.97it/s] 11%|█         | 43271/400000 [00:04<00:39, 8974.06it/s] 11%|█         | 44193/400000 [00:04<00:39, 9043.64it/s] 11%|█▏        | 45100/400000 [00:04<00:39, 8916.16it/s] 11%|█▏        | 45994/400000 [00:05<00:40, 8819.69it/s] 12%|█▏        | 46878/400000 [00:05<00:40, 8787.46it/s] 12%|█▏        | 47758/400000 [00:05<00:40, 8758.28it/s] 12%|█▏        | 48635/400000 [00:05<00:40, 8700.17it/s] 12%|█▏        | 49535/400000 [00:05<00:39, 8785.62it/s] 13%|█▎        | 50415/400000 [00:05<00:39, 8751.62it/s] 13%|█▎        | 51291/400000 [00:05<00:40, 8667.11it/s] 13%|█▎        | 52206/400000 [00:05<00:39, 8805.65it/s] 13%|█▎        | 53090/400000 [00:05<00:39, 8813.92it/s] 14%|█▎        | 54009/400000 [00:06<00:38, 8921.53it/s] 14%|█▎        | 54902/400000 [00:06<00:38, 8856.55it/s] 14%|█▍        | 55789/400000 [00:06<00:40, 8579.99it/s] 14%|█▍        | 56650/400000 [00:06<00:40, 8528.25it/s] 14%|█▍        | 57505/400000 [00:06<00:40, 8479.65it/s] 15%|█▍        | 58373/400000 [00:06<00:40, 8536.87it/s] 15%|█▍        | 59231/400000 [00:06<00:39, 8548.12it/s] 15%|█▌        | 60087/400000 [00:06<00:39, 8501.42it/s] 15%|█▌        | 60964/400000 [00:06<00:39, 8578.85it/s] 15%|█▌        | 61823/400000 [00:06<00:39, 8537.35it/s] 16%|█▌        | 62684/400000 [00:07<00:39, 8556.94it/s] 16%|█▌        | 63543/400000 [00:07<00:39, 8565.36it/s] 16%|█▌        | 64400/400000 [00:07<00:40, 8297.45it/s] 16%|█▋        | 65232/400000 [00:07<00:41, 8026.61it/s] 17%|█▋        | 66058/400000 [00:07<00:41, 8092.94it/s] 17%|█▋        | 66923/400000 [00:07<00:40, 8250.48it/s] 17%|█▋        | 67758/400000 [00:07<00:40, 8279.65it/s] 17%|█▋        | 68613/400000 [00:07<00:39, 8357.78it/s] 17%|█▋        | 69468/400000 [00:07<00:39, 8412.55it/s] 18%|█▊        | 70338/400000 [00:07<00:38, 8493.13it/s] 18%|█▊        | 71189/400000 [00:08<00:38, 8444.81it/s] 18%|█▊        | 72039/400000 [00:08<00:38, 8459.12it/s] 18%|█▊        | 72899/400000 [00:08<00:38, 8498.93it/s] 18%|█▊        | 73750/400000 [00:08<00:38, 8387.72it/s] 19%|█▊        | 74626/400000 [00:08<00:38, 8493.80it/s] 19%|█▉        | 75477/400000 [00:08<00:38, 8482.23it/s] 19%|█▉        | 76359/400000 [00:08<00:37, 8579.30it/s] 19%|█▉        | 77285/400000 [00:08<00:36, 8771.60it/s] 20%|█▉        | 78173/400000 [00:08<00:36, 8801.38it/s] 20%|█▉        | 79055/400000 [00:08<00:36, 8744.71it/s] 20%|█▉        | 79931/400000 [00:09<00:36, 8733.61it/s] 20%|██        | 80821/400000 [00:09<00:36, 8780.66it/s] 20%|██        | 81718/400000 [00:09<00:36, 8833.90it/s] 21%|██        | 82629/400000 [00:09<00:35, 8913.44it/s] 21%|██        | 83560/400000 [00:09<00:35, 9028.29it/s] 21%|██        | 84464/400000 [00:09<00:34, 9030.71it/s] 21%|██▏       | 85368/400000 [00:09<00:35, 8970.88it/s] 22%|██▏       | 86266/400000 [00:09<00:35, 8882.96it/s] 22%|██▏       | 87156/400000 [00:09<00:35, 8888.02it/s] 22%|██▏       | 88046/400000 [00:09<00:35, 8877.23it/s] 22%|██▏       | 88934/400000 [00:10<00:35, 8822.16it/s] 22%|██▏       | 89817/400000 [00:10<00:35, 8694.01it/s] 23%|██▎       | 90711/400000 [00:10<00:35, 8765.13it/s] 23%|██▎       | 91589/400000 [00:10<00:35, 8622.55it/s] 23%|██▎       | 92476/400000 [00:10<00:35, 8694.16it/s] 23%|██▎       | 93347/400000 [00:10<00:35, 8697.99it/s] 24%|██▎       | 94218/400000 [00:10<00:35, 8619.86it/s] 24%|██▍       | 95085/400000 [00:10<00:35, 8633.09it/s] 24%|██▍       | 95979/400000 [00:10<00:34, 8720.47it/s] 24%|██▍       | 96859/400000 [00:10<00:34, 8742.15it/s] 24%|██▍       | 97735/400000 [00:11<00:34, 8746.22it/s] 25%|██▍       | 98654/400000 [00:11<00:33, 8874.36it/s] 25%|██▍       | 99552/400000 [00:11<00:33, 8903.29it/s] 25%|██▌       | 100452/400000 [00:11<00:33, 8931.14it/s] 25%|██▌       | 101353/400000 [00:11<00:33, 8953.60it/s] 26%|██▌       | 102250/400000 [00:11<00:33, 8957.35it/s] 26%|██▌       | 103149/400000 [00:11<00:33, 8967.12it/s] 26%|██▌       | 104070/400000 [00:11<00:32, 9037.00it/s] 26%|██▌       | 104974/400000 [00:11<00:33, 8822.16it/s] 26%|██▋       | 105868/400000 [00:11<00:33, 8854.31it/s] 27%|██▋       | 106782/400000 [00:12<00:32, 8935.88it/s] 27%|██▋       | 107685/400000 [00:12<00:32, 8961.13it/s] 27%|██▋       | 108582/400000 [00:12<00:33, 8615.12it/s] 27%|██▋       | 109449/400000 [00:12<00:33, 8631.13it/s] 28%|██▊       | 110343/400000 [00:12<00:33, 8719.11it/s] 28%|██▊       | 111217/400000 [00:12<00:33, 8568.08it/s] 28%|██▊       | 112114/400000 [00:12<00:33, 8683.99it/s] 28%|██▊       | 113035/400000 [00:12<00:32, 8832.40it/s] 28%|██▊       | 113921/400000 [00:12<00:33, 8598.77it/s] 29%|██▊       | 114819/400000 [00:13<00:32, 8708.10it/s] 29%|██▉       | 115726/400000 [00:13<00:32, 8812.55it/s] 29%|██▉       | 116640/400000 [00:13<00:31, 8905.75it/s] 29%|██▉       | 117533/400000 [00:13<00:32, 8660.28it/s] 30%|██▉       | 118402/400000 [00:13<00:33, 8383.53it/s] 30%|██▉       | 119311/400000 [00:13<00:32, 8582.69it/s] 30%|███       | 120215/400000 [00:13<00:32, 8714.77it/s] 30%|███       | 121090/400000 [00:13<00:32, 8705.61it/s] 30%|███       | 122000/400000 [00:13<00:31, 8818.88it/s] 31%|███       | 122912/400000 [00:13<00:31, 8905.34it/s] 31%|███       | 123805/400000 [00:14<00:31, 8780.98it/s] 31%|███       | 124721/400000 [00:14<00:30, 8891.22it/s] 31%|███▏      | 125628/400000 [00:14<00:30, 8942.32it/s] 32%|███▏      | 126524/400000 [00:14<00:30, 8922.78it/s] 32%|███▏      | 127417/400000 [00:14<00:30, 8848.09it/s] 32%|███▏      | 128304/400000 [00:14<00:30, 8854.02it/s] 32%|███▏      | 129235/400000 [00:14<00:30, 8985.38it/s] 33%|███▎      | 130160/400000 [00:14<00:29, 9062.44it/s] 33%|███▎      | 131067/400000 [00:14<00:30, 8921.57it/s] 33%|███▎      | 131980/400000 [00:14<00:29, 8982.90it/s] 33%|███▎      | 132880/400000 [00:15<00:29, 8930.80it/s] 33%|███▎      | 133796/400000 [00:15<00:29, 8996.28it/s] 34%|███▎      | 134704/400000 [00:15<00:29, 9020.21it/s] 34%|███▍      | 135607/400000 [00:15<00:29, 8973.87it/s] 34%|███▍      | 136505/400000 [00:15<00:29, 8817.92it/s] 34%|███▍      | 137388/400000 [00:15<00:29, 8766.43it/s] 35%|███▍      | 138266/400000 [00:15<00:29, 8759.99it/s] 35%|███▍      | 139153/400000 [00:15<00:29, 8792.25it/s] 35%|███▌      | 140045/400000 [00:15<00:29, 8827.95it/s] 35%|███▌      | 140929/400000 [00:15<00:29, 8822.92it/s] 35%|███▌      | 141812/400000 [00:16<00:29, 8768.07it/s] 36%|███▌      | 142701/400000 [00:16<00:29, 8803.04it/s] 36%|███▌      | 143594/400000 [00:16<00:29, 8838.30it/s] 36%|███▌      | 144478/400000 [00:16<00:29, 8555.05it/s] 36%|███▋      | 145336/400000 [00:16<00:30, 8232.22it/s] 37%|███▋      | 146216/400000 [00:16<00:30, 8392.67it/s] 37%|███▋      | 147084/400000 [00:16<00:29, 8474.55it/s] 37%|███▋      | 147983/400000 [00:16<00:29, 8620.48it/s] 37%|███▋      | 148907/400000 [00:16<00:28, 8797.08it/s] 37%|███▋      | 149801/400000 [00:17<00:28, 8837.46it/s] 38%|███▊      | 150687/400000 [00:17<00:28, 8702.65it/s] 38%|███▊      | 151560/400000 [00:17<00:28, 8688.69it/s] 38%|███▊      | 152431/400000 [00:17<00:28, 8615.05it/s] 38%|███▊      | 153294/400000 [00:17<00:28, 8593.16it/s] 39%|███▊      | 154177/400000 [00:17<00:28, 8660.69it/s] 39%|███▉      | 155070/400000 [00:17<00:28, 8739.53it/s] 39%|███▉      | 155969/400000 [00:17<00:27, 8812.21it/s] 39%|███▉      | 156851/400000 [00:17<00:27, 8796.76it/s] 39%|███▉      | 157732/400000 [00:17<00:27, 8763.92it/s] 40%|███▉      | 158609/400000 [00:18<00:27, 8749.15it/s] 40%|███▉      | 159485/400000 [00:18<00:27, 8633.88it/s] 40%|████      | 160349/400000 [00:18<00:27, 8600.27it/s] 40%|████      | 161210/400000 [00:18<00:27, 8542.74it/s] 41%|████      | 162065/400000 [00:18<00:27, 8498.67it/s] 41%|████      | 162924/400000 [00:18<00:27, 8525.61it/s] 41%|████      | 163793/400000 [00:18<00:27, 8571.45it/s] 41%|████      | 164651/400000 [00:18<00:27, 8471.32it/s] 41%|████▏     | 165499/400000 [00:18<00:27, 8463.02it/s] 42%|████▏     | 166346/400000 [00:18<00:27, 8457.89it/s] 42%|████▏     | 167224/400000 [00:19<00:27, 8551.15it/s] 42%|████▏     | 168095/400000 [00:19<00:26, 8596.61it/s] 42%|████▏     | 168983/400000 [00:19<00:26, 8678.63it/s] 42%|████▏     | 169906/400000 [00:19<00:26, 8836.82it/s] 43%|████▎     | 170791/400000 [00:19<00:26, 8557.43it/s] 43%|████▎     | 171708/400000 [00:19<00:26, 8731.57it/s] 43%|████▎     | 172596/400000 [00:19<00:25, 8774.93it/s] 43%|████▎     | 173476/400000 [00:19<00:25, 8778.88it/s] 44%|████▎     | 174356/400000 [00:19<00:25, 8705.59it/s] 44%|████▍     | 175262/400000 [00:19<00:25, 8808.67it/s] 44%|████▍     | 176144/400000 [00:20<00:25, 8789.37it/s] 44%|████▍     | 177047/400000 [00:20<00:25, 8859.50it/s] 44%|████▍     | 177934/400000 [00:20<00:25, 8817.94it/s] 45%|████▍     | 178822/400000 [00:20<00:25, 8836.45it/s] 45%|████▍     | 179707/400000 [00:20<00:25, 8795.87it/s] 45%|████▌     | 180600/400000 [00:20<00:24, 8835.05it/s] 45%|████▌     | 181484/400000 [00:20<00:24, 8834.17it/s] 46%|████▌     | 182368/400000 [00:20<00:24, 8818.75it/s] 46%|████▌     | 183251/400000 [00:20<00:24, 8739.78it/s] 46%|████▌     | 184135/400000 [00:20<00:24, 8769.11it/s] 46%|████▋     | 185013/400000 [00:21<00:24, 8771.93it/s] 46%|████▋     | 185906/400000 [00:21<00:24, 8815.77it/s] 47%|████▋     | 186833/400000 [00:21<00:23, 8946.76it/s] 47%|████▋     | 187732/400000 [00:21<00:23, 8956.57it/s] 47%|████▋     | 188629/400000 [00:21<00:23, 8902.52it/s] 47%|████▋     | 189520/400000 [00:21<00:23, 8807.26it/s] 48%|████▊     | 190407/400000 [00:21<00:23, 8824.49it/s] 48%|████▊     | 191290/400000 [00:21<00:23, 8777.74it/s] 48%|████▊     | 192170/400000 [00:21<00:23, 8784.34it/s] 48%|████▊     | 193049/400000 [00:21<00:23, 8724.29it/s] 48%|████▊     | 193926/400000 [00:22<00:23, 8734.13it/s] 49%|████▊     | 194800/400000 [00:22<00:23, 8721.70it/s] 49%|████▉     | 195673/400000 [00:22<00:23, 8625.71it/s] 49%|████▉     | 196536/400000 [00:22<00:25, 8101.74it/s] 49%|████▉     | 197422/400000 [00:22<00:24, 8313.45it/s] 50%|████▉     | 198302/400000 [00:22<00:23, 8453.61it/s] 50%|████▉     | 199187/400000 [00:22<00:23, 8567.26it/s] 50%|█████     | 200056/400000 [00:22<00:23, 8601.10it/s] 50%|█████     | 200922/400000 [00:22<00:23, 8616.41it/s] 50%|█████     | 201831/400000 [00:22<00:22, 8751.68it/s] 51%|█████     | 202708/400000 [00:23<00:22, 8661.96it/s] 51%|█████     | 203577/400000 [00:23<00:22, 8667.67it/s] 51%|█████     | 204445/400000 [00:23<00:22, 8615.31it/s] 51%|█████▏    | 205317/400000 [00:23<00:22, 8644.12it/s] 52%|█████▏    | 206205/400000 [00:23<00:22, 8711.51it/s] 52%|█████▏    | 207100/400000 [00:23<00:21, 8779.62it/s] 52%|█████▏    | 207979/400000 [00:23<00:21, 8774.46it/s] 52%|█████▏    | 208865/400000 [00:23<00:21, 8799.00it/s] 52%|█████▏    | 209746/400000 [00:23<00:21, 8783.37it/s] 53%|█████▎    | 210625/400000 [00:23<00:21, 8694.92it/s] 53%|█████▎    | 211495/400000 [00:24<00:21, 8668.34it/s] 53%|█████▎    | 212374/400000 [00:24<00:21, 8704.13it/s] 53%|█████▎    | 213293/400000 [00:24<00:21, 8842.39it/s] 54%|█████▎    | 214178/400000 [00:24<00:21, 8812.54it/s] 54%|█████▍    | 215095/400000 [00:24<00:20, 8914.55it/s] 54%|█████▍    | 215988/400000 [00:24<00:20, 8837.96it/s] 54%|█████▍    | 216876/400000 [00:24<00:20, 8848.86it/s] 54%|█████▍    | 217762/400000 [00:24<00:20, 8843.14it/s] 55%|█████▍    | 218647/400000 [00:24<00:20, 8843.62it/s] 55%|█████▍    | 219532/400000 [00:25<00:20, 8729.98it/s] 55%|█████▌    | 220406/400000 [00:25<00:20, 8717.89it/s] 55%|█████▌    | 221279/400000 [00:25<00:20, 8688.88it/s] 56%|█████▌    | 222149/400000 [00:25<00:20, 8623.83it/s] 56%|█████▌    | 223012/400000 [00:25<00:21, 8258.97it/s] 56%|█████▌    | 223878/400000 [00:25<00:21, 8374.99it/s] 56%|█████▌    | 224719/400000 [00:25<00:21, 8297.65it/s] 56%|█████▋    | 225589/400000 [00:25<00:20, 8413.97it/s] 57%|█████▋    | 226451/400000 [00:25<00:20, 8474.34it/s] 57%|█████▋    | 227315/400000 [00:25<00:20, 8522.82it/s] 57%|█████▋    | 228186/400000 [00:26<00:20, 8576.41it/s] 57%|█████▋    | 229045/400000 [00:26<00:20, 8351.04it/s] 57%|█████▋    | 229918/400000 [00:26<00:20, 8460.36it/s] 58%|█████▊    | 230802/400000 [00:26<00:19, 8569.17it/s] 58%|█████▊    | 231672/400000 [00:26<00:19, 8605.25it/s] 58%|█████▊    | 232568/400000 [00:26<00:19, 8706.72it/s] 58%|█████▊    | 233459/400000 [00:26<00:18, 8766.10it/s] 59%|█████▊    | 234337/400000 [00:26<00:18, 8748.61it/s] 59%|█████▉    | 235221/400000 [00:26<00:18, 8773.86it/s] 59%|█████▉    | 236099/400000 [00:26<00:18, 8720.82it/s] 59%|█████▉    | 236975/400000 [00:27<00:18, 8730.24it/s] 59%|█████▉    | 237852/400000 [00:27<00:18, 8740.34it/s] 60%|█████▉    | 238727/400000 [00:27<00:18, 8698.36it/s] 60%|█████▉    | 239598/400000 [00:27<00:19, 8442.15it/s] 60%|██████    | 240474/400000 [00:27<00:18, 8534.26it/s] 60%|██████    | 241363/400000 [00:27<00:18, 8637.76it/s] 61%|██████    | 242229/400000 [00:27<00:18, 8598.59it/s] 61%|██████    | 243090/400000 [00:27<00:18, 8568.29it/s] 61%|██████    | 244002/400000 [00:27<00:17, 8725.40it/s] 61%|██████    | 244876/400000 [00:27<00:17, 8709.64it/s] 61%|██████▏   | 245750/400000 [00:28<00:17, 8716.71it/s] 62%|██████▏   | 246638/400000 [00:28<00:17, 8763.21it/s] 62%|██████▏   | 247540/400000 [00:28<00:17, 8837.46it/s] 62%|██████▏   | 248425/400000 [00:28<00:17, 8702.22it/s] 62%|██████▏   | 249297/400000 [00:28<00:17, 8565.19it/s] 63%|██████▎   | 250213/400000 [00:28<00:17, 8734.35it/s] 63%|██████▎   | 251114/400000 [00:28<00:16, 8812.79it/s] 63%|██████▎   | 252012/400000 [00:28<00:16, 8860.87it/s] 63%|██████▎   | 252900/400000 [00:28<00:16, 8859.64it/s] 63%|██████▎   | 253794/400000 [00:28<00:16, 8881.57it/s] 64%|██████▎   | 254683/400000 [00:29<00:16, 8851.62it/s] 64%|██████▍   | 255585/400000 [00:29<00:16, 8899.21it/s] 64%|██████▍   | 256476/400000 [00:29<00:16, 8835.53it/s] 64%|██████▍   | 257363/400000 [00:29<00:16, 8844.53it/s] 65%|██████▍   | 258257/400000 [00:29<00:15, 8872.81it/s] 65%|██████▍   | 259145/400000 [00:29<00:16, 8789.19it/s] 65%|██████▌   | 260041/400000 [00:29<00:15, 8837.67it/s] 65%|██████▌   | 260926/400000 [00:29<00:15, 8805.07it/s] 65%|██████▌   | 261807/400000 [00:29<00:15, 8723.42it/s] 66%|██████▌   | 262680/400000 [00:29<00:15, 8723.16it/s] 66%|██████▌   | 263553/400000 [00:30<00:15, 8654.96it/s] 66%|██████▌   | 264432/400000 [00:30<00:15, 8692.59it/s] 66%|██████▋   | 265302/400000 [00:30<00:15, 8694.54it/s] 67%|██████▋   | 266178/400000 [00:30<00:15, 8712.15it/s] 67%|██████▋   | 267050/400000 [00:30<00:15, 8466.25it/s] 67%|██████▋   | 267914/400000 [00:30<00:15, 8516.29it/s] 67%|██████▋   | 268772/400000 [00:30<00:15, 8534.32it/s] 67%|██████▋   | 269664/400000 [00:30<00:15, 8644.80it/s] 68%|██████▊   | 270546/400000 [00:30<00:14, 8695.08it/s] 68%|██████▊   | 271417/400000 [00:30<00:14, 8657.19it/s] 68%|██████▊   | 272314/400000 [00:31<00:14, 8745.95it/s] 68%|██████▊   | 273216/400000 [00:31<00:14, 8823.50it/s] 69%|██████▊   | 274101/400000 [00:31<00:14, 8830.56it/s] 69%|██████▊   | 274985/400000 [00:31<00:14, 8388.42it/s] 69%|██████▉   | 275831/400000 [00:31<00:14, 8409.60it/s] 69%|██████▉   | 276705/400000 [00:31<00:14, 8505.67it/s] 69%|██████▉   | 277559/400000 [00:31<00:14, 8470.35it/s] 70%|██████▉   | 278449/400000 [00:31<00:14, 8593.06it/s] 70%|██████▉   | 279317/400000 [00:31<00:14, 8616.00it/s] 70%|███████   | 280180/400000 [00:32<00:14, 8547.20it/s] 70%|███████   | 281052/400000 [00:32<00:13, 8597.30it/s] 70%|███████   | 281929/400000 [00:32<00:13, 8645.30it/s] 71%|███████   | 282795/400000 [00:32<00:13, 8391.31it/s] 71%|███████   | 283662/400000 [00:32<00:13, 8471.02it/s] 71%|███████   | 284511/400000 [00:32<00:13, 8284.35it/s] 71%|███████▏  | 285342/400000 [00:32<00:13, 8286.52it/s] 72%|███████▏  | 286209/400000 [00:32<00:13, 8395.54it/s] 72%|███████▏  | 287067/400000 [00:32<00:13, 8449.85it/s] 72%|███████▏  | 287948/400000 [00:32<00:13, 8549.35it/s] 72%|███████▏  | 288804/400000 [00:33<00:13, 8520.45it/s] 72%|███████▏  | 289671/400000 [00:33<00:12, 8563.86it/s] 73%|███████▎  | 290528/400000 [00:33<00:12, 8546.87it/s] 73%|███████▎  | 291401/400000 [00:33<00:12, 8600.31it/s] 73%|███████▎  | 292273/400000 [00:33<00:12, 8635.37it/s] 73%|███████▎  | 293137/400000 [00:33<00:12, 8633.85it/s] 74%|███████▎  | 294001/400000 [00:33<00:12, 8479.36it/s] 74%|███████▎  | 294850/400000 [00:33<00:12, 8450.10it/s] 74%|███████▍  | 295697/400000 [00:33<00:12, 8453.73it/s] 74%|███████▍  | 296563/400000 [00:33<00:12, 8513.53it/s] 74%|███████▍  | 297431/400000 [00:34<00:11, 8562.76it/s] 75%|███████▍  | 298320/400000 [00:34<00:11, 8657.36it/s] 75%|███████▍  | 299187/400000 [00:34<00:11, 8659.25it/s] 75%|███████▌  | 300071/400000 [00:34<00:11, 8710.05it/s] 75%|███████▌  | 300943/400000 [00:34<00:11, 8370.41it/s] 75%|███████▌  | 301823/400000 [00:34<00:11, 8493.11it/s] 76%|███████▌  | 302675/400000 [00:34<00:11, 8412.16it/s] 76%|███████▌  | 303540/400000 [00:34<00:11, 8480.67it/s] 76%|███████▌  | 304399/400000 [00:34<00:11, 8512.19it/s] 76%|███████▋  | 305290/400000 [00:34<00:10, 8626.22it/s] 77%|███████▋  | 306158/400000 [00:35<00:10, 8639.51it/s] 77%|███████▋  | 307023/400000 [00:35<00:10, 8631.19it/s] 77%|███████▋  | 307908/400000 [00:35<00:10, 8693.55it/s] 77%|███████▋  | 308801/400000 [00:35<00:10, 8761.41it/s] 77%|███████▋  | 309678/400000 [00:35<00:10, 8747.35it/s] 78%|███████▊  | 310573/400000 [00:35<00:10, 8805.89it/s] 78%|███████▊  | 311454/400000 [00:35<00:10, 8619.06it/s] 78%|███████▊  | 312318/400000 [00:35<00:10, 8603.65it/s] 78%|███████▊  | 313180/400000 [00:35<00:10, 8586.28it/s] 79%|███████▊  | 314040/400000 [00:35<00:10, 8493.74it/s] 79%|███████▊  | 314919/400000 [00:36<00:09, 8578.26it/s] 79%|███████▉  | 315794/400000 [00:36<00:09, 8628.65it/s] 79%|███████▉  | 316658/400000 [00:36<00:09, 8593.11it/s] 79%|███████▉  | 317525/400000 [00:36<00:09, 8615.57it/s] 80%|███████▉  | 318407/400000 [00:36<00:09, 8673.67it/s] 80%|███████▉  | 319275/400000 [00:36<00:09, 8666.15it/s] 80%|████████  | 320142/400000 [00:36<00:09, 8656.58it/s] 80%|████████  | 321013/400000 [00:36<00:09, 8671.38it/s] 80%|████████  | 321881/400000 [00:36<00:09, 8670.37it/s] 81%|████████  | 322749/400000 [00:36<00:08, 8653.33it/s] 81%|████████  | 323629/400000 [00:37<00:08, 8696.04it/s] 81%|████████  | 324506/400000 [00:37<00:08, 8717.54it/s] 81%|████████▏ | 325378/400000 [00:37<00:08, 8533.67it/s] 82%|████████▏ | 326233/400000 [00:37<00:09, 7705.73it/s] 82%|████████▏ | 327093/400000 [00:37<00:09, 7953.23it/s] 82%|████████▏ | 327920/400000 [00:37<00:08, 8043.99it/s] 82%|████████▏ | 328776/400000 [00:37<00:08, 8189.00it/s] 82%|████████▏ | 329621/400000 [00:37<00:08, 8265.11it/s] 83%|████████▎ | 330473/400000 [00:37<00:08, 8338.02it/s] 83%|████████▎ | 331328/400000 [00:38<00:08, 8399.07it/s] 83%|████████▎ | 332191/400000 [00:38<00:08, 8466.46it/s] 83%|████████▎ | 333065/400000 [00:38<00:07, 8544.16it/s] 83%|████████▎ | 333922/400000 [00:38<00:07, 8522.88it/s] 84%|████████▎ | 334786/400000 [00:38<00:07, 8557.56it/s] 84%|████████▍ | 335668/400000 [00:38<00:07, 8632.64it/s] 84%|████████▍ | 336552/400000 [00:38<00:07, 8691.43it/s] 84%|████████▍ | 337429/400000 [00:38<00:07, 8712.11it/s] 85%|████████▍ | 338301/400000 [00:38<00:07, 8686.79it/s] 85%|████████▍ | 339171/400000 [00:38<00:07, 8673.55it/s] 85%|████████▌ | 340039/400000 [00:39<00:07, 8547.26it/s] 85%|████████▌ | 340899/400000 [00:39<00:06, 8562.21it/s] 85%|████████▌ | 341773/400000 [00:39<00:06, 8614.77it/s] 86%|████████▌ | 342676/400000 [00:39<00:06, 8732.49it/s] 86%|████████▌ | 343569/400000 [00:39<00:06, 8787.64it/s] 86%|████████▌ | 344449/400000 [00:39<00:06, 8751.87it/s] 86%|████████▋ | 345355/400000 [00:39<00:06, 8839.92it/s] 87%|████████▋ | 346261/400000 [00:39<00:06, 8902.30it/s] 87%|████████▋ | 347152/400000 [00:39<00:05, 8810.05it/s] 87%|████████▋ | 348034/400000 [00:39<00:05, 8728.47it/s] 87%|████████▋ | 348908/400000 [00:40<00:05, 8715.68it/s] 87%|████████▋ | 349780/400000 [00:40<00:05, 8670.97it/s] 88%|████████▊ | 350660/400000 [00:40<00:05, 8708.26it/s] 88%|████████▊ | 351532/400000 [00:40<00:05, 8659.24it/s] 88%|████████▊ | 352399/400000 [00:40<00:05, 8194.91it/s] 88%|████████▊ | 353224/400000 [00:40<00:05, 8209.41it/s] 89%|████████▊ | 354101/400000 [00:40<00:05, 8369.32it/s] 89%|████████▊ | 354942/400000 [00:40<00:05, 8373.09it/s] 89%|████████▉ | 355832/400000 [00:40<00:05, 8524.09it/s] 89%|████████▉ | 356695/400000 [00:40<00:05, 8552.84it/s] 89%|████████▉ | 357565/400000 [00:41<00:04, 8595.22it/s] 90%|████████▉ | 358456/400000 [00:41<00:04, 8684.54it/s] 90%|████████▉ | 359326/400000 [00:41<00:04, 8644.38it/s] 90%|█████████ | 360206/400000 [00:41<00:04, 8689.11it/s] 90%|█████████ | 361104/400000 [00:41<00:04, 8773.39it/s] 90%|█████████ | 361982/400000 [00:41<00:04, 8661.18it/s] 91%|█████████ | 362849/400000 [00:41<00:04, 8642.11it/s] 91%|█████████ | 363714/400000 [00:41<00:04, 8637.18it/s] 91%|█████████ | 364579/400000 [00:41<00:04, 8635.76it/s] 91%|█████████▏| 365443/400000 [00:41<00:04, 8625.89it/s] 92%|█████████▏| 366309/400000 [00:42<00:03, 8635.68it/s] 92%|█████████▏| 367187/400000 [00:42<00:03, 8676.00it/s] 92%|█████████▏| 368055/400000 [00:42<00:03, 8632.43it/s] 92%|█████████▏| 368919/400000 [00:42<00:03, 8454.24it/s] 92%|█████████▏| 369792/400000 [00:42<00:03, 8534.58it/s] 93%|█████████▎| 370670/400000 [00:42<00:03, 8606.15it/s] 93%|█████████▎| 371532/400000 [00:42<00:03, 8558.23it/s] 93%|█████████▎| 372389/400000 [00:42<00:03, 8444.49it/s] 93%|█████████▎| 373258/400000 [00:42<00:03, 8516.09it/s] 94%|█████████▎| 374111/400000 [00:42<00:03, 8457.69it/s] 94%|█████████▎| 374977/400000 [00:43<00:02, 8515.13it/s] 94%|█████████▍| 375838/400000 [00:43<00:02, 8541.13it/s] 94%|█████████▍| 376693/400000 [00:43<00:02, 8424.09it/s] 94%|█████████▍| 377537/400000 [00:43<00:02, 8310.67it/s] 95%|█████████▍| 378369/400000 [00:43<00:02, 8115.52it/s] 95%|█████████▍| 379247/400000 [00:43<00:02, 8303.05it/s] 95%|█████████▌| 380126/400000 [00:43<00:02, 8440.44it/s] 95%|█████████▌| 380992/400000 [00:43<00:02, 8503.26it/s] 95%|█████████▌| 381887/400000 [00:43<00:02, 8631.98it/s] 96%|█████████▌| 382752/400000 [00:44<00:01, 8629.30it/s] 96%|█████████▌| 383618/400000 [00:44<00:01, 8636.96it/s] 96%|█████████▌| 384494/400000 [00:44<00:01, 8672.14it/s] 96%|█████████▋| 385376/400000 [00:44<00:01, 8714.23it/s] 97%|█████████▋| 386248/400000 [00:44<00:01, 8554.46it/s] 97%|█████████▋| 387105/400000 [00:44<00:01, 8532.91it/s] 97%|█████████▋| 387962/400000 [00:44<00:01, 8543.61it/s] 97%|█████████▋| 388849/400000 [00:44<00:01, 8638.47it/s] 97%|█████████▋| 389714/400000 [00:44<00:01, 8634.62it/s] 98%|█████████▊| 390592/400000 [00:44<00:01, 8676.55it/s] 98%|█████████▊| 391476/400000 [00:45<00:00, 8723.18it/s] 98%|█████████▊| 392349/400000 [00:45<00:00, 8672.53it/s] 98%|█████████▊| 393217/400000 [00:45<00:00, 8672.43it/s] 99%|█████████▊| 394106/400000 [00:45<00:00, 8734.41it/s] 99%|█████████▊| 394980/400000 [00:45<00:00, 8679.56it/s] 99%|█████████▉| 395849/400000 [00:45<00:00, 8668.20it/s] 99%|█████████▉| 396731/400000 [00:45<00:00, 8712.42it/s] 99%|█████████▉| 397603/400000 [00:45<00:00, 8605.12it/s]100%|█████████▉| 398470/400000 [00:45<00:00, 8623.57it/s]100%|█████████▉| 399333/400000 [00:45<00:00, 8542.76it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8693.20it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff64fddb518> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011287046178045532 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011198492951217703 	 Accuracy: 52

  model saves at 52% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15803 out of table with 15797 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15803 out of table with 15797 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

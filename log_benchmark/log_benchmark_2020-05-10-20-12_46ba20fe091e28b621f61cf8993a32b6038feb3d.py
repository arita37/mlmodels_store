
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/46ba20fe091e28b621f61cf8993a32b6038feb3d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '46ba20fe091e28b621f61cf8993a32b6038feb3d', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/46ba20fe091e28b621f61cf8993a32b6038feb3d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/46ba20fe091e28b621f61cf8993a32b6038feb3d

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f444a7504a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 20:12:23.467066
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 20:12:23.470908
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 20:12:23.474171
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 20:12:23.477404
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4442aa04a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355168.3438
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 303891.1562
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 225946.2656
Epoch 4/10

1/1 [==============================] - 0s 101ms/step - loss: 151316.1562
Epoch 5/10

1/1 [==============================] - 0s 112ms/step - loss: 93240.4922
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 57263.8672
Epoch 7/10

1/1 [==============================] - 0s 88ms/step - loss: 36522.5859
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 24424.4102
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 17170.6895
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 12657.3877

  #### Inference Need return ypred, ytrue ######################### 
[[-6.10559821e-01 -2.19583809e-01 -4.22819376e-01  5.35063505e-01
  -4.52097178e-01  1.70321465e+00  3.58474791e-01  3.00764650e-01
  -7.48791397e-02 -6.36098146e-01  2.19230205e-02 -9.37328339e-01
   1.04045153e-01  4.97325480e-01  7.46987820e-01 -2.82661766e-01
   5.25540829e-01  9.89667416e-01  4.83585328e-01  1.03757873e-01
   1.15797424e+00  2.44760573e-01  1.80615246e-01  4.57993895e-02
   1.53602257e-01  1.61591649e-01 -3.94200385e-01  2.17615739e-01
   8.80736470e-01 -1.03401172e+00 -1.65579468e-03 -5.75899601e-01
   1.12999725e+00 -8.35379243e-01 -3.98180068e-01  2.97020733e-01
   1.11739433e+00 -1.96059942e-01 -2.56313652e-01  7.49213219e-01
   5.51607609e-01  8.25580657e-02 -5.47749758e-01 -1.97821632e-01
   5.06465256e-01 -1.81506246e-01 -1.04658437e+00 -3.43791246e-01
   2.45452702e-01 -2.22089723e-01  2.65256345e-01  7.55084872e-01
   9.36987102e-02  1.11525208e-01 -3.70001018e-01  2.04172865e-01
  -2.23363563e-02  5.95634997e-01 -4.42235507e-02 -1.90317646e-01
  -4.65657383e-01  4.80078816e-01 -1.70583457e-01  1.35144675e+00
  -6.42213464e-01 -6.09654486e-01  3.68878305e-01 -8.39829624e-01
  -6.40078783e-02 -4.16007936e-01  1.44274127e+00  5.54984868e-01
  -4.96067226e-01 -4.59136903e-01  1.95523024e-01  2.04572514e-01
   5.80501616e-01 -4.00093734e-01 -3.70939136e-01  4.81538713e-01
   5.51070809e-01  3.56834620e-01  9.41975892e-01  1.40808833e+00
  -2.05707386e-01 -6.59473896e-01 -3.04354608e-01 -3.15507382e-01
  -1.42038453e+00  3.11754286e-01 -9.26828980e-02 -3.41646641e-01
  -3.34998250e-01  3.55153829e-01  6.17193431e-02  2.36014396e-01
   1.54880285e-02  7.75025487e-01  1.49829412e+00  1.09360182e+00
  -7.84948230e-01  3.02307874e-01  6.88759923e-01 -1.10586792e-01
  -2.05506191e-01 -4.91882026e-01  8.66747379e-01  6.24963522e-01
  -1.25890821e-02  4.23152626e-01  2.85460800e-01  9.10384595e-01
  -5.26533246e-01  4.49305117e-01  2.28378773e-01  1.90149292e-01
  -6.87352598e-01 -3.61556143e-01 -9.58910167e-01 -1.81971669e-01
  -1.01619646e-01  5.87769985e+00  6.52282047e+00  5.89869499e+00
   5.25905228e+00  4.85382175e+00  6.19078779e+00  4.60069275e+00
   5.80660772e+00  5.73499680e+00  5.36011457e+00  4.73154736e+00
   4.93876648e+00  5.32678032e+00  6.05571079e+00  5.22386742e+00
   5.73034382e+00  4.97139311e+00  6.51923895e+00  4.65276241e+00
   5.32976961e+00  5.05561495e+00  5.72208738e+00  5.07236767e+00
   5.31912136e+00  4.29451323e+00  5.09429693e+00  6.10178995e+00
   5.42628813e+00  5.18070650e+00  5.94927454e+00  6.37737751e+00
   5.08052063e+00  6.09977865e+00  4.52703714e+00  5.63155365e+00
   4.56072044e+00  5.88140297e+00  6.26481724e+00  5.08456373e+00
   6.05206108e+00  4.37940311e+00  6.22994089e+00  5.22418499e+00
   5.39157295e+00  4.66785717e+00  6.18970156e+00  5.94706488e+00
   6.82227802e+00  5.36388731e+00  5.63593435e+00  4.21675777e+00
   5.58715105e+00  5.32792521e+00  5.70733166e+00  4.98847008e+00
   5.42633104e+00  6.24666691e+00  6.12208939e+00  5.37950039e+00
   3.34904075e-01  8.53880465e-01  1.46267223e+00  4.03120995e-01
   1.71074843e+00  8.31386268e-01  1.13388371e+00  2.06154060e+00
   1.27410901e+00  1.68038392e+00  9.80752826e-01  1.61269152e+00
   6.75585449e-01  1.29358053e+00  1.19693518e+00  7.09747314e-01
   8.34983528e-01  7.27540731e-01  5.80520093e-01  1.20784891e+00
   1.67808545e+00  7.45972097e-01  7.00370193e-01  5.80811262e-01
   1.49049568e+00  1.26955366e+00  1.23771322e+00  5.60954511e-01
   1.30934763e+00  2.03280640e+00  9.28825438e-01  3.09416950e-01
   7.66797543e-01  1.78228641e+00  1.40787411e+00  7.55220950e-01
   9.36199427e-01  7.19180942e-01  1.81584728e+00  5.65725327e-01
   1.48631048e+00  1.42229342e+00  1.37914252e+00  6.41778290e-01
   7.58065283e-01  1.66067576e+00  5.61218560e-01  1.44642353e+00
   1.28378081e+00  8.82268429e-01  6.55789614e-01  6.92120552e-01
   4.18930054e-01  7.95043468e-01  1.15161991e+00  2.11017656e+00
   7.70757318e-01  6.95125759e-01  8.12811196e-01  4.81203437e-01
   5.82576573e-01  2.02379131e+00  7.28915572e-01  1.14348686e+00
   1.17061925e+00  4.60397363e-01  1.42616391e+00  4.82664645e-01
   8.66967440e-01  1.21344924e+00  7.40008593e-01  1.66567385e+00
   1.12835026e+00  1.68385828e+00  1.29064965e+00  3.43943596e-01
   5.46100497e-01  1.04464674e+00  1.27431297e+00  2.81680584e-01
   1.22397327e+00  6.67304516e-01  4.10475671e-01  6.67750478e-01
   9.62839484e-01  1.57668257e+00  1.63757062e+00  6.63045585e-01
   1.20768011e+00  4.16530252e-01  1.71341515e+00  1.29461694e+00
   3.10026884e-01  5.41097462e-01  4.14027572e-01  1.03928423e+00
   7.13116407e-01  2.28952825e-01  3.22653294e-01  1.95898080e+00
   2.01804638e+00  1.38978374e+00  4.33483005e-01  1.47005808e+00
   1.60302877e+00  7.69586325e-01  1.51559091e+00  1.06800473e+00
   2.08196974e+00  5.09538114e-01  1.04398692e+00  9.20885324e-01
   1.76934612e+00  7.41057754e-01  1.05894303e+00  6.69904053e-01
   1.40806782e+00  1.13083196e+00  1.55689299e+00  1.07063198e+00
   3.19343209e-02  5.40372801e+00  5.34021044e+00  5.78302288e+00
   5.98328304e+00  6.22742319e+00  6.81125498e+00  5.67283583e+00
   6.52519178e+00  6.60869169e+00  6.78491879e+00  5.90582657e+00
   6.19606018e+00  5.74949646e+00  6.39017487e+00  5.43494844e+00
   6.60244751e+00  6.37042427e+00  6.31473255e+00  6.13672161e+00
   6.19627714e+00  5.82075500e+00  6.01983595e+00  5.94155264e+00
   5.51087952e+00  5.74574661e+00  5.67616558e+00  5.88716555e+00
   6.88935328e+00  6.14530373e+00  5.21681976e+00  6.32809258e+00
   5.93867588e+00  5.87286854e+00  6.02280188e+00  5.59468460e+00
   6.70358896e+00  5.51405764e+00  6.47248316e+00  6.58443642e+00
   6.66138887e+00  6.42816019e+00  6.42560959e+00  6.06337547e+00
   6.84276724e+00  5.97361135e+00  5.48849916e+00  5.86120558e+00
   5.84378767e+00  5.46518850e+00  5.84161711e+00  5.91097403e+00
   4.92036438e+00  6.62106037e+00  5.78409958e+00  6.16272736e+00
   5.38579369e+00  5.81616020e+00  5.80616665e+00  6.30149364e+00
  -6.41110849e+00 -7.44222736e+00  4.81380653e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 20:12:32.488544
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.346
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 20:12:32.492863
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    9298.7
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 20:12:32.495981
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.7347
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 20:12:32.499074
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -831.765
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139930611964840
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139929787550912
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139929787551416
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139929787146480
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139929787146984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139929787147488

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f443e923ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.650662
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.614359
grad_step = 000002, loss = 0.595894
grad_step = 000003, loss = 0.577865
grad_step = 000004, loss = 0.555910
grad_step = 000005, loss = 0.530397
grad_step = 000006, loss = 0.509432
grad_step = 000007, loss = 0.494092
grad_step = 000008, loss = 0.475051
grad_step = 000009, loss = 0.453075
grad_step = 000010, loss = 0.434590
grad_step = 000011, loss = 0.423237
grad_step = 000012, loss = 0.412820
grad_step = 000013, loss = 0.399924
grad_step = 000014, loss = 0.387481
grad_step = 000015, loss = 0.376242
grad_step = 000016, loss = 0.365925
grad_step = 000017, loss = 0.354770
grad_step = 000018, loss = 0.342231
grad_step = 000019, loss = 0.330018
grad_step = 000020, loss = 0.318849
grad_step = 000021, loss = 0.308544
grad_step = 000022, loss = 0.298410
grad_step = 000023, loss = 0.287564
grad_step = 000024, loss = 0.276895
grad_step = 000025, loss = 0.267223
grad_step = 000026, loss = 0.258095
grad_step = 000027, loss = 0.248652
grad_step = 000028, loss = 0.238797
grad_step = 000029, loss = 0.229573
grad_step = 000030, loss = 0.221283
grad_step = 000031, loss = 0.213128
grad_step = 000032, loss = 0.204585
grad_step = 000033, loss = 0.196171
grad_step = 000034, loss = 0.188433
grad_step = 000035, loss = 0.181009
grad_step = 000036, loss = 0.173551
grad_step = 000037, loss = 0.166256
grad_step = 000038, loss = 0.159365
grad_step = 000039, loss = 0.152720
grad_step = 000040, loss = 0.146111
grad_step = 000041, loss = 0.139709
grad_step = 000042, loss = 0.133680
grad_step = 000043, loss = 0.127862
grad_step = 000044, loss = 0.122109
grad_step = 000045, loss = 0.116535
grad_step = 000046, loss = 0.111245
grad_step = 000047, loss = 0.106169
grad_step = 000048, loss = 0.101243
grad_step = 000049, loss = 0.096512
grad_step = 000050, loss = 0.091975
grad_step = 000051, loss = 0.087577
grad_step = 000052, loss = 0.083305
grad_step = 000053, loss = 0.079221
grad_step = 000054, loss = 0.075352
grad_step = 000055, loss = 0.071603
grad_step = 000056, loss = 0.067969
grad_step = 000057, loss = 0.064532
grad_step = 000058, loss = 0.061236
grad_step = 000059, loss = 0.058032
grad_step = 000060, loss = 0.054984
grad_step = 000061, loss = 0.052066
grad_step = 000062, loss = 0.049235
grad_step = 000063, loss = 0.046558
grad_step = 000064, loss = 0.044009
grad_step = 000065, loss = 0.041533
grad_step = 000066, loss = 0.039187
grad_step = 000067, loss = 0.036951
grad_step = 000068, loss = 0.034785
grad_step = 000069, loss = 0.032737
grad_step = 000070, loss = 0.030786
grad_step = 000071, loss = 0.028920
grad_step = 000072, loss = 0.027161
grad_step = 000073, loss = 0.025478
grad_step = 000074, loss = 0.023871
grad_step = 000075, loss = 0.022359
grad_step = 000076, loss = 0.020923
grad_step = 000077, loss = 0.019563
grad_step = 000078, loss = 0.018283
grad_step = 000079, loss = 0.017071
grad_step = 000080, loss = 0.015931
grad_step = 000081, loss = 0.014861
grad_step = 000082, loss = 0.013850
grad_step = 000083, loss = 0.012904
grad_step = 000084, loss = 0.012018
grad_step = 000085, loss = 0.011192
grad_step = 000086, loss = 0.010420
grad_step = 000087, loss = 0.009701
grad_step = 000088, loss = 0.009033
grad_step = 000089, loss = 0.008413
grad_step = 000090, loss = 0.007840
grad_step = 000091, loss = 0.007308
grad_step = 000092, loss = 0.006820
grad_step = 000093, loss = 0.006369
grad_step = 000094, loss = 0.005958
grad_step = 000095, loss = 0.005581
grad_step = 000096, loss = 0.005239
grad_step = 000097, loss = 0.004926
grad_step = 000098, loss = 0.004642
grad_step = 000099, loss = 0.004378
grad_step = 000100, loss = 0.004135
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003905
grad_step = 000102, loss = 0.003695
grad_step = 000103, loss = 0.003507
grad_step = 000104, loss = 0.003343
grad_step = 000105, loss = 0.003200
grad_step = 000106, loss = 0.003074
grad_step = 000107, loss = 0.002964
grad_step = 000108, loss = 0.002869
grad_step = 000109, loss = 0.002789
grad_step = 000110, loss = 0.002718
grad_step = 000111, loss = 0.002657
grad_step = 000112, loss = 0.002592
grad_step = 000113, loss = 0.002524
grad_step = 000114, loss = 0.002450
grad_step = 000115, loss = 0.002382
grad_step = 000116, loss = 0.002328
grad_step = 000117, loss = 0.002290
grad_step = 000118, loss = 0.002266
grad_step = 000119, loss = 0.002248
grad_step = 000120, loss = 0.002235
grad_step = 000121, loss = 0.002220
grad_step = 000122, loss = 0.002202
grad_step = 000123, loss = 0.002176
grad_step = 000124, loss = 0.002146
grad_step = 000125, loss = 0.002116
grad_step = 000126, loss = 0.002090
grad_step = 000127, loss = 0.002071
grad_step = 000128, loss = 0.002056
grad_step = 000129, loss = 0.002045
grad_step = 000130, loss = 0.002034
grad_step = 000131, loss = 0.002025
grad_step = 000132, loss = 0.002016
grad_step = 000133, loss = 0.002010
grad_step = 000134, loss = 0.002005
grad_step = 000135, loss = 0.002027
grad_step = 000136, loss = 0.002125
grad_step = 000137, loss = 0.002351
grad_step = 000138, loss = 0.002187
grad_step = 000139, loss = 0.001988
grad_step = 000140, loss = 0.002119
grad_step = 000141, loss = 0.002168
grad_step = 000142, loss = 0.001996
grad_step = 000143, loss = 0.001937
grad_step = 000144, loss = 0.002055
grad_step = 000145, loss = 0.002014
grad_step = 000146, loss = 0.001902
grad_step = 000147, loss = 0.001999
grad_step = 000148, loss = 0.002044
grad_step = 000149, loss = 0.001896
grad_step = 000150, loss = 0.001908
grad_step = 000151, loss = 0.001994
grad_step = 000152, loss = 0.001911
grad_step = 000153, loss = 0.001861
grad_step = 000154, loss = 0.001924
grad_step = 000155, loss = 0.001927
grad_step = 000156, loss = 0.001857
grad_step = 000157, loss = 0.001820
grad_step = 000158, loss = 0.001860
grad_step = 000159, loss = 0.001896
grad_step = 000160, loss = 0.001859
grad_step = 000161, loss = 0.001819
grad_step = 000162, loss = 0.001824
grad_step = 000163, loss = 0.001883
grad_step = 000164, loss = 0.001984
grad_step = 000165, loss = 0.002263
grad_step = 000166, loss = 0.002805
grad_step = 000167, loss = 0.003566
grad_step = 000168, loss = 0.002031
grad_step = 000169, loss = 0.002502
grad_step = 000170, loss = 0.002759
grad_step = 000171, loss = 0.002033
grad_step = 000172, loss = 0.002902
grad_step = 000173, loss = 0.001930
grad_step = 000174, loss = 0.002462
grad_step = 000175, loss = 0.001954
grad_step = 000176, loss = 0.002362
grad_step = 000177, loss = 0.001988
grad_step = 000178, loss = 0.002181
grad_step = 000179, loss = 0.001952
grad_step = 000180, loss = 0.002141
grad_step = 000181, loss = 0.001926
grad_step = 000182, loss = 0.002062
grad_step = 000183, loss = 0.001875
grad_step = 000184, loss = 0.002039
grad_step = 000185, loss = 0.001853
grad_step = 000186, loss = 0.001990
grad_step = 000187, loss = 0.001823
grad_step = 000188, loss = 0.001977
grad_step = 000189, loss = 0.001814
grad_step = 000190, loss = 0.001928
grad_step = 000191, loss = 0.001795
grad_step = 000192, loss = 0.001906
grad_step = 000193, loss = 0.001795
grad_step = 000194, loss = 0.001867
grad_step = 000195, loss = 0.001781
grad_step = 000196, loss = 0.001844
grad_step = 000197, loss = 0.001786
grad_step = 000198, loss = 0.001814
grad_step = 000199, loss = 0.001768
grad_step = 000200, loss = 0.001793
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001770
grad_step = 000202, loss = 0.001768
grad_step = 000203, loss = 0.001757
grad_step = 000204, loss = 0.001743
grad_step = 000205, loss = 0.001756
grad_step = 000206, loss = 0.001726
grad_step = 000207, loss = 0.001739
grad_step = 000208, loss = 0.001711
grad_step = 000209, loss = 0.001723
grad_step = 000210, loss = 0.001706
grad_step = 000211, loss = 0.001704
grad_step = 000212, loss = 0.001692
grad_step = 000213, loss = 0.001691
grad_step = 000214, loss = 0.001680
grad_step = 000215, loss = 0.001675
grad_step = 000216, loss = 0.001676
grad_step = 000217, loss = 0.001661
grad_step = 000218, loss = 0.001658
grad_step = 000219, loss = 0.001647
grad_step = 000220, loss = 0.001648
grad_step = 000221, loss = 0.001643
grad_step = 000222, loss = 0.001646
grad_step = 000223, loss = 0.001658
grad_step = 000224, loss = 0.001698
grad_step = 000225, loss = 0.001786
grad_step = 000226, loss = 0.002025
grad_step = 000227, loss = 0.001971
grad_step = 000228, loss = 0.001921
grad_step = 000229, loss = 0.001732
grad_step = 000230, loss = 0.001795
grad_step = 000231, loss = 0.001768
grad_step = 000232, loss = 0.001677
grad_step = 000233, loss = 0.001832
grad_step = 000234, loss = 0.001881
grad_step = 000235, loss = 0.001650
grad_step = 000236, loss = 0.001607
grad_step = 000237, loss = 0.001687
grad_step = 000238, loss = 0.001753
grad_step = 000239, loss = 0.001792
grad_step = 000240, loss = 0.001789
grad_step = 000241, loss = 0.001917
grad_step = 000242, loss = 0.002009
grad_step = 000243, loss = 0.002026
grad_step = 000244, loss = 0.001721
grad_step = 000245, loss = 0.001568
grad_step = 000246, loss = 0.001676
grad_step = 000247, loss = 0.001800
grad_step = 000248, loss = 0.001715
grad_step = 000249, loss = 0.001566
grad_step = 000250, loss = 0.001587
grad_step = 000251, loss = 0.001674
grad_step = 000252, loss = 0.001631
grad_step = 000253, loss = 0.001566
grad_step = 000254, loss = 0.001575
grad_step = 000255, loss = 0.001589
grad_step = 000256, loss = 0.001544
grad_step = 000257, loss = 0.001517
grad_step = 000258, loss = 0.001556
grad_step = 000259, loss = 0.001567
grad_step = 000260, loss = 0.001526
grad_step = 000261, loss = 0.001494
grad_step = 000262, loss = 0.001506
grad_step = 000263, loss = 0.001506
grad_step = 000264, loss = 0.001478
grad_step = 000265, loss = 0.001460
grad_step = 000266, loss = 0.001468
grad_step = 000267, loss = 0.001472
grad_step = 000268, loss = 0.001457
grad_step = 000269, loss = 0.001455
grad_step = 000270, loss = 0.001514
grad_step = 000271, loss = 0.001723
grad_step = 000272, loss = 0.002460
grad_step = 000273, loss = 0.002333
grad_step = 000274, loss = 0.002160
grad_step = 000275, loss = 0.001533
grad_step = 000276, loss = 0.002164
grad_step = 000277, loss = 0.001763
grad_step = 000278, loss = 0.001816
grad_step = 000279, loss = 0.001845
grad_step = 000280, loss = 0.001683
grad_step = 000281, loss = 0.001800
grad_step = 000282, loss = 0.001665
grad_step = 000283, loss = 0.001694
grad_step = 000284, loss = 0.001681
grad_step = 000285, loss = 0.001613
grad_step = 000286, loss = 0.001657
grad_step = 000287, loss = 0.001573
grad_step = 000288, loss = 0.001597
grad_step = 000289, loss = 0.001566
grad_step = 000290, loss = 0.001501
grad_step = 000291, loss = 0.001560
grad_step = 000292, loss = 0.001444
grad_step = 000293, loss = 0.001535
grad_step = 000294, loss = 0.001430
grad_step = 000295, loss = 0.001461
grad_step = 000296, loss = 0.001454
grad_step = 000297, loss = 0.001425
grad_step = 000298, loss = 0.001455
grad_step = 000299, loss = 0.001410
grad_step = 000300, loss = 0.001454
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001466
grad_step = 000302, loss = 0.001546
grad_step = 000303, loss = 0.001717
grad_step = 000304, loss = 0.001951
grad_step = 000305, loss = 0.002013
grad_step = 000306, loss = 0.001625
grad_step = 000307, loss = 0.001453
grad_step = 000308, loss = 0.001704
grad_step = 000309, loss = 0.001587
grad_step = 000310, loss = 0.001476
grad_step = 000311, loss = 0.001654
grad_step = 000312, loss = 0.001454
grad_step = 000313, loss = 0.001508
grad_step = 000314, loss = 0.001579
grad_step = 000315, loss = 0.001415
grad_step = 000316, loss = 0.001457
grad_step = 000317, loss = 0.001549
grad_step = 000318, loss = 0.001422
grad_step = 000319, loss = 0.001399
grad_step = 000320, loss = 0.001489
grad_step = 000321, loss = 0.001452
grad_step = 000322, loss = 0.001375
grad_step = 000323, loss = 0.001406
grad_step = 000324, loss = 0.001448
grad_step = 000325, loss = 0.001410
grad_step = 000326, loss = 0.001364
grad_step = 000327, loss = 0.001391
grad_step = 000328, loss = 0.001420
grad_step = 000329, loss = 0.001382
grad_step = 000330, loss = 0.001357
grad_step = 000331, loss = 0.001379
grad_step = 000332, loss = 0.001388
grad_step = 000333, loss = 0.001366
grad_step = 000334, loss = 0.001353
grad_step = 000335, loss = 0.001367
grad_step = 000336, loss = 0.001376
grad_step = 000337, loss = 0.001358
grad_step = 000338, loss = 0.001346
grad_step = 000339, loss = 0.001350
grad_step = 000340, loss = 0.001357
grad_step = 000341, loss = 0.001353
grad_step = 000342, loss = 0.001345
grad_step = 000343, loss = 0.001345
grad_step = 000344, loss = 0.001353
grad_step = 000345, loss = 0.001363
grad_step = 000346, loss = 0.001370
grad_step = 000347, loss = 0.001392
grad_step = 000348, loss = 0.001441
grad_step = 000349, loss = 0.001554
grad_step = 000350, loss = 0.001648
grad_step = 000351, loss = 0.001778
grad_step = 000352, loss = 0.001628
grad_step = 000353, loss = 0.001470
grad_step = 000354, loss = 0.001374
grad_step = 000355, loss = 0.001415
grad_step = 000356, loss = 0.001451
grad_step = 000357, loss = 0.001409
grad_step = 000358, loss = 0.001422
grad_step = 000359, loss = 0.001428
grad_step = 000360, loss = 0.001353
grad_step = 000361, loss = 0.001358
grad_step = 000362, loss = 0.001419
grad_step = 000363, loss = 0.001389
grad_step = 000364, loss = 0.001330
grad_step = 000365, loss = 0.001332
grad_step = 000366, loss = 0.001371
grad_step = 000367, loss = 0.001376
grad_step = 000368, loss = 0.001333
grad_step = 000369, loss = 0.001316
grad_step = 000370, loss = 0.001339
grad_step = 000371, loss = 0.001360
grad_step = 000372, loss = 0.001354
grad_step = 000373, loss = 0.001328
grad_step = 000374, loss = 0.001319
grad_step = 000375, loss = 0.001328
grad_step = 000376, loss = 0.001334
grad_step = 000377, loss = 0.001326
grad_step = 000378, loss = 0.001310
grad_step = 000379, loss = 0.001303
grad_step = 000380, loss = 0.001308
grad_step = 000381, loss = 0.001315
grad_step = 000382, loss = 0.001317
grad_step = 000383, loss = 0.001312
grad_step = 000384, loss = 0.001311
grad_step = 000385, loss = 0.001326
grad_step = 000386, loss = 0.001367
grad_step = 000387, loss = 0.001444
grad_step = 000388, loss = 0.001608
grad_step = 000389, loss = 0.001758
grad_step = 000390, loss = 0.001969
grad_step = 000391, loss = 0.001674
grad_step = 000392, loss = 0.001390
grad_step = 000393, loss = 0.001305
grad_step = 000394, loss = 0.001458
grad_step = 000395, loss = 0.001516
grad_step = 000396, loss = 0.001349
grad_step = 000397, loss = 0.001368
grad_step = 000398, loss = 0.001456
grad_step = 000399, loss = 0.001345
grad_step = 000400, loss = 0.001341
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001428
grad_step = 000402, loss = 0.001345
grad_step = 000403, loss = 0.001301
grad_step = 000404, loss = 0.001363
grad_step = 000405, loss = 0.001347
grad_step = 000406, loss = 0.001300
grad_step = 000407, loss = 0.001311
grad_step = 000408, loss = 0.001332
grad_step = 000409, loss = 0.001310
grad_step = 000410, loss = 0.001279
grad_step = 000411, loss = 0.001294
grad_step = 000412, loss = 0.001321
grad_step = 000413, loss = 0.001318
grad_step = 000414, loss = 0.001292
grad_step = 000415, loss = 0.001270
grad_step = 000416, loss = 0.001270
grad_step = 000417, loss = 0.001285
grad_step = 000418, loss = 0.001300
grad_step = 000419, loss = 0.001314
grad_step = 000420, loss = 0.001321
grad_step = 000421, loss = 0.001330
grad_step = 000422, loss = 0.001337
grad_step = 000423, loss = 0.001344
grad_step = 000424, loss = 0.001342
grad_step = 000425, loss = 0.001342
grad_step = 000426, loss = 0.001333
grad_step = 000427, loss = 0.001323
grad_step = 000428, loss = 0.001302
grad_step = 000429, loss = 0.001281
grad_step = 000430, loss = 0.001262
grad_step = 000431, loss = 0.001251
grad_step = 000432, loss = 0.001251
grad_step = 000433, loss = 0.001258
grad_step = 000434, loss = 0.001267
grad_step = 000435, loss = 0.001271
grad_step = 000436, loss = 0.001270
grad_step = 000437, loss = 0.001264
grad_step = 000438, loss = 0.001256
grad_step = 000439, loss = 0.001248
grad_step = 000440, loss = 0.001243
grad_step = 000441, loss = 0.001239
grad_step = 000442, loss = 0.001238
grad_step = 000443, loss = 0.001238
grad_step = 000444, loss = 0.001239
grad_step = 000445, loss = 0.001241
grad_step = 000446, loss = 0.001245
grad_step = 000447, loss = 0.001253
grad_step = 000448, loss = 0.001266
grad_step = 000449, loss = 0.001292
grad_step = 000450, loss = 0.001332
grad_step = 000451, loss = 0.001409
grad_step = 000452, loss = 0.001479
grad_step = 000453, loss = 0.001584
grad_step = 000454, loss = 0.001542
grad_step = 000455, loss = 0.001473
grad_step = 000456, loss = 0.001327
grad_step = 000457, loss = 0.001263
grad_step = 000458, loss = 0.001305
grad_step = 000459, loss = 0.001333
grad_step = 000460, loss = 0.001294
grad_step = 000461, loss = 0.001234
grad_step = 000462, loss = 0.001263
grad_step = 000463, loss = 0.001314
grad_step = 000464, loss = 0.001282
grad_step = 000465, loss = 0.001241
grad_step = 000466, loss = 0.001254
grad_step = 000467, loss = 0.001273
grad_step = 000468, loss = 0.001255
grad_step = 000469, loss = 0.001222
grad_step = 000470, loss = 0.001221
grad_step = 000471, loss = 0.001244
grad_step = 000472, loss = 0.001256
grad_step = 000473, loss = 0.001247
grad_step = 000474, loss = 0.001230
grad_step = 000475, loss = 0.001224
grad_step = 000476, loss = 0.001229
grad_step = 000477, loss = 0.001234
grad_step = 000478, loss = 0.001234
grad_step = 000479, loss = 0.001226
grad_step = 000480, loss = 0.001217
grad_step = 000481, loss = 0.001210
grad_step = 000482, loss = 0.001207
grad_step = 000483, loss = 0.001210
grad_step = 000484, loss = 0.001219
grad_step = 000485, loss = 0.001235
grad_step = 000486, loss = 0.001260
grad_step = 000487, loss = 0.001302
grad_step = 000488, loss = 0.001379
grad_step = 000489, loss = 0.001474
grad_step = 000490, loss = 0.001646
grad_step = 000491, loss = 0.001686
grad_step = 000492, loss = 0.001711
grad_step = 000493, loss = 0.001437
grad_step = 000494, loss = 0.001224
grad_step = 000495, loss = 0.001231
grad_step = 000496, loss = 0.001373
grad_step = 000497, loss = 0.001396
grad_step = 000498, loss = 0.001260
grad_step = 000499, loss = 0.001246
grad_step = 000500, loss = 0.001337
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001284
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

  date_run                              2020-05-10 20:12:54.322955
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.221675
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 20:12:54.328492
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.116362
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 20:12:54.334746
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.141201
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 20:12:54.339923
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.768167
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
0   2020-05-10 20:12:23.467066  ...    mean_absolute_error
1   2020-05-10 20:12:23.470908  ...     mean_squared_error
2   2020-05-10 20:12:23.474171  ...  median_absolute_error
3   2020-05-10 20:12:23.477404  ...               r2_score
4   2020-05-10 20:12:32.488544  ...    mean_absolute_error
5   2020-05-10 20:12:32.492863  ...     mean_squared_error
6   2020-05-10 20:12:32.495981  ...  median_absolute_error
7   2020-05-10 20:12:32.499074  ...               r2_score
8   2020-05-10 20:12:54.322955  ...    mean_absolute_error
9   2020-05-10 20:12:54.328492  ...     mean_squared_error
10  2020-05-10 20:12:54.334746  ...  median_absolute_error
11  2020-05-10 20:12:54.339923  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3719168/9912422 [00:00<00:00, 37189144.16it/s]9920512it [00:00, 34395952.11it/s]                             
0it [00:00, ?it/s]32768it [00:00, 581619.24it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 480402.37it/s]1654784it [00:00, 12034225.15it/s]                         
0it [00:00, ?it/s]8192it [00:00, 176275.20it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0ec0fa780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb089844c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0ec0b1e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb089844da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0ec0b1e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb09eaaacf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0ec0faf98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb09e02beb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0ec0b1e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb09e02bf98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0ec0b1e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc49207e1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=684c7ecd42895f930ed09855110bb30cda815ad9615d553d783d3416d73d1a2e
  Stored in directory: /tmp/pip-ephem-wheel-cache-skbf_k2h/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc481d81e80> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2826240/17464789 [===>..........................] - ETA: 0s
11468800/17464789 [==================>...........] - ETA: 0s
16408576/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 20:14:17.994186: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 20:14:17.998024: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-10 20:14:17.998183: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562664b8f380 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 20:14:17.998199: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.2833 - accuracy: 0.5250
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4443 - accuracy: 0.5145 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6411 - accuracy: 0.5017
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7096 - accuracy: 0.4972
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6688 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6570 - accuracy: 0.5006
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6428 - accuracy: 0.5016
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 4s - loss: 7.6652 - accuracy: 0.5001
12000/25000 [=============>................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6713 - accuracy: 0.4997
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6611 - accuracy: 0.5004
15000/25000 [=================>............] - ETA: 2s - loss: 7.6319 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6450 - accuracy: 0.5014
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6496 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6489 - accuracy: 0.5012
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6375 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6425 - accuracy: 0.5016
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6413 - accuracy: 0.5017
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 9s 353us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 20:14:33.539330
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 20:14:33.539330  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 20:14:39.069503: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 20:14:39.074835: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-10 20:14:39.075076: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559b00d2eb30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 20:14:39.075091: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1405520be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 937ms/step - loss: 1.5222 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.4705 - val_crf_viterbi_accuracy: 0.1067

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1407445fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5823 - accuracy: 0.5055
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5041 - accuracy: 0.5106
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5465 - accuracy: 0.5078
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5242 - accuracy: 0.5093
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5497 - accuracy: 0.5076
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5644 - accuracy: 0.5067
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
11000/25000 [============>.................] - ETA: 4s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 3s - loss: 7.6500 - accuracy: 0.5011
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6478 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6492 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6506 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 9s 359us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f13a9329470> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:36:17, 9.35kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:09:24, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:45:45, 18.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:56:27, 26.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:14:31, 38.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<4:20:31, 54.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<3:01:15, 77.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<2:06:07, 111kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.2M/862M [00:01<1:27:55, 159kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:02<1:01:20, 226kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:02<42:47, 323kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:02<29:55, 459kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.8M/862M [00:02<20:57, 652kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:02<14:40, 926kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<10:20, 1.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<07:50, 1.72MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<07:22, 1.82MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<08:48, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:56, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<05:04, 2.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<07:32, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<06:41, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<04:59, 2.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:30, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<05:53, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:08<04:26, 2.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<06:16, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<07:05, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<05:37, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<06:03, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<05:34, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<04:14, 3.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<06:02, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<05:33, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<04:12, 3.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<03:40, 3.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<10:14:00, 21.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<7:09:56, 30.3kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:16<5:00:11, 43.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<3:37:06, 59.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<2:34:34, 83.8kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<1:48:43, 119kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<1:17:54, 165kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<57:15, 225kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<40:41, 316kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:20<28:32, 449kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<1:51:38, 115kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<1:19:26, 161kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:22<55:49, 229kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<41:56, 304kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<30:39, 416kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<21:44, 585kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<18:11, 697kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:00, 905kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<10:03, 1.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<10:02, 1.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<09:35, 1.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:14, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<05:15, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<08:37, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:18, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:23, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:41, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:15, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:42, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:59, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:29, 2.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:09, 2.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:46, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:40, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:11, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:58, 3.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:38, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:15, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:55, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:36, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:27, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:03, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:40, 3.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<12:07, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<09:43, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<07:03, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:46, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<07:53, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:04, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:23, 2.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<09:17, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:45, 1.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:41, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:07, 2.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<50:11, 238kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<37:55, 314kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<27:06, 439kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<19:10, 620kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<16:34, 716kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<12:57, 915kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:24, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<09:03, 1.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<09:14, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:04, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:06, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<12:51, 912kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<10:21, 1.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:31, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:44, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:14, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:26, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:39, 2.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<12:37, 917kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<10:11, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:27, 1.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:38, 1.51MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:02, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:11, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:27, 2.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<10:30, 1.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:40, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:23, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:52, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:36, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:52, 1.93MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:19, 2.62MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:05, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:23, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:08, 2.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:04, 3.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<11:19, 993kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<09:13, 1.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:43, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:03, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:13, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:37, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:35, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:29, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:07, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:44, 2.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<11:28, 962kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:16, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:48, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<07:03, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:29, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:46, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:16, 2.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:39, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:13, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:58, 2.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:03, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:48, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:39, 2.94MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:56, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:56, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:41, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<03:27, 3.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:42, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:13, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<03:56, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:00, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:03, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:46, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:28, 3.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:25, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:39, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:53, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:15, 2.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:48, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:41, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:55, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:38, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:21, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:03, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:40, 2.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<11:09, 926kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:00, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:32, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:45, 1.52MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:00, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:23, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:54, 2.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:50, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:44, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:18, 2.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:10, 1.96MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:57, 1.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:40, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:23, 2.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:24, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:21, 1.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:43, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:24, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:06, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:45, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:27, 2.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:18, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:31, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:08, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:58, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:37, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:30, 2.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:31, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:17, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:16, 2.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:21, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:17, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:10, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<03:04, 3.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:22, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:51, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:38, 2.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:34, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:18, 1.80MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:10, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:05, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:51, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:26, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:19, 2.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:27, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:12, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:09, 2.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:00, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<15:31, 603kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<11:55, 785kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<08:35, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:57, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:43, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:51, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<04:12, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:53, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:35, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:49, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:19, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:47, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:34, 2.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:25, 2.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:12, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:06, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<02:59, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:43, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:08, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:32, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:56, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:41, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<03:23, 2.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<09:44, 911kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:41, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:37, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:59, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:40, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:21, 2.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<15:31, 563kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<11:51, 736kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<08:32, 1.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<07:46, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<07:27, 1.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:38, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<04:03, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:07, 1.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:17, 1.62MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:56, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:33, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:09, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:02, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<02:54, 2.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:23, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:51, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:59, 1.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:14, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:37, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:20, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:09, 2.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:12, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:34, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:25, 2.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:06, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:53, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:53, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:49, 2.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<08:46, 932kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:05, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:09, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:17, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:34, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:22, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:09, 2.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:53, 904kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:09, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<05:13, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:20, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:21, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<03:09, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<08:46, 901kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:01, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:07, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:14, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:28, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:13, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:03, 2.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:05, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:27, 1.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:19, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<03:56, 1.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:39, 2.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:46, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:32, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:05, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:12, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:20, 3.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:41, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:06, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:04, 2.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:48, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:24, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:30, 2.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:32, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:02, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:00, 2.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:38, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:15, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:19, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:27, 2.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:53, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:33, 2.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<02:41, 2.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:24, 2.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:12, 2.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:26, 2.93MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:12, 2.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:54, 1.82MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:07, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:16, 3.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<06:46, 1.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:33, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:05, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<04:20, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:50, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:53, 2.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<03:28, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:02, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:13, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:20, 2.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<07:20, 934kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:56, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:18, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:25, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:40, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:40, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:38, 2.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<07:25, 904kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:57, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:20, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<04:34, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:30, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:31, 2.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<05:10, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:19, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:11, 2.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:40, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:06, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:11, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<02:18, 2.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:10, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:39, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:43, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:15, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:45, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:56, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:09, 2.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:29, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:10, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:22, 2.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:58, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:31, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:49, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:03, 3.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<06:33, 939kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:18, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:52, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:59, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:12, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:13, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:23, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:11, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:54, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:10, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:47, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:40, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:56, 3.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<06:15, 941kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<05:00, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:37, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<02:35, 2.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<1:58:31, 49.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<1:24:16, 69.0kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<59:13, 98.0kB/s]  .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<41:12, 140kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<33:32, 171kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<24:07, 238kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<16:59, 337kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:47<13:00, 436kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<09:45, 581kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:58, 811kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<06:01, 930kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<05:25, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:02, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:53, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:27, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:45, 1.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:45, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:03, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:25, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:42, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:57, 2.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:52, 920kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<04:41, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:23, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:32, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:37, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:46, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:00, 2.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:31, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:05, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:18, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:42, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:59, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:19, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<01:43, 2.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:39, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:27, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:51, 2.74MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:21, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:50, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:16, 2.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:39, 3.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:16, 944kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:16, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:07, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:12, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:23, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:36, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:52, 2.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:26, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:51, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:11, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:35, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:48, 992kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:28, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:21, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:23, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:16, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:31, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:34, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:46, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:01, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:20, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:41, 2.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:50, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:30, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:52, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:15, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:00, 2.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:33, 2.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:00, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:29, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<01:56, 2.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:25, 3.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:29, 1.74MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:15, 1.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:40, 2.58MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:04, 2.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:27, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:55, 2.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:23, 3.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:58, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:33, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:53, 2.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:11, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:29, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:56, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:24, 2.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:54, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:30, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:50, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:07, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:25, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:53, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:21, 2.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:42, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:20, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:43, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:00, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:21, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:51, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:20, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<04:06, 927kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:18, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:24, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:27, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:35, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:59, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:27, 2.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:03, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:52, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:24, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<01:44, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:02, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:09, 3.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:24, 1.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:04, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:31, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<01:06, 3.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<23:50, 145kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<17:30, 197kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<12:25, 277kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<08:37, 394kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<08:50, 383kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<06:34, 515kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<04:39, 722kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:55, 845kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:30, 943kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:38, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:52, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<04:00, 811kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<03:10, 1.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:18, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:15, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:19, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:46, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:15, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:04, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:30, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:49, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:54, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:03, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:35, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:09, 2.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:30, 1.96MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:07, 2.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:23, 2.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:38, 1.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:17, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<00:56, 3.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:27, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:05, 2.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:20, 2.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:34, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:15, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:54, 3.00MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:51, 943kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:18, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:40, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:42, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:48, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:22, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:59, 2.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:41, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:29, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:05, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:16, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:24, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:07, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:48, 3.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:21, 554kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:18, 728kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:21, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:08, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:45, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:17, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:22, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:30, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:09, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:50, 2.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:29, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:17, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:56, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:07, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:01, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:46, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:58, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:09, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:55, 2.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:39, 3.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:56, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:34, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:09, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:11, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:17, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:59, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:42, 2.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:16, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:05, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:48, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:55, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:03, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:35, 2.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:48, 951kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:27, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:02, 1.61MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:03, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:07, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:52, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:36, 2.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:39, 949kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:20, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:58, 1.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:58, 1.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:02, 1.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:47, 1.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:33, 2.59MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:54, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:45, 1.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:33, 2.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:24, 3.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:24, 975kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:17, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:58, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:40, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:33, 836kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:14, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:53, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:51, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:53, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:28, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:50, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:43, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:35, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:29, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:21, 2.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:38, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:33, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:24, 2.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:28, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:33, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:25, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:18, 2.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:33, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:24, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:17, 576kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:58, 756kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:40, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:34, 1.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:25, 1.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:17, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:24, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:13, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:20, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:18, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:13, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:14, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 2.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:22, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:17, 1.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:05, 2.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:28, 546kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:20, 721kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:13, 1.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:10, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:08, 854kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:06, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.82MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 950/400000 [00:00<00:42, 9497.27it/s]  0%|          | 1871/400000 [00:00<00:42, 9407.44it/s]  1%|          | 2632/400000 [00:00<00:45, 8781.05it/s]  1%|          | 3487/400000 [00:00<00:45, 8709.06it/s]  1%|          | 4376/400000 [00:00<00:45, 8759.51it/s]  1%|▏         | 5217/400000 [00:00<00:45, 8650.93it/s]  2%|▏         | 6098/400000 [00:00<00:45, 8696.78it/s]  2%|▏         | 6999/400000 [00:00<00:44, 8787.03it/s]  2%|▏         | 7906/400000 [00:00<00:44, 8867.60it/s]  2%|▏         | 8846/400000 [00:01<00:43, 9017.97it/s]  2%|▏         | 9780/400000 [00:01<00:42, 9109.66it/s]  3%|▎         | 10690/400000 [00:01<00:42, 9105.87it/s]  3%|▎         | 11625/400000 [00:01<00:42, 9175.45it/s]  3%|▎         | 12535/400000 [00:01<00:42, 9041.21it/s]  3%|▎         | 13434/400000 [00:01<00:43, 8907.69it/s]  4%|▎         | 14353/400000 [00:01<00:42, 8990.35it/s]  4%|▍         | 15275/400000 [00:01<00:42, 9055.60it/s]  4%|▍         | 16182/400000 [00:01<00:42, 9059.53it/s]  4%|▍         | 17087/400000 [00:01<00:46, 8180.52it/s]  4%|▍         | 17921/400000 [00:02<00:49, 7761.49it/s]  5%|▍         | 18763/400000 [00:02<00:47, 7946.94it/s]  5%|▍         | 19626/400000 [00:02<00:46, 8139.57it/s]  5%|▌         | 20523/400000 [00:02<00:45, 8371.10it/s]  5%|▌         | 21482/400000 [00:02<00:43, 8700.59it/s]  6%|▌         | 22425/400000 [00:02<00:42, 8905.09it/s]  6%|▌         | 23378/400000 [00:02<00:41, 9082.71it/s]  6%|▌         | 24320/400000 [00:02<00:40, 9180.49it/s]  6%|▋         | 25243/400000 [00:02<00:42, 8887.28it/s]  7%|▋         | 26188/400000 [00:02<00:41, 9048.77it/s]  7%|▋         | 27152/400000 [00:03<00:40, 9217.89it/s]  7%|▋         | 28089/400000 [00:03<00:40, 9260.61it/s]  7%|▋         | 29018/400000 [00:03<00:40, 9244.32it/s]  7%|▋         | 29945/400000 [00:03<00:40, 9071.71it/s]  8%|▊         | 30855/400000 [00:03<00:40, 9050.17it/s]  8%|▊         | 31762/400000 [00:03<00:40, 9046.46it/s]  8%|▊         | 32668/400000 [00:03<00:41, 8934.45it/s]  8%|▊         | 33563/400000 [00:03<00:41, 8781.73it/s]  9%|▊         | 34443/400000 [00:03<00:43, 8492.53it/s]  9%|▉         | 35296/400000 [00:04<00:43, 8425.08it/s]  9%|▉         | 36141/400000 [00:04<00:43, 8269.80it/s]  9%|▉         | 36971/400000 [00:04<00:44, 8175.06it/s]  9%|▉         | 37879/400000 [00:04<00:42, 8425.96it/s] 10%|▉         | 38744/400000 [00:04<00:42, 8491.44it/s] 10%|▉         | 39649/400000 [00:04<00:41, 8651.13it/s] 10%|█         | 40563/400000 [00:04<00:40, 8791.25it/s] 10%|█         | 41445/400000 [00:04<00:41, 8643.30it/s] 11%|█         | 42373/400000 [00:04<00:40, 8823.15it/s] 11%|█         | 43260/400000 [00:04<00:40, 8836.77it/s] 11%|█         | 44203/400000 [00:05<00:39, 9006.19it/s] 11%|█▏        | 45106/400000 [00:05<00:39, 8892.76it/s] 11%|█▏        | 45997/400000 [00:05<00:39, 8874.60it/s] 12%|█▏        | 46886/400000 [00:05<00:39, 8852.33it/s] 12%|█▏        | 47773/400000 [00:05<00:40, 8642.75it/s] 12%|█▏        | 48676/400000 [00:05<00:40, 8753.42it/s] 12%|█▏        | 49600/400000 [00:05<00:39, 8893.14it/s] 13%|█▎        | 50491/400000 [00:05<00:41, 8516.31it/s] 13%|█▎        | 51348/400000 [00:05<00:41, 8438.82it/s] 13%|█▎        | 52231/400000 [00:05<00:40, 8551.43it/s] 13%|█▎        | 53192/400000 [00:06<00:39, 8843.05it/s] 14%|█▎        | 54100/400000 [00:06<00:38, 8911.14it/s] 14%|█▍        | 55009/400000 [00:06<00:38, 8962.53it/s] 14%|█▍        | 55934/400000 [00:06<00:38, 9044.55it/s] 14%|█▍        | 56841/400000 [00:06<00:38, 8899.04it/s] 14%|█▍        | 57793/400000 [00:06<00:37, 9075.31it/s] 15%|█▍        | 58703/400000 [00:06<00:37, 9078.48it/s] 15%|█▍        | 59647/400000 [00:06<00:37, 9182.16it/s] 15%|█▌        | 60567/400000 [00:06<00:37, 9112.19it/s] 15%|█▌        | 61480/400000 [00:06<00:39, 8633.10it/s] 16%|█▌        | 62358/400000 [00:07<00:38, 8673.98it/s] 16%|█▌        | 63277/400000 [00:07<00:38, 8819.63it/s] 16%|█▌        | 64163/400000 [00:07<00:38, 8829.36it/s] 16%|█▋        | 65049/400000 [00:07<00:38, 8804.96it/s] 16%|█▋        | 65932/400000 [00:07<00:38, 8684.85it/s] 17%|█▋        | 66888/400000 [00:07<00:37, 8929.15it/s] 17%|█▋        | 67807/400000 [00:07<00:36, 9004.90it/s] 17%|█▋        | 68725/400000 [00:07<00:36, 9056.18it/s] 17%|█▋        | 69633/400000 [00:07<00:36, 9047.57it/s] 18%|█▊        | 70548/400000 [00:07<00:36, 9077.05it/s] 18%|█▊        | 71518/400000 [00:08<00:35, 9254.67it/s] 18%|█▊        | 72467/400000 [00:08<00:35, 9323.88it/s] 18%|█▊        | 73401/400000 [00:08<00:35, 9217.54it/s] 19%|█▊        | 74324/400000 [00:08<00:36, 8977.08it/s] 19%|█▉        | 75224/400000 [00:08<00:36, 8833.84it/s] 19%|█▉        | 76110/400000 [00:08<00:37, 8708.17it/s] 19%|█▉        | 76992/400000 [00:08<00:36, 8739.28it/s] 19%|█▉        | 77920/400000 [00:08<00:36, 8894.25it/s] 20%|█▉        | 78831/400000 [00:08<00:35, 8952.97it/s] 20%|█▉        | 79728/400000 [00:09<00:36, 8838.70it/s] 20%|██        | 80628/400000 [00:09<00:35, 8884.31it/s] 20%|██        | 81563/400000 [00:09<00:35, 9018.62it/s] 21%|██        | 82487/400000 [00:09<00:34, 9082.65it/s] 21%|██        | 83397/400000 [00:09<00:34, 9084.59it/s] 21%|██        | 84307/400000 [00:09<00:35, 8943.55it/s] 21%|██▏       | 85273/400000 [00:09<00:34, 9146.85it/s] 22%|██▏       | 86233/400000 [00:09<00:33, 9277.39it/s] 22%|██▏       | 87163/400000 [00:09<00:33, 9242.33it/s] 22%|██▏       | 88089/400000 [00:09<00:33, 9187.36it/s] 22%|██▏       | 89009/400000 [00:10<00:34, 9046.34it/s] 22%|██▏       | 89929/400000 [00:10<00:34, 9090.49it/s] 23%|██▎       | 90839/400000 [00:10<00:35, 8721.70it/s] 23%|██▎       | 91715/400000 [00:10<00:36, 8369.38it/s] 23%|██▎       | 92558/400000 [00:10<00:37, 8256.69it/s] 23%|██▎       | 93388/400000 [00:10<00:38, 7978.15it/s] 24%|██▎       | 94191/400000 [00:10<00:38, 7981.14it/s] 24%|██▍       | 95070/400000 [00:10<00:37, 8207.31it/s] 24%|██▍       | 95895/400000 [00:10<00:37, 8112.03it/s] 24%|██▍       | 96710/400000 [00:10<00:37, 8113.52it/s] 24%|██▍       | 97656/400000 [00:11<00:35, 8474.01it/s] 25%|██▍       | 98587/400000 [00:11<00:34, 8708.20it/s] 25%|██▍       | 99542/400000 [00:11<00:33, 8943.16it/s] 25%|██▌       | 100490/400000 [00:11<00:32, 9095.22it/s] 25%|██▌       | 101404/400000 [00:11<00:33, 8906.53it/s] 26%|██▌       | 102371/400000 [00:11<00:32, 9120.65it/s] 26%|██▌       | 103308/400000 [00:11<00:32, 9193.26it/s] 26%|██▌       | 104269/400000 [00:11<00:31, 9313.38it/s] 26%|██▋       | 105203/400000 [00:11<00:33, 8818.36it/s] 27%|██▋       | 106093/400000 [00:12<00:34, 8628.77it/s] 27%|██▋       | 106981/400000 [00:12<00:33, 8702.42it/s] 27%|██▋       | 107856/400000 [00:12<00:34, 8577.80it/s] 27%|██▋       | 108718/400000 [00:12<00:34, 8430.82it/s] 27%|██▋       | 109580/400000 [00:12<00:34, 8484.77it/s] 28%|██▊       | 110431/400000 [00:12<00:34, 8394.67it/s] 28%|██▊       | 111368/400000 [00:12<00:33, 8664.11it/s] 28%|██▊       | 112319/400000 [00:12<00:32, 8901.42it/s] 28%|██▊       | 113214/400000 [00:12<00:32, 8861.08it/s] 29%|██▊       | 114134/400000 [00:12<00:31, 8959.91it/s] 29%|██▉       | 115033/400000 [00:13<00:32, 8882.63it/s] 29%|██▉       | 115978/400000 [00:13<00:31, 9043.76it/s] 29%|██▉       | 116928/400000 [00:13<00:30, 9173.63it/s] 29%|██▉       | 117848/400000 [00:13<00:30, 9166.56it/s] 30%|██▉       | 118785/400000 [00:13<00:30, 9224.62it/s] 30%|██▉       | 119709/400000 [00:13<00:31, 8927.50it/s] 30%|███       | 120605/400000 [00:13<00:31, 8815.22it/s] 30%|███       | 121496/400000 [00:13<00:31, 8843.13it/s] 31%|███       | 122439/400000 [00:13<00:30, 9009.61it/s] 31%|███       | 123410/400000 [00:13<00:30, 9207.45it/s] 31%|███       | 124334/400000 [00:14<00:30, 8935.50it/s] 31%|███▏      | 125286/400000 [00:14<00:30, 9102.24it/s] 32%|███▏      | 126219/400000 [00:14<00:29, 9167.36it/s] 32%|███▏      | 127182/400000 [00:14<00:29, 9298.80it/s] 32%|███▏      | 128114/400000 [00:14<00:29, 9278.13it/s] 32%|███▏      | 129044/400000 [00:14<00:29, 9280.59it/s] 32%|███▏      | 129994/400000 [00:14<00:28, 9345.19it/s] 33%|███▎      | 130930/400000 [00:14<00:29, 9253.92it/s] 33%|███▎      | 131898/400000 [00:14<00:28, 9375.04it/s] 33%|███▎      | 132837/400000 [00:14<00:28, 9334.31it/s] 33%|███▎      | 133772/400000 [00:15<00:28, 9320.64it/s] 34%|███▎      | 134716/400000 [00:15<00:28, 9354.13it/s] 34%|███▍      | 135652/400000 [00:15<00:28, 9221.85it/s] 34%|███▍      | 136575/400000 [00:15<00:28, 9203.14it/s] 34%|███▍      | 137501/400000 [00:15<00:28, 9219.24it/s] 35%|███▍      | 138424/400000 [00:15<00:28, 9126.71it/s] 35%|███▍      | 139378/400000 [00:15<00:28, 9246.22it/s] 35%|███▌      | 140326/400000 [00:15<00:27, 9313.66it/s] 35%|███▌      | 141281/400000 [00:15<00:27, 9381.22it/s] 36%|███▌      | 142220/400000 [00:15<00:27, 9306.26it/s] 36%|███▌      | 143152/400000 [00:16<00:28, 9154.88it/s] 36%|███▌      | 144074/400000 [00:16<00:28, 9119.29it/s] 36%|███▋      | 145042/400000 [00:16<00:27, 9279.02it/s] 36%|███▋      | 145986/400000 [00:16<00:27, 9325.32it/s] 37%|███▋      | 146920/400000 [00:16<00:27, 9104.43it/s] 37%|███▋      | 147833/400000 [00:16<00:28, 8945.74it/s] 37%|███▋      | 148730/400000 [00:16<00:29, 8439.38it/s] 37%|███▋      | 149614/400000 [00:16<00:29, 8554.73it/s] 38%|███▊      | 150475/400000 [00:16<00:29, 8415.42it/s] 38%|███▊      | 151321/400000 [00:17<00:30, 8153.85it/s] 38%|███▊      | 152237/400000 [00:17<00:29, 8431.45it/s] 38%|███▊      | 153186/400000 [00:17<00:28, 8722.25it/s] 39%|███▊      | 154165/400000 [00:17<00:27, 9015.50it/s] 39%|███▉      | 155123/400000 [00:17<00:26, 9176.33it/s] 39%|███▉      | 156047/400000 [00:17<00:26, 9051.97it/s] 39%|███▉      | 156996/400000 [00:17<00:26, 9178.57it/s] 39%|███▉      | 157934/400000 [00:17<00:26, 9236.82it/s] 40%|███▉      | 158886/400000 [00:17<00:25, 9319.45it/s] 40%|███▉      | 159820/400000 [00:17<00:25, 9249.66it/s] 40%|████      | 160747/400000 [00:18<00:26, 9145.90it/s] 40%|████      | 161663/400000 [00:18<00:26, 9077.04it/s] 41%|████      | 162591/400000 [00:18<00:25, 9135.42it/s] 41%|████      | 163547/400000 [00:18<00:25, 9256.84it/s] 41%|████      | 164474/400000 [00:18<00:25, 9155.77it/s] 41%|████▏     | 165391/400000 [00:18<00:26, 8928.87it/s] 42%|████▏     | 166286/400000 [00:18<00:26, 8817.07it/s] 42%|████▏     | 167170/400000 [00:18<00:26, 8810.72it/s] 42%|████▏     | 168069/400000 [00:18<00:26, 8863.52it/s] 42%|████▏     | 168957/400000 [00:18<00:26, 8683.78it/s] 42%|████▏     | 169827/400000 [00:19<00:26, 8563.46it/s] 43%|████▎     | 170689/400000 [00:19<00:26, 8578.33it/s] 43%|████▎     | 171611/400000 [00:19<00:26, 8759.98it/s] 43%|████▎     | 172519/400000 [00:19<00:25, 8853.27it/s] 43%|████▎     | 173426/400000 [00:19<00:25, 8915.17it/s] 44%|████▎     | 174319/400000 [00:19<00:25, 8864.49it/s] 44%|████▍     | 175207/400000 [00:19<00:25, 8653.25it/s] 44%|████▍     | 176148/400000 [00:19<00:25, 8865.70it/s] 44%|████▍     | 177038/400000 [00:19<00:25, 8853.64it/s] 44%|████▍     | 177962/400000 [00:20<00:24, 8965.53it/s] 45%|████▍     | 178861/400000 [00:20<00:25, 8808.79it/s] 45%|████▍     | 179744/400000 [00:20<00:25, 8530.62it/s] 45%|████▌     | 180601/400000 [00:20<00:25, 8539.63it/s] 45%|████▌     | 181528/400000 [00:20<00:24, 8744.84it/s] 46%|████▌     | 182439/400000 [00:20<00:24, 8850.34it/s] 46%|████▌     | 183327/400000 [00:20<00:25, 8639.87it/s] 46%|████▌     | 184194/400000 [00:20<00:25, 8597.71it/s] 46%|████▋     | 185133/400000 [00:20<00:24, 8820.53it/s] 47%|████▋     | 186027/400000 [00:20<00:24, 8854.39it/s] 47%|████▋     | 186957/400000 [00:21<00:23, 8981.44it/s] 47%|████▋     | 187857/400000 [00:21<00:23, 8847.27it/s] 47%|████▋     | 188744/400000 [00:21<00:24, 8718.84it/s] 47%|████▋     | 189658/400000 [00:21<00:23, 8839.88it/s] 48%|████▊     | 190556/400000 [00:21<00:23, 8881.04it/s] 48%|████▊     | 191482/400000 [00:21<00:23, 8989.69it/s] 48%|████▊     | 192383/400000 [00:21<00:23, 8926.81it/s] 48%|████▊     | 193321/400000 [00:21<00:22, 9057.15it/s] 49%|████▊     | 194228/400000 [00:21<00:22, 9009.14it/s] 49%|████▉     | 195130/400000 [00:21<00:24, 8497.15it/s] 49%|████▉     | 195993/400000 [00:22<00:23, 8536.48it/s] 49%|████▉     | 196872/400000 [00:22<00:23, 8608.95it/s] 49%|████▉     | 197802/400000 [00:22<00:22, 8803.11it/s] 50%|████▉     | 198732/400000 [00:22<00:22, 8945.87it/s] 50%|████▉     | 199654/400000 [00:22<00:22, 9025.61it/s] 50%|█████     | 200570/400000 [00:22<00:22, 9064.83it/s] 50%|█████     | 201479/400000 [00:22<00:22, 8837.18it/s] 51%|█████     | 202366/400000 [00:22<00:22, 8778.98it/s] 51%|█████     | 203256/400000 [00:22<00:22, 8812.89it/s] 51%|█████     | 204139/400000 [00:22<00:22, 8794.73it/s] 51%|█████▏    | 205020/400000 [00:23<00:22, 8645.59it/s] 51%|█████▏    | 205886/400000 [00:23<00:22, 8534.27it/s] 52%|█████▏    | 206743/400000 [00:23<00:22, 8544.12it/s] 52%|█████▏    | 207599/400000 [00:23<00:23, 8291.78it/s] 52%|█████▏    | 208431/400000 [00:23<00:23, 8110.46it/s] 52%|█████▏    | 209245/400000 [00:23<00:23, 8079.41it/s] 53%|█████▎    | 210067/400000 [00:23<00:23, 8118.18it/s] 53%|█████▎    | 210918/400000 [00:23<00:22, 8230.51it/s] 53%|█████▎    | 211823/400000 [00:23<00:22, 8459.17it/s] 53%|█████▎    | 212714/400000 [00:24<00:21, 8589.21it/s] 53%|█████▎    | 213612/400000 [00:24<00:21, 8698.57it/s] 54%|█████▎    | 214484/400000 [00:24<00:21, 8607.83it/s] 54%|█████▍    | 215444/400000 [00:24<00:20, 8881.16it/s] 54%|█████▍    | 216336/400000 [00:24<00:20, 8781.52it/s] 54%|█████▍    | 217275/400000 [00:24<00:20, 8954.72it/s] 55%|█████▍    | 218237/400000 [00:24<00:19, 9143.53it/s] 55%|█████▍    | 219155/400000 [00:24<00:20, 9038.29it/s] 55%|█████▌    | 220089/400000 [00:24<00:19, 9126.28it/s] 55%|█████▌    | 221056/400000 [00:24<00:19, 9280.62it/s] 56%|█████▌    | 222012/400000 [00:25<00:19, 9362.32it/s] 56%|█████▌    | 222971/400000 [00:25<00:18, 9428.96it/s] 56%|█████▌    | 223916/400000 [00:25<00:19, 9057.24it/s] 56%|█████▌    | 224826/400000 [00:25<00:19, 8850.11it/s] 56%|█████▋    | 225768/400000 [00:25<00:19, 9012.93it/s] 57%|█████▋    | 226718/400000 [00:25<00:18, 9151.64it/s] 57%|█████▋    | 227637/400000 [00:25<00:18, 9160.21it/s] 57%|█████▋    | 228556/400000 [00:25<00:19, 8773.92it/s] 57%|█████▋    | 229486/400000 [00:25<00:19, 8924.37it/s] 58%|█████▊    | 230454/400000 [00:25<00:18, 9136.55it/s] 58%|█████▊    | 231424/400000 [00:26<00:18, 9296.60it/s] 58%|█████▊    | 232383/400000 [00:26<00:17, 9381.37it/s] 58%|█████▊    | 233324/400000 [00:26<00:18, 9134.20it/s] 59%|█████▊    | 234241/400000 [00:26<00:18, 8838.43it/s] 59%|█████▉    | 235130/400000 [00:26<00:19, 8583.26it/s] 59%|█████▉    | 236100/400000 [00:26<00:18, 8888.94it/s] 59%|█████▉    | 237015/400000 [00:26<00:18, 8962.32it/s] 59%|█████▉    | 237942/400000 [00:26<00:17, 9051.75it/s] 60%|█████▉    | 238851/400000 [00:26<00:17, 9008.08it/s] 60%|█████▉    | 239755/400000 [00:27<00:17, 8985.09it/s] 60%|██████    | 240669/400000 [00:27<00:17, 9029.00it/s] 60%|██████    | 241610/400000 [00:27<00:17, 9139.91it/s] 61%|██████    | 242526/400000 [00:27<00:17, 9048.67it/s] 61%|██████    | 243489/400000 [00:27<00:16, 9215.05it/s] 61%|██████    | 244412/400000 [00:27<00:16, 9209.78it/s] 61%|██████▏   | 245371/400000 [00:27<00:16, 9318.46it/s] 62%|██████▏   | 246327/400000 [00:27<00:16, 9388.36it/s] 62%|██████▏   | 247267/400000 [00:27<00:16, 9092.23it/s] 62%|██████▏   | 248193/400000 [00:27<00:16, 9137.43it/s] 62%|██████▏   | 249134/400000 [00:28<00:16, 9215.81it/s] 63%|██████▎   | 250082/400000 [00:28<00:16, 9291.33it/s] 63%|██████▎   | 251013/400000 [00:28<00:16, 9139.23it/s] 63%|██████▎   | 251929/400000 [00:28<00:16, 8775.78it/s] 63%|██████▎   | 252863/400000 [00:28<00:16, 8937.66it/s] 63%|██████▎   | 253794/400000 [00:28<00:16, 9044.00it/s] 64%|██████▎   | 254702/400000 [00:28<00:16, 9007.91it/s] 64%|██████▍   | 255628/400000 [00:28<00:15, 9076.76it/s] 64%|██████▍   | 256538/400000 [00:28<00:15, 9035.77it/s] 64%|██████▍   | 257461/400000 [00:28<00:15, 9092.14it/s] 65%|██████▍   | 258421/400000 [00:29<00:15, 9236.13it/s] 65%|██████▍   | 259358/400000 [00:29<00:15, 9273.42it/s] 65%|██████▌   | 260287/400000 [00:29<00:15, 9240.86it/s] 65%|██████▌   | 261212/400000 [00:29<00:15, 9149.91it/s] 66%|██████▌   | 262162/400000 [00:29<00:14, 9251.19it/s] 66%|██████▌   | 263088/400000 [00:29<00:15, 8999.90it/s] 66%|██████▌   | 263990/400000 [00:29<00:15, 8969.92it/s] 66%|██████▌   | 264889/400000 [00:29<00:15, 8905.88it/s] 66%|██████▋   | 265781/400000 [00:29<00:15, 8752.34it/s] 67%|██████▋   | 266658/400000 [00:29<00:15, 8472.66it/s] 67%|██████▋   | 267618/400000 [00:30<00:15, 8781.26it/s] 67%|██████▋   | 268552/400000 [00:30<00:14, 8939.97it/s] 67%|██████▋   | 269451/400000 [00:30<00:15, 8659.71it/s] 68%|██████▊   | 270404/400000 [00:30<00:14, 8901.70it/s] 68%|██████▊   | 271353/400000 [00:30<00:14, 9070.20it/s] 68%|██████▊   | 272290/400000 [00:30<00:13, 9156.59it/s] 68%|██████▊   | 273229/400000 [00:30<00:13, 9223.03it/s] 69%|██████▊   | 274154/400000 [00:30<00:14, 8897.68it/s] 69%|██████▉   | 275048/400000 [00:30<00:14, 8905.01it/s] 69%|██████▉   | 275971/400000 [00:31<00:13, 8999.83it/s] 69%|██████▉   | 276874/400000 [00:31<00:14, 8788.46it/s] 69%|██████▉   | 277756/400000 [00:31<00:14, 8473.72it/s] 70%|██████▉   | 278608/400000 [00:31<00:14, 8316.05it/s] 70%|██████▉   | 279559/400000 [00:31<00:13, 8640.70it/s] 70%|███████   | 280455/400000 [00:31<00:13, 8732.57it/s] 70%|███████   | 281333/400000 [00:31<00:13, 8638.76it/s] 71%|███████   | 282201/400000 [00:31<00:13, 8622.02it/s] 71%|███████   | 283066/400000 [00:31<00:13, 8536.21it/s] 71%|███████   | 283922/400000 [00:31<00:13, 8315.47it/s] 71%|███████   | 284846/400000 [00:32<00:13, 8571.50it/s] 71%|███████▏  | 285800/400000 [00:32<00:12, 8840.73it/s] 72%|███████▏  | 286756/400000 [00:32<00:12, 9044.17it/s] 72%|███████▏  | 287665/400000 [00:32<00:12, 8942.06it/s] 72%|███████▏  | 288631/400000 [00:32<00:12, 9145.63it/s] 72%|███████▏  | 289571/400000 [00:32<00:11, 9220.46it/s] 73%|███████▎  | 290496/400000 [00:32<00:11, 9226.12it/s] 73%|███████▎  | 291421/400000 [00:32<00:11, 9214.89it/s] 73%|███████▎  | 292344/400000 [00:32<00:11, 9002.68it/s] 73%|███████▎  | 293278/400000 [00:32<00:11, 9100.06it/s] 74%|███████▎  | 294241/400000 [00:33<00:11, 9251.77it/s] 74%|███████▍  | 295182/400000 [00:33<00:11, 9296.07it/s] 74%|███████▍  | 296133/400000 [00:33<00:11, 9355.51it/s] 74%|███████▍  | 297070/400000 [00:33<00:11, 9206.80it/s] 74%|███████▍  | 297997/400000 [00:33<00:11, 9224.77it/s] 75%|███████▍  | 298921/400000 [00:33<00:11, 9163.69it/s] 75%|███████▍  | 299839/400000 [00:33<00:10, 9146.63it/s] 75%|███████▌  | 300769/400000 [00:33<00:10, 9190.90it/s] 75%|███████▌  | 301689/400000 [00:33<00:10, 9079.25it/s] 76%|███████▌  | 302667/400000 [00:33<00:10, 9277.04it/s] 76%|███████▌  | 303597/400000 [00:34<00:10, 9002.17it/s] 76%|███████▌  | 304501/400000 [00:34<00:10, 8919.58it/s] 76%|███████▋  | 305456/400000 [00:34<00:10, 9099.40it/s] 77%|███████▋  | 306369/400000 [00:34<00:10, 9079.66it/s] 77%|███████▋  | 307336/400000 [00:34<00:10, 9247.25it/s] 77%|███████▋  | 308293/400000 [00:34<00:09, 9341.40it/s] 77%|███████▋  | 309262/400000 [00:34<00:09, 9441.54it/s] 78%|███████▊  | 310228/400000 [00:34<00:09, 9505.75it/s] 78%|███████▊  | 311180/400000 [00:34<00:09, 9374.22it/s] 78%|███████▊  | 312119/400000 [00:34<00:09, 9374.32it/s] 78%|███████▊  | 313058/400000 [00:35<00:09, 9157.63it/s] 78%|███████▊  | 313976/400000 [00:35<00:09, 8888.36it/s] 79%|███████▊  | 314868/400000 [00:35<00:09, 8811.97it/s] 79%|███████▉  | 315766/400000 [00:35<00:09, 8860.83it/s] 79%|███████▉  | 316719/400000 [00:35<00:09, 9050.87it/s] 79%|███████▉  | 317670/400000 [00:35<00:08, 9183.85it/s] 80%|███████▉  | 318591/400000 [00:35<00:09, 9030.24it/s] 80%|███████▉  | 319508/400000 [00:35<00:08, 9069.96it/s] 80%|████████  | 320418/400000 [00:35<00:08, 9076.87it/s] 80%|████████  | 321327/400000 [00:36<00:08, 8778.68it/s] 81%|████████  | 322242/400000 [00:36<00:08, 8886.45it/s] 81%|████████  | 323133/400000 [00:36<00:08, 8816.54it/s] 81%|████████  | 324017/400000 [00:36<00:08, 8678.47it/s] 81%|████████  | 324895/400000 [00:36<00:08, 8707.91it/s] 81%|████████▏ | 325808/400000 [00:36<00:08, 8828.72it/s] 82%|████████▏ | 326709/400000 [00:36<00:08, 8879.67it/s] 82%|████████▏ | 327598/400000 [00:36<00:08, 8844.04it/s] 82%|████████▏ | 328484/400000 [00:36<00:08, 8633.79it/s] 82%|████████▏ | 329388/400000 [00:36<00:08, 8750.34it/s] 83%|████████▎ | 330272/400000 [00:37<00:07, 8774.70it/s] 83%|████████▎ | 331202/400000 [00:37<00:07, 8925.85it/s] 83%|████████▎ | 332119/400000 [00:37<00:07, 8997.63it/s] 83%|████████▎ | 333020/400000 [00:37<00:07, 8946.65it/s] 83%|████████▎ | 333945/400000 [00:37<00:07, 9033.23it/s] 84%|████████▎ | 334869/400000 [00:37<00:07, 9092.02it/s] 84%|████████▍ | 335787/400000 [00:37<00:07, 9116.21it/s] 84%|████████▍ | 336704/400000 [00:37<00:06, 9129.92it/s] 84%|████████▍ | 337618/400000 [00:37<00:06, 9115.33it/s] 85%|████████▍ | 338552/400000 [00:37<00:06, 9180.67it/s] 85%|████████▍ | 339480/400000 [00:38<00:06, 9209.44it/s] 85%|████████▌ | 340447/400000 [00:38<00:06, 9340.60it/s] 85%|████████▌ | 341391/400000 [00:38<00:06, 9370.10it/s] 86%|████████▌ | 342329/400000 [00:38<00:06, 9289.51it/s] 86%|████████▌ | 343259/400000 [00:38<00:06, 9240.38it/s] 86%|████████▌ | 344184/400000 [00:38<00:06, 9159.66it/s] 86%|████████▋ | 345122/400000 [00:38<00:05, 9224.21it/s] 87%|████████▋ | 346070/400000 [00:38<00:05, 9299.26it/s] 87%|████████▋ | 347012/400000 [00:38<00:05, 9331.86it/s] 87%|████████▋ | 347946/400000 [00:38<00:05, 9247.61it/s] 87%|████████▋ | 348872/400000 [00:39<00:05, 9116.49it/s] 87%|████████▋ | 349806/400000 [00:39<00:05, 9180.79it/s] 88%|████████▊ | 350725/400000 [00:39<00:05, 9140.21it/s] 88%|████████▊ | 351640/400000 [00:39<00:05, 9064.02it/s] 88%|████████▊ | 352547/400000 [00:39<00:05, 8606.87it/s] 88%|████████▊ | 353473/400000 [00:39<00:05, 8791.96it/s] 89%|████████▊ | 354451/400000 [00:39<00:05, 9064.19it/s] 89%|████████▉ | 355424/400000 [00:39<00:04, 9251.55it/s] 89%|████████▉ | 356354/400000 [00:39<00:04, 9154.16it/s] 89%|████████▉ | 357308/400000 [00:40<00:04, 9264.28it/s] 90%|████████▉ | 358251/400000 [00:40<00:04, 9313.26it/s] 90%|████████▉ | 359185/400000 [00:40<00:04, 8896.51it/s] 90%|█████████ | 360126/400000 [00:40<00:04, 9043.45it/s] 90%|█████████ | 361035/400000 [00:40<00:04, 8991.21it/s] 90%|█████████ | 361969/400000 [00:40<00:04, 9092.40it/s] 91%|█████████ | 362923/400000 [00:40<00:04, 9222.20it/s] 91%|█████████ | 363888/400000 [00:40<00:03, 9344.74it/s] 91%|█████████ | 364825/400000 [00:40<00:03, 9351.16it/s] 91%|█████████▏| 365762/400000 [00:40<00:03, 9164.06it/s] 92%|█████████▏| 366745/400000 [00:41<00:03, 9350.45it/s] 92%|█████████▏| 367685/400000 [00:41<00:03, 9364.69it/s] 92%|█████████▏| 368648/400000 [00:41<00:03, 9442.14it/s] 92%|█████████▏| 369594/400000 [00:41<00:03, 9428.24it/s] 93%|█████████▎| 370538/400000 [00:41<00:03, 9142.42it/s] 93%|█████████▎| 371480/400000 [00:41<00:03, 9222.12it/s] 93%|█████████▎| 372438/400000 [00:41<00:02, 9325.63it/s] 93%|█████████▎| 373392/400000 [00:41<00:02, 9388.48it/s] 94%|█████████▎| 374333/400000 [00:41<00:02, 9190.93it/s] 94%|█████████▍| 375254/400000 [00:41<00:02, 8705.09it/s] 94%|█████████▍| 376173/400000 [00:42<00:02, 8843.85it/s] 94%|█████████▍| 377097/400000 [00:42<00:02, 8958.98it/s] 95%|█████████▍| 378018/400000 [00:42<00:02, 9031.53it/s] 95%|█████████▍| 378968/400000 [00:42<00:02, 9149.01it/s] 95%|█████████▍| 379886/400000 [00:42<00:02, 8380.55it/s] 95%|█████████▌| 380846/400000 [00:42<00:02, 8712.10it/s] 95%|█████████▌| 381731/400000 [00:42<00:02, 8691.44it/s] 96%|█████████▌| 382610/400000 [00:42<00:02, 8540.20it/s] 96%|█████████▌| 383559/400000 [00:42<00:01, 8803.31it/s] 96%|█████████▌| 384459/400000 [00:43<00:01, 8859.22it/s] 96%|█████████▋| 385433/400000 [00:43<00:01, 9105.15it/s] 97%|█████████▋| 386364/400000 [00:43<00:01, 9164.89it/s] 97%|█████████▋| 387337/400000 [00:43<00:01, 9327.19it/s] 97%|█████████▋| 388313/400000 [00:43<00:01, 9452.68it/s] 97%|█████████▋| 389261/400000 [00:43<00:01, 8905.66it/s] 98%|█████████▊| 390169/400000 [00:43<00:01, 8956.09it/s] 98%|█████████▊| 391071/400000 [00:43<00:00, 8969.37it/s] 98%|█████████▊| 391973/400000 [00:43<00:00, 8960.13it/s] 98%|█████████▊| 392919/400000 [00:43<00:00, 9102.72it/s] 98%|█████████▊| 393832/400000 [00:44<00:00, 9084.98it/s] 99%|█████████▊| 394775/400000 [00:44<00:00, 9184.49it/s] 99%|█████████▉| 395695/400000 [00:44<00:00, 9103.55it/s] 99%|█████████▉| 396636/400000 [00:44<00:00, 9192.84it/s] 99%|█████████▉| 397557/400000 [00:44<00:00, 9097.77it/s]100%|█████████▉| 398475/400000 [00:44<00:00, 9120.98it/s]100%|█████████▉| 399437/400000 [00:44<00:00, 9262.82it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8949.23it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f13a59c5a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011147844177458932 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.01115138016416875 	 Accuracy: 64

  model saves at 64% accuracy 

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

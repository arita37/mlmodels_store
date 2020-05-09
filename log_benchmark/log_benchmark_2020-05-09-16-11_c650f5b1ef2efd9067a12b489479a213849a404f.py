
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'c650f5b1ef2efd9067a12b489479a213849a404f', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c650f5b1ef2efd9067a12b489479a213849a404f

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4a6c4f7470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 16:11:28.991076
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 16:11:28.995116
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 16:11:28.998343
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 16:11:29.001774
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4a648474a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 354921.5312
Epoch 2/10

1/1 [==============================] - 0s 87ms/step - loss: 270829.6875
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 178175.5625
Epoch 4/10

1/1 [==============================] - 0s 79ms/step - loss: 98677.8203
Epoch 5/10

1/1 [==============================] - 0s 82ms/step - loss: 51652.1680
Epoch 6/10

1/1 [==============================] - 0s 78ms/step - loss: 28458.0898
Epoch 7/10

1/1 [==============================] - 0s 79ms/step - loss: 16970.6328
Epoch 8/10

1/1 [==============================] - 0s 84ms/step - loss: 11046.8516
Epoch 9/10

1/1 [==============================] - 0s 87ms/step - loss: 7724.0410
Epoch 10/10

1/1 [==============================] - 0s 81ms/step - loss: 5723.3721

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.90459967e-01 -1.61688352e+00 -1.90285933e+00 -7.63504624e-01
  -6.73978925e-02  6.66386425e-01 -8.54398608e-01 -2.25508511e-02
   1.24157119e+00  3.86140525e-01 -5.77627778e-01 -1.36081278e-02
   1.30406880e+00 -3.52258325e-01  9.96531546e-01  8.83780718e-01
   5.05420208e-01 -7.11864769e-01  2.76140004e-01  1.71273732e+00
  -1.14458418e+00 -2.78875023e-01  1.00610113e+00  7.89840937e-01
   1.98374081e+00 -6.00133300e-01 -1.07821238e+00 -5.44578254e-01
   4.00110602e-01 -8.00310016e-01 -1.34092763e-01 -1.23393106e+00
  -4.99957353e-01  3.55093032e-02  1.53288960e+00 -5.81185222e-01
   1.14148474e+00 -4.09629136e-01  5.90638518e-01 -7.24502563e-01
   2.68101692e-03  1.50558639e+00 -1.44369781e-01 -1.14842498e+00
   1.08734202e+00  1.11739647e+00  6.71762466e-01 -1.20996296e+00
   1.29165506e+00 -9.95990634e-01  8.01386356e-01  6.22987986e-01
  -5.43584704e-01 -5.95172524e-01  1.45654297e+00 -2.23269010e+00
   1.80282548e-01  8.92642379e-01  9.26567435e-01  1.10415328e+00
  -4.70822215e-01  8.42151833e+00  1.04235773e+01  9.55286884e+00
   9.83647823e+00  9.10603714e+00  9.72892189e+00  9.32397175e+00
   1.04089918e+01  7.18995285e+00  8.08778477e+00  8.35451698e+00
   8.03168869e+00  9.15016747e+00  8.35137749e+00  7.79060173e+00
   9.00868797e+00  7.43777466e+00  9.45937347e+00  8.56578541e+00
   8.50129986e+00  8.25571918e+00  9.97472191e+00  8.01828671e+00
   8.45690918e+00  7.22980356e+00  9.03347492e+00  8.29299355e+00
   9.29527092e+00  1.15202866e+01  8.89466000e+00  8.30308151e+00
   8.45144558e+00  8.63523579e+00  9.64706421e+00  1.14569435e+01
   8.96808624e+00  1.12415400e+01  8.08615685e+00  8.28934669e+00
   8.24344826e+00  9.28389645e+00  8.52878761e+00  8.82577515e+00
   1.02792377e+01  1.01757097e+01  9.16540909e+00  6.53510571e+00
   9.62351513e+00  8.50048637e+00  8.92092133e+00  8.34154797e+00
   7.25158024e+00  9.35354328e+00  8.45465565e+00  9.90283489e+00
   8.55034828e+00  1.00803814e+01  9.62423897e+00  9.40990353e+00
   6.42041683e-01  1.30454314e+00 -2.16261673e+00  1.03795779e+00
  -3.87071133e-01 -2.24124849e-01  3.81891191e-01  7.52546906e-01
   1.86930209e-01 -5.29772997e-01 -1.56259167e+00  3.74702334e-01
   5.38874090e-01  2.24354044e-01 -1.36638880e+00 -1.60998869e+00
   2.27213526e+00 -4.35513854e-01 -5.45344830e-01 -3.84892821e-01
   3.92602205e-01 -2.00841212e+00 -5.07968426e-01  5.76577842e-01
   6.29478693e-01 -1.35175633e+00  7.96963871e-01 -2.34468293e+00
   9.13278222e-01 -2.49567926e-01  2.18122625e+00  9.28267777e-01
   6.75500631e-01 -4.13268209e-02 -6.67820215e-01  1.21900511e+00
  -1.13520813e+00  7.98707008e-01 -1.53425789e+00  8.70922089e-01
   5.15157163e-01  9.16052461e-02 -4.69026387e-01 -8.34437609e-01
  -8.57169330e-02  3.96824539e-01 -1.09892499e+00 -7.37046659e-01
  -7.31109440e-01  4.26540017e-01 -7.36587524e-01 -1.93394065e+00
   2.09515977e+00 -2.61124325e+00  1.50340128e+00  3.51513505e-01
   1.75299573e+00 -3.35020757e+00 -5.06740689e-01 -9.56415832e-01
   1.72151387e+00  1.87788427e+00  3.23677242e-01  1.96435761e+00
   6.97122276e-01  1.16464353e+00  1.69253945e+00  1.62369776e+00
   2.97348595e+00  1.76764846e+00  2.66040707e+00  2.24418521e-01
   2.21582174e+00  2.66235495e+00  1.53376698e+00  4.28859234e-01
   2.58137321e+00  5.23141146e-01  1.17877960e-01  1.19668722e+00
   1.84507108e+00  1.18191862e+00  1.82353616e-01  3.22988224e+00
   1.71872056e+00  3.79786670e-01  3.25947106e-01  1.34624624e+00
   1.29107237e+00  2.83543730e+00  2.52323151e-01  1.58296466e+00
   2.85725117e+00  2.11015403e-01  2.04684973e+00  6.35088027e-01
   1.97455740e+00  2.43293822e-01  9.76765275e-01  3.12635422e-01
   2.91938496e+00  1.35295129e+00  1.11905515e+00  3.33085775e-01
   1.57487607e+00  2.26804614e-01  9.55291390e-01  9.21829939e-02
   1.25786674e+00  9.59251583e-01  1.82806230e+00  5.62868237e-01
   5.45205474e-01  1.07473910e-01  7.86381841e-01  6.68698847e-01
   2.22909808e-01  5.32970428e-01  6.34755850e-01  4.53640938e-01
   3.54902148e-01  1.03446255e+01  1.01714811e+01  9.19348526e+00
   9.15300179e+00  1.10503130e+01  9.81868172e+00  7.15224457e+00
   9.54573441e+00  8.96454716e+00  8.67021942e+00  8.80915833e+00
   9.49768543e+00  9.07835197e+00  9.92145920e+00  9.59019661e+00
   1.11575384e+01  8.83354759e+00  9.58769226e+00  9.05906105e+00
   1.04576063e+01  9.57649136e+00  9.59160900e+00  1.10253887e+01
   9.13760567e+00  8.83251667e+00  7.23417950e+00  7.59168768e+00
   9.49331856e+00  9.16743088e+00  1.11752377e+01  1.00321751e+01
   1.10654869e+01  8.02019215e+00  6.90784073e+00  8.88868046e+00
   1.03241997e+01  9.59115314e+00  1.02187672e+01  9.78648472e+00
   7.41745710e+00  9.86468601e+00  8.25804520e+00  9.83220673e+00
   9.11032391e+00  7.80649328e+00  1.04486561e+01  9.26433754e+00
   7.90447903e+00  9.23943996e+00  1.02948275e+01  1.03581839e+01
   7.83647966e+00  9.29797077e+00  7.67960691e+00  9.37504387e+00
   9.62413216e+00  9.66930676e+00  8.97307396e+00  7.99478340e+00
   6.89932883e-01  6.94628119e-01  1.90287066e+00  2.76184499e-01
   1.56217158e-01  7.93092310e-01  1.65850556e+00  5.53970397e-01
   5.44718504e-01  1.04587817e+00  7.52865791e-01  4.43935871e-01
   1.97411299e-01  1.53008330e+00  1.44714439e+00  9.51120317e-01
   2.15124989e+00  4.30226088e-01  4.61775064e-01  8.93241882e-01
   2.98042238e-01  8.90347838e-01  6.23007238e-01  8.75937343e-01
   1.80322027e+00  1.26488590e+00  5.83012879e-01  1.25905907e+00
   1.70396256e+00  9.27337408e-01  2.42621422e+00  7.26909876e-01
   1.75906122e+00  7.32097447e-01  2.35297155e+00  3.13666165e-01
   1.35815907e+00  3.29278469e+00  8.33796084e-01  1.28882432e+00
   5.51532567e-01  2.00721931e+00  1.44571912e+00  1.79010987e-01
   3.88644040e-01  1.57685328e+00  3.30270052e-01  1.51564646e+00
   3.02828372e-01  5.96323907e-01  4.33380067e-01  3.22272587e+00
   8.85693371e-01  7.52934456e-01  3.41123152e+00  2.05456555e-01
   3.42488575e+00  6.78536534e-01  6.44331992e-01  1.10416126e+00
  -5.26842117e+00  6.87358713e+00 -1.66248608e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 16:11:37.437031
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.8921
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 16:11:37.440092
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8661.08
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 16:11:37.442724
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.0957
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 16:11:37.445765
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -774.661
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139956932556392
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139955856925584
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139955856926088
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139955856926592
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139955856927096
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139955856927600

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4a64847438> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.514372
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.493561
grad_step = 000002, loss = 0.478684
grad_step = 000003, loss = 0.463244
grad_step = 000004, loss = 0.443563
grad_step = 000005, loss = 0.421301
grad_step = 000006, loss = 0.398064
grad_step = 000007, loss = 0.375985
grad_step = 000008, loss = 0.364261
grad_step = 000009, loss = 0.364171
grad_step = 000010, loss = 0.355962
grad_step = 000011, loss = 0.339089
grad_step = 000012, loss = 0.323897
grad_step = 000013, loss = 0.314354
grad_step = 000014, loss = 0.306705
grad_step = 000015, loss = 0.300172
grad_step = 000016, loss = 0.292904
grad_step = 000017, loss = 0.284039
grad_step = 000018, loss = 0.274116
grad_step = 000019, loss = 0.264146
grad_step = 000020, loss = 0.254941
grad_step = 000021, loss = 0.246981
grad_step = 000022, loss = 0.240084
grad_step = 000023, loss = 0.233074
grad_step = 000024, loss = 0.225285
grad_step = 000025, loss = 0.217146
grad_step = 000026, loss = 0.209279
grad_step = 000027, loss = 0.202067
grad_step = 000028, loss = 0.195242
grad_step = 000029, loss = 0.188330
grad_step = 000030, loss = 0.181285
grad_step = 000031, loss = 0.174326
grad_step = 000032, loss = 0.167687
grad_step = 000033, loss = 0.161398
grad_step = 000034, loss = 0.155155
grad_step = 000035, loss = 0.148873
grad_step = 000036, loss = 0.142714
grad_step = 000037, loss = 0.136781
grad_step = 000038, loss = 0.131080
grad_step = 000039, loss = 0.125492
grad_step = 000040, loss = 0.119978
grad_step = 000041, loss = 0.114583
grad_step = 000042, loss = 0.109360
grad_step = 000043, loss = 0.104350
grad_step = 000044, loss = 0.099432
grad_step = 000045, loss = 0.094608
grad_step = 000046, loss = 0.089893
grad_step = 000047, loss = 0.085413
grad_step = 000048, loss = 0.081135
grad_step = 000049, loss = 0.076953
grad_step = 000050, loss = 0.072800
grad_step = 000051, loss = 0.068763
grad_step = 000052, loss = 0.064923
grad_step = 000053, loss = 0.061274
grad_step = 000054, loss = 0.057728
grad_step = 000055, loss = 0.054278
grad_step = 000056, loss = 0.050939
grad_step = 000057, loss = 0.047748
grad_step = 000058, loss = 0.044691
grad_step = 000059, loss = 0.041761
grad_step = 000060, loss = 0.038948
grad_step = 000061, loss = 0.036264
grad_step = 000062, loss = 0.033707
grad_step = 000063, loss = 0.031271
grad_step = 000064, loss = 0.028955
grad_step = 000065, loss = 0.026750
grad_step = 000066, loss = 0.024675
grad_step = 000067, loss = 0.022740
grad_step = 000068, loss = 0.020924
grad_step = 000069, loss = 0.019180
grad_step = 000070, loss = 0.017526
grad_step = 000071, loss = 0.015990
grad_step = 000072, loss = 0.014597
grad_step = 000073, loss = 0.013317
grad_step = 000074, loss = 0.012149
grad_step = 000075, loss = 0.011066
grad_step = 000076, loss = 0.010083
grad_step = 000077, loss = 0.009171
grad_step = 000078, loss = 0.008345
grad_step = 000079, loss = 0.007618
grad_step = 000080, loss = 0.006977
grad_step = 000081, loss = 0.006404
grad_step = 000082, loss = 0.005893
grad_step = 000083, loss = 0.005441
grad_step = 000084, loss = 0.005033
grad_step = 000085, loss = 0.004654
grad_step = 000086, loss = 0.004312
grad_step = 000087, loss = 0.004016
grad_step = 000088, loss = 0.003763
grad_step = 000089, loss = 0.003541
grad_step = 000090, loss = 0.003350
grad_step = 000091, loss = 0.003192
grad_step = 000092, loss = 0.003061
grad_step = 000093, loss = 0.002952
grad_step = 000094, loss = 0.002869
grad_step = 000095, loss = 0.002814
grad_step = 000096, loss = 0.002775
grad_step = 000097, loss = 0.002731
grad_step = 000098, loss = 0.002665
grad_step = 000099, loss = 0.002549
grad_step = 000100, loss = 0.002441
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002390
grad_step = 000102, loss = 0.002395
grad_step = 000103, loss = 0.002411
grad_step = 000104, loss = 0.002391
grad_step = 000105, loss = 0.002333
grad_step = 000106, loss = 0.002276
grad_step = 000107, loss = 0.002257
grad_step = 000108, loss = 0.002269
grad_step = 000109, loss = 0.002277
grad_step = 000110, loss = 0.002260
grad_step = 000111, loss = 0.002223
grad_step = 000112, loss = 0.002194
grad_step = 000113, loss = 0.002190
grad_step = 000114, loss = 0.002197
grad_step = 000115, loss = 0.002198
grad_step = 000116, loss = 0.002181
grad_step = 000117, loss = 0.002158
grad_step = 000118, loss = 0.002141
grad_step = 000119, loss = 0.002137
grad_step = 000120, loss = 0.002138
grad_step = 000121, loss = 0.002136
grad_step = 000122, loss = 0.002126
grad_step = 000123, loss = 0.002110
grad_step = 000124, loss = 0.002096
grad_step = 000125, loss = 0.002087
grad_step = 000126, loss = 0.002083
grad_step = 000127, loss = 0.002081
grad_step = 000128, loss = 0.002076
grad_step = 000129, loss = 0.002068
grad_step = 000130, loss = 0.002058
grad_step = 000131, loss = 0.002047
grad_step = 000132, loss = 0.002037
grad_step = 000133, loss = 0.002030
grad_step = 000134, loss = 0.002024
grad_step = 000135, loss = 0.002019
grad_step = 000136, loss = 0.002015
grad_step = 000137, loss = 0.002011
grad_step = 000138, loss = 0.002006
grad_step = 000139, loss = 0.002001
grad_step = 000140, loss = 0.001994
grad_step = 000141, loss = 0.001989
grad_step = 000142, loss = 0.001983
grad_step = 000143, loss = 0.001977
grad_step = 000144, loss = 0.001971
grad_step = 000145, loss = 0.001969
grad_step = 000146, loss = 0.001964
grad_step = 000147, loss = 0.001962
grad_step = 000148, loss = 0.001960
grad_step = 000149, loss = 0.001964
grad_step = 000150, loss = 0.001969
grad_step = 000151, loss = 0.001981
grad_step = 000152, loss = 0.002002
grad_step = 000153, loss = 0.002029
grad_step = 000154, loss = 0.002059
grad_step = 000155, loss = 0.002073
grad_step = 000156, loss = 0.002069
grad_step = 000157, loss = 0.002010
grad_step = 000158, loss = 0.001943
grad_step = 000159, loss = 0.001896
grad_step = 000160, loss = 0.001885
grad_step = 000161, loss = 0.001907
grad_step = 000162, loss = 0.001938
grad_step = 000163, loss = 0.001956
grad_step = 000164, loss = 0.001938
grad_step = 000165, loss = 0.001901
grad_step = 000166, loss = 0.001869
grad_step = 000167, loss = 0.001855
grad_step = 000168, loss = 0.001862
grad_step = 000169, loss = 0.001883
grad_step = 000170, loss = 0.001889
grad_step = 000171, loss = 0.001876
grad_step = 000172, loss = 0.001854
grad_step = 000173, loss = 0.001839
grad_step = 000174, loss = 0.001837
grad_step = 000175, loss = 0.001836
grad_step = 000176, loss = 0.001833
grad_step = 000177, loss = 0.001826
grad_step = 000178, loss = 0.001821
grad_step = 000179, loss = 0.001820
grad_step = 000180, loss = 0.001820
grad_step = 000181, loss = 0.001825
grad_step = 000182, loss = 0.001835
grad_step = 000183, loss = 0.001841
grad_step = 000184, loss = 0.001837
grad_step = 000185, loss = 0.001820
grad_step = 000186, loss = 0.001800
grad_step = 000187, loss = 0.001790
grad_step = 000188, loss = 0.001790
grad_step = 000189, loss = 0.001796
grad_step = 000190, loss = 0.001803
grad_step = 000191, loss = 0.001808
grad_step = 000192, loss = 0.001803
grad_step = 000193, loss = 0.001793
grad_step = 000194, loss = 0.001781
grad_step = 000195, loss = 0.001771
grad_step = 000196, loss = 0.001765
grad_step = 000197, loss = 0.001764
grad_step = 000198, loss = 0.001768
grad_step = 000199, loss = 0.001774
grad_step = 000200, loss = 0.001784
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001795
grad_step = 000202, loss = 0.001812
grad_step = 000203, loss = 0.001825
grad_step = 000204, loss = 0.001852
grad_step = 000205, loss = 0.001903
grad_step = 000206, loss = 0.001992
grad_step = 000207, loss = 0.002159
grad_step = 000208, loss = 0.002304
grad_step = 000209, loss = 0.002334
grad_step = 000210, loss = 0.002058
grad_step = 000211, loss = 0.001804
grad_step = 000212, loss = 0.001783
grad_step = 000213, loss = 0.001933
grad_step = 000214, loss = 0.002058
grad_step = 000215, loss = 0.001953
grad_step = 000216, loss = 0.001769
grad_step = 000217, loss = 0.001777
grad_step = 000218, loss = 0.001907
grad_step = 000219, loss = 0.001897
grad_step = 000220, loss = 0.001786
grad_step = 000221, loss = 0.001755
grad_step = 000222, loss = 0.001809
grad_step = 000223, loss = 0.001831
grad_step = 000224, loss = 0.001783
grad_step = 000225, loss = 0.001727
grad_step = 000226, loss = 0.001754
grad_step = 000227, loss = 0.001803
grad_step = 000228, loss = 0.001763
grad_step = 000229, loss = 0.001717
grad_step = 000230, loss = 0.001733
grad_step = 000231, loss = 0.001758
grad_step = 000232, loss = 0.001750
grad_step = 000233, loss = 0.001722
grad_step = 000234, loss = 0.001713
grad_step = 000235, loss = 0.001730
grad_step = 000236, loss = 0.001739
grad_step = 000237, loss = 0.001718
grad_step = 000238, loss = 0.001702
grad_step = 000239, loss = 0.001712
grad_step = 000240, loss = 0.001721
grad_step = 000241, loss = 0.001712
grad_step = 000242, loss = 0.001700
grad_step = 000243, loss = 0.001697
grad_step = 000244, loss = 0.001702
grad_step = 000245, loss = 0.001705
grad_step = 000246, loss = 0.001700
grad_step = 000247, loss = 0.001691
grad_step = 000248, loss = 0.001688
grad_step = 000249, loss = 0.001692
grad_step = 000250, loss = 0.001694
grad_step = 000251, loss = 0.001688
grad_step = 000252, loss = 0.001683
grad_step = 000253, loss = 0.001681
grad_step = 000254, loss = 0.001681
grad_step = 000255, loss = 0.001682
grad_step = 000256, loss = 0.001681
grad_step = 000257, loss = 0.001677
grad_step = 000258, loss = 0.001673
grad_step = 000259, loss = 0.001672
grad_step = 000260, loss = 0.001673
grad_step = 000261, loss = 0.001673
grad_step = 000262, loss = 0.001673
grad_step = 000263, loss = 0.001675
grad_step = 000264, loss = 0.001686
grad_step = 000265, loss = 0.001710
grad_step = 000266, loss = 0.001764
grad_step = 000267, loss = 0.001825
grad_step = 000268, loss = 0.001913
grad_step = 000269, loss = 0.001924
grad_step = 000270, loss = 0.001899
grad_step = 000271, loss = 0.001817
grad_step = 000272, loss = 0.001726
grad_step = 000273, loss = 0.001670
grad_step = 000274, loss = 0.001668
grad_step = 000275, loss = 0.001710
grad_step = 000276, loss = 0.001757
grad_step = 000277, loss = 0.001769
grad_step = 000278, loss = 0.001736
grad_step = 000279, loss = 0.001680
grad_step = 000280, loss = 0.001644
grad_step = 000281, loss = 0.001647
grad_step = 000282, loss = 0.001673
grad_step = 000283, loss = 0.001696
grad_step = 000284, loss = 0.001697
grad_step = 000285, loss = 0.001682
grad_step = 000286, loss = 0.001664
grad_step = 000287, loss = 0.001649
grad_step = 000288, loss = 0.001643
grad_step = 000289, loss = 0.001640
grad_step = 000290, loss = 0.001638
grad_step = 000291, loss = 0.001640
grad_step = 000292, loss = 0.001644
grad_step = 000293, loss = 0.001651
grad_step = 000294, loss = 0.001653
grad_step = 000295, loss = 0.001649
grad_step = 000296, loss = 0.001639
grad_step = 000297, loss = 0.001629
grad_step = 000298, loss = 0.001620
grad_step = 000299, loss = 0.001617
grad_step = 000300, loss = 0.001615
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001615
grad_step = 000302, loss = 0.001613
grad_step = 000303, loss = 0.001610
grad_step = 000304, loss = 0.001607
grad_step = 000305, loss = 0.001606
grad_step = 000306, loss = 0.001608
grad_step = 000307, loss = 0.001611
grad_step = 000308, loss = 0.001617
grad_step = 000309, loss = 0.001627
grad_step = 000310, loss = 0.001643
grad_step = 000311, loss = 0.001673
grad_step = 000312, loss = 0.001726
grad_step = 000313, loss = 0.001815
grad_step = 000314, loss = 0.001961
grad_step = 000315, loss = 0.002112
grad_step = 000316, loss = 0.002228
grad_step = 000317, loss = 0.002128
grad_step = 000318, loss = 0.001871
grad_step = 000319, loss = 0.001646
grad_step = 000320, loss = 0.001628
grad_step = 000321, loss = 0.001771
grad_step = 000322, loss = 0.001858
grad_step = 000323, loss = 0.001807
grad_step = 000324, loss = 0.001678
grad_step = 000325, loss = 0.001613
grad_step = 000326, loss = 0.001671
grad_step = 000327, loss = 0.001736
grad_step = 000328, loss = 0.001714
grad_step = 000329, loss = 0.001627
grad_step = 000330, loss = 0.001587
grad_step = 000331, loss = 0.001629
grad_step = 000332, loss = 0.001672
grad_step = 000333, loss = 0.001658
grad_step = 000334, loss = 0.001604
grad_step = 000335, loss = 0.001577
grad_step = 000336, loss = 0.001589
grad_step = 000337, loss = 0.001616
grad_step = 000338, loss = 0.001623
grad_step = 000339, loss = 0.001602
grad_step = 000340, loss = 0.001573
grad_step = 000341, loss = 0.001556
grad_step = 000342, loss = 0.001567
grad_step = 000343, loss = 0.001587
grad_step = 000344, loss = 0.001590
grad_step = 000345, loss = 0.001573
grad_step = 000346, loss = 0.001550
grad_step = 000347, loss = 0.001544
grad_step = 000348, loss = 0.001553
grad_step = 000349, loss = 0.001562
grad_step = 000350, loss = 0.001562
grad_step = 000351, loss = 0.001553
grad_step = 000352, loss = 0.001542
grad_step = 000353, loss = 0.001536
grad_step = 000354, loss = 0.001535
grad_step = 000355, loss = 0.001537
grad_step = 000356, loss = 0.001539
grad_step = 000357, loss = 0.001540
grad_step = 000358, loss = 0.001538
grad_step = 000359, loss = 0.001532
grad_step = 000360, loss = 0.001526
grad_step = 000361, loss = 0.001521
grad_step = 000362, loss = 0.001519
grad_step = 000363, loss = 0.001519
grad_step = 000364, loss = 0.001519
grad_step = 000365, loss = 0.001520
grad_step = 000366, loss = 0.001519
grad_step = 000367, loss = 0.001518
grad_step = 000368, loss = 0.001518
grad_step = 000369, loss = 0.001517
grad_step = 000370, loss = 0.001516
grad_step = 000371, loss = 0.001516
grad_step = 000372, loss = 0.001516
grad_step = 000373, loss = 0.001517
grad_step = 000374, loss = 0.001520
grad_step = 000375, loss = 0.001527
grad_step = 000376, loss = 0.001541
grad_step = 000377, loss = 0.001565
grad_step = 000378, loss = 0.001608
grad_step = 000379, loss = 0.001680
grad_step = 000380, loss = 0.001799
grad_step = 000381, loss = 0.001956
grad_step = 000382, loss = 0.002135
grad_step = 000383, loss = 0.002207
grad_step = 000384, loss = 0.002079
grad_step = 000385, loss = 0.001770
grad_step = 000386, loss = 0.001524
grad_step = 000387, loss = 0.001530
grad_step = 000388, loss = 0.001708
grad_step = 000389, loss = 0.001842
grad_step = 000390, loss = 0.001758
grad_step = 000391, loss = 0.001569
grad_step = 000392, loss = 0.001491
grad_step = 000393, loss = 0.001585
grad_step = 000394, loss = 0.001686
grad_step = 000395, loss = 0.001654
grad_step = 000396, loss = 0.001528
grad_step = 000397, loss = 0.001470
grad_step = 000398, loss = 0.001528
grad_step = 000399, loss = 0.001597
grad_step = 000400, loss = 0.001576
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001496
grad_step = 000402, loss = 0.001462
grad_step = 000403, loss = 0.001497
grad_step = 000404, loss = 0.001538
grad_step = 000405, loss = 0.001528
grad_step = 000406, loss = 0.001482
grad_step = 000407, loss = 0.001460
grad_step = 000408, loss = 0.001481
grad_step = 000409, loss = 0.001511
grad_step = 000410, loss = 0.001518
grad_step = 000411, loss = 0.001507
grad_step = 000412, loss = 0.001509
grad_step = 000413, loss = 0.001548
grad_step = 000414, loss = 0.001595
grad_step = 000415, loss = 0.001610
grad_step = 000416, loss = 0.001571
grad_step = 000417, loss = 0.001504
grad_step = 000418, loss = 0.001459
grad_step = 000419, loss = 0.001455
grad_step = 000420, loss = 0.001476
grad_step = 000421, loss = 0.001491
grad_step = 000422, loss = 0.001483
grad_step = 000423, loss = 0.001464
grad_step = 000424, loss = 0.001449
grad_step = 000425, loss = 0.001450
grad_step = 000426, loss = 0.001454
grad_step = 000427, loss = 0.001453
grad_step = 000428, loss = 0.001448
grad_step = 000429, loss = 0.001440
grad_step = 000430, loss = 0.001436
grad_step = 000431, loss = 0.001432
grad_step = 000432, loss = 0.001429
grad_step = 000433, loss = 0.001426
grad_step = 000434, loss = 0.001424
grad_step = 000435, loss = 0.001424
grad_step = 000436, loss = 0.001423
grad_step = 000437, loss = 0.001421
grad_step = 000438, loss = 0.001415
grad_step = 000439, loss = 0.001410
grad_step = 000440, loss = 0.001406
grad_step = 000441, loss = 0.001403
grad_step = 000442, loss = 0.001400
grad_step = 000443, loss = 0.001399
grad_step = 000444, loss = 0.001398
grad_step = 000445, loss = 0.001398
grad_step = 000446, loss = 0.001399
grad_step = 000447, loss = 0.001400
grad_step = 000448, loss = 0.001401
grad_step = 000449, loss = 0.001402
grad_step = 000450, loss = 0.001403
grad_step = 000451, loss = 0.001405
grad_step = 000452, loss = 0.001410
grad_step = 000453, loss = 0.001420
grad_step = 000454, loss = 0.001440
grad_step = 000455, loss = 0.001473
grad_step = 000456, loss = 0.001533
grad_step = 000457, loss = 0.001623
grad_step = 000458, loss = 0.001764
grad_step = 000459, loss = 0.001921
grad_step = 000460, loss = 0.002055
grad_step = 000461, loss = 0.002038
grad_step = 000462, loss = 0.001832
grad_step = 000463, loss = 0.001557
grad_step = 000464, loss = 0.001397
grad_step = 000465, loss = 0.001461
grad_step = 000466, loss = 0.001628
grad_step = 000467, loss = 0.001719
grad_step = 000468, loss = 0.001608
grad_step = 000469, loss = 0.001435
grad_step = 000470, loss = 0.001395
grad_step = 000471, loss = 0.001497
grad_step = 000472, loss = 0.001563
grad_step = 000473, loss = 0.001506
grad_step = 000474, loss = 0.001386
grad_step = 000475, loss = 0.001350
grad_step = 000476, loss = 0.001412
grad_step = 000477, loss = 0.001469
grad_step = 000478, loss = 0.001456
grad_step = 000479, loss = 0.001404
grad_step = 000480, loss = 0.001371
grad_step = 000481, loss = 0.001385
grad_step = 000482, loss = 0.001408
grad_step = 000483, loss = 0.001406
grad_step = 000484, loss = 0.001382
grad_step = 000485, loss = 0.001354
grad_step = 000486, loss = 0.001345
grad_step = 000487, loss = 0.001350
grad_step = 000488, loss = 0.001355
grad_step = 000489, loss = 0.001346
grad_step = 000490, loss = 0.001333
grad_step = 000491, loss = 0.001323
grad_step = 000492, loss = 0.001320
grad_step = 000493, loss = 0.001321
grad_step = 000494, loss = 0.001321
grad_step = 000495, loss = 0.001316
grad_step = 000496, loss = 0.001312
grad_step = 000497, loss = 0.001307
grad_step = 000498, loss = 0.001303
grad_step = 000499, loss = 0.001300
grad_step = 000500, loss = 0.001297
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001297
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

  date_run                              2020-05-09 16:11:52.254111
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.205481
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 16:11:52.259521
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0944436
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 16:11:52.266180
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.134372
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 16:11:52.271195
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.435104
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
0   2020-05-09 16:11:28.991076  ...    mean_absolute_error
1   2020-05-09 16:11:28.995116  ...     mean_squared_error
2   2020-05-09 16:11:28.998343  ...  median_absolute_error
3   2020-05-09 16:11:29.001774  ...               r2_score
4   2020-05-09 16:11:37.437031  ...    mean_absolute_error
5   2020-05-09 16:11:37.440092  ...     mean_squared_error
6   2020-05-09 16:11:37.442724  ...  median_absolute_error
7   2020-05-09 16:11:37.445765  ...               r2_score
8   2020-05-09 16:11:52.254111  ...    mean_absolute_error
9   2020-05-09 16:11:52.259521  ...     mean_squared_error
10  2020-05-09 16:11:52.266180  ...  median_absolute_error
11  2020-05-09 16:11:52.271195  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:18, 126765.78it/s] 84%|████████▍ | 8306688/9912422 [00:00<00:08, 180974.36it/s]9920512it [00:00, 41072294.34it/s]                           
0it [00:00, ?it/s]32768it [00:00, 597251.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470445.05it/s]1654784it [00:00, 11946091.39it/s]                         
0it [00:00, ?it/s]8192it [00:00, 220485.11it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc04cf5e780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbfea6a29b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc04cf15e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbfea6a2da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc04cf15e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc04cf5ee80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc04cf5e780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbfff910cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbfea6a2da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbfff910cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc04cf15e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4fd97b31d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=abfa91b45717cdfaeb5e1a75dbbfd2c513a04850889fd6aeceedd45688080385
  Stored in directory: /tmp/pip-ephem-wheel-cache-afh85abu/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4fcf921080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4341760/17464789 [======>.......................] - ETA: 0s
13131776/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 16:13:13.137905: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 16:13:13.141126: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-05-09 16:13:13.141225: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5635162e6bb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 16:13:13.141235: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 9s - loss: 7.5133 - accuracy: 0.5100
 2000/25000 [=>............................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 3000/25000 [==>...........................] - ETA: 5s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 3s - loss: 7.6250 - accuracy: 0.5027
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6905 - accuracy: 0.4984
10000/25000 [===========>..................] - ETA: 2s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 2s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 2s - loss: 7.6411 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6502 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 1s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6852 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 0s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6569 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 6s 227us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 16:13:24.064804
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 16:13:24.064804  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 16:13:28.859906: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 16:13:28.864401: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-05-09 16:13:28.865148: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561c47904330 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 16:13:28.865204: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1a0e9f5cc0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 916ms/step - loss: 1.2138 - crf_viterbi_accuracy: 0.1200 - val_loss: 1.1478 - val_crf_viterbi_accuracy: 0.6800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f19e993af60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 9s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 3000/25000 [==>...........................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 4s - loss: 7.5325 - accuracy: 0.5088
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5440 - accuracy: 0.5080
 7000/25000 [=======>......................] - ETA: 3s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6072 - accuracy: 0.5039
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6114 - accuracy: 0.5036
11000/25000 [============>.................] - ETA: 2s - loss: 7.5914 - accuracy: 0.5049
12000/25000 [=============>................] - ETA: 2s - loss: 7.5874 - accuracy: 0.5052
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5782 - accuracy: 0.5058
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6108 - accuracy: 0.5036
15000/25000 [=================>............] - ETA: 2s - loss: 7.6073 - accuracy: 0.5039
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6245 - accuracy: 0.5027
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6188 - accuracy: 0.5031
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6351 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 0s - loss: 7.6413 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6403 - accuracy: 0.5017
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6374 - accuracy: 0.5019
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
25000/25000 [==============================] - 6s 229us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f19a56c9278> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<43:38:23, 5.49kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<30:46:45, 7.78kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<21:35:48, 11.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:07:12, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<10:33:13, 22.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.80M/862M [00:02<7:20:09, 32.3kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:02<5:07:14, 46.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:02<3:33:42, 65.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:02<2:28:41, 93.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:02<1:43:30, 134kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:02<1:12:16, 191kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<50:24, 273kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<35:10, 389kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.0M/862M [00:03<24:36, 553kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:03<17:14, 784kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:05<13:56, 965kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:05<13:25, 1.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:05<10:19, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:05<07:26, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:07<10:56, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:07<09:07, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:44, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:09<07:39, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<08:06, 1.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<06:17, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:09<04:33, 2.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:11<11:45, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:11<09:22, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<06:53, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:13<07:55, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:13<08:14, 1.60MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<06:27, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:13<04:40, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:15<1:36:48, 135kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:15<1:09:06, 189kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<48:37, 269kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:17<36:58, 352kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:17<27:12, 479kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<19:17, 674kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<16:33, 783kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:19<12:55, 1.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<09:21, 1.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<09:35, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:21<09:28, 1.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:21<07:18, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<05:16, 2.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<34:10, 375kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:23<25:16, 507kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:23<18:00, 711kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<15:26, 826kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:25<13:32, 942kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:25<10:08, 1.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:14, 1.75MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<35:24, 358kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<26:06, 486kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<18:32, 683kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<15:46, 800kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<13:43, 919kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<10:10, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<07:15, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<12:16, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:55, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:16, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<07:53, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<08:11, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:17, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<04:34, 2.71MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<08:19, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:08, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:16, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:20, 2.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<18:44, 659kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<14:26, 855kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:21, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<10:01, 1.22MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:37, 1.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:19, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:15, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<12:08, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:47, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:07, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:43, 1.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<08:00, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:10, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<04:27, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<10:05, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:21, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:07, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<06:59, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:26, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:50, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:12, 2.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<30:33, 390kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<22:38, 526kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<16:05, 739kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<13:55, 851kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<12:16, 965kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:13, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<06:34, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<32:54, 358kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<24:16, 485kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<17:14, 682kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<14:39, 799kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<12:47, 916kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:28, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<06:46, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<09:50, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:07, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:56, 1.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<06:45, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<07:12, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:39, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:06, 2.81MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<30:13, 381kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<22:22, 514kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<15:53, 723kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<13:41, 836kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<10:46, 1.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:49, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<08:04, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<08:04, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:15, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:30, 2.51MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<38:59, 290kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<28:29, 397kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<20:11, 558kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<16:37, 676kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<14:01, 801kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:23, 1.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<07:23, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<38:43, 288kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<28:17, 395kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<20:00, 557kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<16:29, 673kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<13:54, 798kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:13, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:17, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<09:47, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<08:01, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:51, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:35, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<06:56, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:21, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:53, 2.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<07:49, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:37, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:55, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:52, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:16, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<03:58, 2.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:12, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:56, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:43, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:25, 3.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<28:05, 380kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<20:47, 514kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<14:47, 720kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<12:42, 835kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<11:10, 950kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<08:22, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<05:57, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<29:29, 358kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<21:45, 484kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<15:26, 681kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<13:08, 797kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<11:25, 917kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<08:29, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<06:01, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<22:18, 466kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<17:53, 582kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<13:03, 796kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<09:13, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<30:12, 342kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<22:13, 465kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<15:45, 654kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<13:18, 771kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<11:31, 891kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<08:31, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<07:49, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:33, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:51, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<05:40, 1.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<06:07, 1.65MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:49, 2.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:29, 2.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<32:39, 308kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<23:53, 421kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<16:54, 593kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<14:07, 707kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<12:01, 831kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<08:57, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:21, 1.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<34:13, 290kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<24:58, 397kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<17:41, 559kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<14:38, 674kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<11:17, 872kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<08:06, 1.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:52, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<07:30, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<05:41, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:06, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<07:32, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<06:18, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:39, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:25, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<05:51, 1.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<04:32, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<03:20, 2.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:35, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<04:53, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<03:40, 2.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:45, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:19, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<03:13, 2.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:27, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:07, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<04:04, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<02:57, 3.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:39, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<07:02, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<05:07, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:42, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:53, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:33, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:17, 2.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<07:23, 1.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:07, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:31, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<05:16, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:27, 2.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:33, 2.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:08, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:07, 2.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:17, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:56, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<02:56, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:09, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:51, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<02:55, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:03, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:45, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<03:43, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<02:46, 3.19MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:38, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:11, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:09, 2.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:10, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<03:49, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:51, 3.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:40, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:40, 2.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<02:43, 3.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<04:29, 1.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<04:03, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:01, 2.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:13, 3.85MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<48:21, 177kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<35:35, 240kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<25:19, 337kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<17:45, 478kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<17:26, 486kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<13:06, 646kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<09:20, 904kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<06:37, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<2:53:26, 48.5kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<2:03:09, 68.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<1:26:31, 97.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<1:00:18, 138kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<1:41:29, 82.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<1:11:53, 116kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<50:22, 165kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<36:57, 224kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<27:35, 300kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<19:39, 420kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<13:47, 596kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<16:00, 513kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<12:03, 680kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<08:35, 951kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<07:52, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<06:22, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:38, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<05:04, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:23, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:16, 2.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:05, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<04:33, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:33, 2.24MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:34, 3.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<07:00, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:44, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:11, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:41, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:53, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:49, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:44, 2.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<42:45, 182kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<30:41, 254kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<21:35, 359kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<16:51, 458kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<13:25, 575kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<09:44, 791kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:52, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<10:49, 707kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<08:23, 912kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:03, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:55, 1.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:45, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:22, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:09, 2.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<05:13, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:27, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:17, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:58, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<04:19, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:20, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<02:26, 3.04MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<04:55, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<04:12, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:08, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:51, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<04:16, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<03:22, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:26, 2.97MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<23:25, 309kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<17:08, 422kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<12:07, 594kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<10:07, 709kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<08:36, 833kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<06:19, 1.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:30, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<06:41, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<05:26, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:58, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:22, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:49, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<02:51, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:33, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:58, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<03:09, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:16, 3.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<22:17, 309kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<16:19, 422kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<11:33, 594kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<09:33, 714kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<08:06, 842kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<05:58, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:13, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<39:03, 173kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<28:02, 241kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<19:42, 341kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<15:13, 439kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<12:04, 553kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<08:44, 764kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:11, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<06:54, 957kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<05:32, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<04:01, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<04:19, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:43, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:45, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:22, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:02, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:17, 2.81MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:02, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:46, 2.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:00, 3.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<17:19, 366kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<12:47, 495kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<09:05, 694kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<07:44, 811kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<06:44, 929kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:00, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:33, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<06:20, 978kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<05:05, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:41, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:57, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<04:04, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:10, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:16, 2.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<15:36, 389kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<11:34, 524kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<08:12, 736kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<07:04, 848kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<06:13, 964kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:39, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:18, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<17:08, 346kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<12:37, 469kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:56, 660kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<07:32, 777kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<06:31, 897kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:52, 1.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<03:26, 1.68MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<15:47, 367kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<11:40, 495kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<08:17, 695kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<07:03, 811kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<06:09, 927kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:34, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<03:14, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<07:28, 757kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<05:49, 969kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:11, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<04:10, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<04:07, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<03:07, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:15, 2.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<03:50, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:16, 1.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:25, 2.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:54, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:28, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<01:47, 3.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<04:06, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:26, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:31, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:57, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<03:12, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:28, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<01:50, 2.86MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<02:44, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<02:23, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<01:59, 2.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<01:27, 3.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:18, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:25, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:37, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<01:58, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<02:34, 1.99MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:19, 2.19MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<01:44, 2.91MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<02:21, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:42, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:07, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:33, 3.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<03:08, 1.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<02:43, 1.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:00, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:30, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:47, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:12, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:35, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<08:08, 592kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<06:12, 776kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:25, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<04:08, 1.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<03:55, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:57, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:07, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<03:30, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<02:57, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:10, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<02:33, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<02:46, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:33, 2.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:54, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:30, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:51, 2.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:18, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:34, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:01, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:27, 3.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:36, 957kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<03:41, 1.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:41, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:50, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:55, 1.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:16, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:37, 2.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<06:02, 708kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<04:41, 909kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<03:23, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:17, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<02:44, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:01, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:20, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<02:31, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<01:58, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:24, 2.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<10:27, 389kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<07:44, 524kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<05:29, 734kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<04:42, 848kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:08, 964kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<03:04, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:11, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:19, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:44, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:59, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:15, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:23, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:50, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:20, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:45, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:19, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:42, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:01, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:12, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:44, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:14, 2.93MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<12:26, 293kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<09:05, 401kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:24, 565kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<05:14, 683kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:26, 805kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:15, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<02:18, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<03:31, 996kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:50, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:03, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:28, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<05:26, 633kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<05:54, 582kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<04:37, 742kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:21, 1.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:54, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<03:00, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:20, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:39, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:27, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:40, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:03, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:29, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:52, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<02:11, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:44, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:15, 2.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:13, 1.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:22, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:50, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:20, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:42, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<02:01, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:37, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:10, 2.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:06, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:13, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:42, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:14, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:46, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:57, 1.50MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:31, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:09, 2.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:30, 1.91MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<02:09, 1.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:47, 1.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:20, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:01, 2.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<00:45, 3.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<05:31, 510kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<04:55, 572kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<03:42, 759kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:37, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:30, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:43, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:09, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:32, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:07, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<04:02, 663kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<03:46, 710kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:52, 931kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:02, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:27, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<05:14, 498kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<04:36, 567kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:23, 765kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:25, 1.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:43, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<03:07, 813kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<03:04, 824kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<02:21, 1.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:41, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:50, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<02:05, 1.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:40, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:12, 2.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:28, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:48, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:26, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:02, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:46, 3.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<02:33, 914kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<02:33, 912kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:58, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:24, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:36, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:48, 1.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:24, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:01, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:45, 2.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<02:35, 850kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:28, 885kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:54, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:21, 1.59MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:31, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:45, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:23, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:00, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:33, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:13, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:53, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:39, 3.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<02:33, 777kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<02:26, 814kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:33<01:51, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:18, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:27, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:38, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:16, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:55, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:40, 2.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<02:21, 786kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<02:12, 835kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:41, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:11, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:20, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:28, 1.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:08, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:49, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:36, 2.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<31:49, 53.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<22:48, 74.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<15:59, 106kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<11:04, 151kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<07:39, 215kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<07:18, 225kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<05:38, 291kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<04:03, 402kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<02:49, 568kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<02:22, 663kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<02:08, 733kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:35, 978kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:07, 1.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:12, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:17, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<01:01, 1.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:43, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:54, 1.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<01:04, 1.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:51, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:36, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<00:48, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<01:00, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:47, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:34, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:45, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:55, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:32, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:43, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<00:52, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:41, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:29, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:22, 3.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:23, 829kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:21, 854kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:01, 1.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:43, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:31, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:17, 838kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:14, 876kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:56, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:39, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:44, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:49, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:39, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:27, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:35, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<00:33, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:23, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:31, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:38, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:30, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:21, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:29, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:35, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:27, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:19, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:26, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:31, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:24, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:17, 2.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:23, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:28, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:15, 2.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:10, 3.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<02:23, 253kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:49, 329kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<01:18, 455kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:52, 641kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:44, 713kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:39, 798kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:29, 1.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:19, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:22, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:23, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:17, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:08, 2.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<04:32, 86.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<03:15, 120kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<02:15, 170kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<01:26, 242kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<01:03, 307kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:49, 395kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:34, 545kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:21, 767kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:20, 761kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:17, 847kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:12, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.69MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1141/400000 [00:00<00:34, 11404.90it/s]  1%|          | 2267/400000 [00:00<00:35, 11359.12it/s]  1%|          | 3394/400000 [00:00<00:34, 11331.62it/s]  1%|          | 4430/400000 [00:00<00:35, 11019.40it/s]  1%|▏         | 5604/400000 [00:00<00:35, 11224.58it/s]  2%|▏         | 6759/400000 [00:00<00:34, 11319.21it/s]  2%|▏         | 7932/400000 [00:00<00:34, 11438.34it/s]  2%|▏         | 9094/400000 [00:00<00:34, 11489.57it/s]  3%|▎         | 10200/400000 [00:00<00:34, 11355.31it/s]  3%|▎         | 11359/400000 [00:01<00:34, 11420.93it/s]  3%|▎         | 12499/400000 [00:01<00:33, 11413.93it/s]  3%|▎         | 13709/400000 [00:01<00:33, 11609.54it/s]  4%|▎         | 14908/400000 [00:01<00:32, 11718.86it/s]  4%|▍         | 16070/400000 [00:01<00:33, 11576.44it/s]  4%|▍         | 17263/400000 [00:01<00:32, 11679.44it/s]  5%|▍         | 18491/400000 [00:01<00:32, 11852.78it/s]  5%|▍         | 19674/400000 [00:01<00:32, 11744.37it/s]  5%|▌         | 20847/400000 [00:01<00:32, 11663.39it/s]  6%|▌         | 22013/400000 [00:01<00:33, 11215.98it/s]  6%|▌         | 23138/400000 [00:02<00:34, 11026.63it/s]  6%|▌         | 24255/400000 [00:02<00:33, 11067.75it/s]  6%|▋         | 25467/400000 [00:02<00:32, 11361.77it/s]  7%|▋         | 26610/400000 [00:02<00:32, 11379.14it/s]  7%|▋         | 27751/400000 [00:02<00:33, 11176.24it/s]  7%|▋         | 28943/400000 [00:02<00:32, 11387.50it/s]  8%|▊         | 30160/400000 [00:02<00:31, 11610.55it/s]  8%|▊         | 31337/400000 [00:02<00:31, 11657.65it/s]  8%|▊         | 32505/400000 [00:02<00:32, 11309.21it/s]  8%|▊         | 33640/400000 [00:02<00:32, 11315.69it/s]  9%|▊         | 34775/400000 [00:03<00:32, 11284.35it/s]  9%|▉         | 35951/400000 [00:03<00:31, 11420.91it/s]  9%|▉         | 37095/400000 [00:03<00:31, 11422.70it/s] 10%|▉         | 38241/400000 [00:03<00:31, 11430.99it/s] 10%|▉         | 39385/400000 [00:03<00:31, 11325.15it/s] 10%|█         | 40552/400000 [00:03<00:31, 11424.88it/s] 10%|█         | 41696/400000 [00:03<00:31, 11325.74it/s] 11%|█         | 42830/400000 [00:03<00:32, 11092.55it/s] 11%|█         | 43949/400000 [00:03<00:32, 11118.35it/s] 11%|█▏        | 45072/400000 [00:03<00:31, 11149.77it/s] 12%|█▏        | 46277/400000 [00:04<00:31, 11405.26it/s] 12%|█▏        | 47434/400000 [00:04<00:30, 11453.29it/s] 12%|█▏        | 48581/400000 [00:04<00:30, 11353.92it/s] 12%|█▏        | 49718/400000 [00:04<00:31, 11038.24it/s] 13%|█▎        | 50828/400000 [00:04<00:31, 11054.76it/s] 13%|█▎        | 52019/400000 [00:04<00:30, 11296.89it/s] 13%|█▎        | 53228/400000 [00:04<00:30, 11521.22it/s] 14%|█▎        | 54429/400000 [00:04<00:29, 11663.26it/s] 14%|█▍        | 55633/400000 [00:04<00:29, 11771.22it/s] 14%|█▍        | 56813/400000 [00:04<00:29, 11493.43it/s] 15%|█▍        | 58038/400000 [00:05<00:29, 11707.49it/s] 15%|█▍        | 59245/400000 [00:05<00:28, 11813.49it/s] 15%|█▌        | 60450/400000 [00:05<00:28, 11882.95it/s] 15%|█▌        | 61641/400000 [00:05<00:29, 11367.29it/s] 16%|█▌        | 62784/400000 [00:05<00:30, 11129.34it/s] 16%|█▌        | 63952/400000 [00:05<00:29, 11285.46it/s] 16%|█▋        | 65169/400000 [00:05<00:29, 11536.00it/s] 17%|█▋        | 66348/400000 [00:05<00:28, 11609.99it/s] 17%|█▋        | 67513/400000 [00:05<00:29, 11193.74it/s] 17%|█▋        | 68687/400000 [00:06<00:29, 11351.15it/s] 17%|█▋        | 69895/400000 [00:06<00:28, 11558.32it/s] 18%|█▊        | 71111/400000 [00:06<00:28, 11732.00it/s] 18%|█▊        | 72288/400000 [00:06<00:27, 11704.14it/s] 18%|█▊        | 73461/400000 [00:06<00:28, 11558.13it/s] 19%|█▊        | 74638/400000 [00:06<00:28, 11619.81it/s] 19%|█▉        | 75844/400000 [00:06<00:27, 11748.37it/s] 19%|█▉        | 77055/400000 [00:06<00:27, 11853.31it/s] 20%|█▉        | 78242/400000 [00:06<00:27, 11740.57it/s] 20%|█▉        | 79418/400000 [00:06<00:27, 11589.42it/s] 20%|██        | 80579/400000 [00:07<00:27, 11565.35it/s] 20%|██        | 81737/400000 [00:07<00:27, 11388.45it/s] 21%|██        | 82919/400000 [00:07<00:27, 11514.17it/s] 21%|██        | 84130/400000 [00:07<00:27, 11684.99it/s] 21%|██▏       | 85300/400000 [00:07<00:27, 11391.18it/s] 22%|██▏       | 86442/400000 [00:07<00:27, 11390.40it/s] 22%|██▏       | 87620/400000 [00:07<00:27, 11503.44it/s] 22%|██▏       | 88772/400000 [00:07<00:27, 11180.23it/s] 22%|██▏       | 89921/400000 [00:07<00:27, 11269.09it/s] 23%|██▎       | 91066/400000 [00:07<00:27, 11321.58it/s] 23%|██▎       | 92239/400000 [00:08<00:26, 11439.93it/s] 23%|██▎       | 93449/400000 [00:08<00:26, 11628.99it/s] 24%|██▎       | 94630/400000 [00:08<00:26, 11681.26it/s] 24%|██▍       | 95809/400000 [00:08<00:25, 11713.31it/s] 24%|██▍       | 96982/400000 [00:08<00:26, 11510.05it/s] 25%|██▍       | 98153/400000 [00:08<00:26, 11567.83it/s] 25%|██▍       | 99368/400000 [00:08<00:25, 11735.92it/s] 25%|██▌       | 100543/400000 [00:08<00:26, 11377.33it/s] 25%|██▌       | 101685/400000 [00:08<00:26, 11142.52it/s] 26%|██▌       | 102803/400000 [00:08<00:27, 10784.08it/s] 26%|██▌       | 103946/400000 [00:09<00:26, 10969.65it/s] 26%|██▋       | 105152/400000 [00:09<00:26, 11274.14it/s] 27%|██▋       | 106317/400000 [00:09<00:25, 11382.45it/s] 27%|██▋       | 107508/400000 [00:09<00:25, 11535.12it/s] 27%|██▋       | 108665/400000 [00:09<00:25, 11513.17it/s] 27%|██▋       | 109836/400000 [00:09<00:25, 11570.73it/s] 28%|██▊       | 111053/400000 [00:09<00:24, 11743.97it/s] 28%|██▊       | 112251/400000 [00:09<00:24, 11810.46it/s] 28%|██▊       | 113461/400000 [00:09<00:24, 11895.13it/s] 29%|██▊       | 114652/400000 [00:09<00:24, 11725.36it/s] 29%|██▉       | 115826/400000 [00:10<00:24, 11648.70it/s] 29%|██▉       | 116992/400000 [00:10<00:24, 11643.74it/s] 30%|██▉       | 118158/400000 [00:10<00:24, 11411.58it/s] 30%|██▉       | 119301/400000 [00:10<00:26, 10537.52it/s] 30%|███       | 120370/400000 [00:10<00:28, 9908.07it/s]  30%|███       | 121379/400000 [00:10<00:29, 9566.10it/s] 31%|███       | 122351/400000 [00:10<00:29, 9361.57it/s] 31%|███       | 123299/400000 [00:10<00:29, 9253.45it/s] 31%|███       | 124233/400000 [00:10<00:30, 9153.44it/s] 31%|███▏      | 125158/400000 [00:11<00:29, 9180.88it/s] 32%|███▏      | 126244/400000 [00:11<00:28, 9627.22it/s] 32%|███▏      | 127444/400000 [00:11<00:26, 10232.95it/s] 32%|███▏      | 128575/400000 [00:11<00:25, 10533.58it/s] 32%|███▏      | 129711/400000 [00:11<00:25, 10767.27it/s] 33%|███▎      | 130799/400000 [00:11<00:25, 10673.10it/s] 33%|███▎      | 131962/400000 [00:11<00:24, 10941.82it/s] 33%|███▎      | 133093/400000 [00:11<00:24, 11046.82it/s] 34%|███▎      | 134274/400000 [00:11<00:23, 11263.25it/s] 34%|███▍      | 135406/400000 [00:11<00:23, 11208.04it/s] 34%|███▍      | 136531/400000 [00:12<00:24, 10857.91it/s] 34%|███▍      | 137622/400000 [00:12<00:25, 10457.73it/s] 35%|███▍      | 138719/400000 [00:12<00:24, 10603.60it/s] 35%|███▍      | 139867/400000 [00:12<00:23, 10850.60it/s] 35%|███▌      | 140957/400000 [00:12<00:24, 10660.01it/s] 36%|███▌      | 142107/400000 [00:12<00:23, 10898.28it/s] 36%|███▌      | 143291/400000 [00:12<00:22, 11163.91it/s] 36%|███▌      | 144445/400000 [00:12<00:22, 11272.39it/s] 36%|███▋      | 145640/400000 [00:12<00:22, 11466.32it/s] 37%|███▋      | 146790/400000 [00:13<00:22, 11207.23it/s] 37%|███▋      | 147964/400000 [00:13<00:22, 11360.92it/s] 37%|███▋      | 149149/400000 [00:13<00:21, 11500.77it/s] 38%|███▊      | 150357/400000 [00:13<00:21, 11667.49it/s] 38%|███▊      | 151528/400000 [00:13<00:21, 11678.18it/s] 38%|███▊      | 152698/400000 [00:13<00:21, 11677.25it/s] 38%|███▊      | 153867/400000 [00:13<00:21, 11633.01it/s] 39%|███▉      | 155070/400000 [00:13<00:20, 11746.46it/s] 39%|███▉      | 156246/400000 [00:13<00:21, 11463.47it/s] 39%|███▉      | 157395/400000 [00:13<00:21, 11286.06it/s] 40%|███▉      | 158526/400000 [00:14<00:21, 11258.24it/s] 40%|███▉      | 159662/400000 [00:14<00:21, 11287.54it/s] 40%|████      | 160853/400000 [00:14<00:20, 11464.82it/s] 41%|████      | 162048/400000 [00:14<00:20, 11605.68it/s] 41%|████      | 163226/400000 [00:14<00:20, 11656.92it/s] 41%|████      | 164393/400000 [00:14<00:20, 11599.07it/s] 41%|████▏     | 165554/400000 [00:14<00:21, 11116.80it/s] 42%|████▏     | 166725/400000 [00:14<00:20, 11287.47it/s] 42%|████▏     | 167929/400000 [00:14<00:20, 11502.18it/s] 42%|████▏     | 169131/400000 [00:14<00:19, 11652.36it/s] 43%|████▎     | 170300/400000 [00:15<00:20, 11278.73it/s] 43%|████▎     | 171433/400000 [00:15<00:20, 11275.67it/s] 43%|████▎     | 172649/400000 [00:15<00:19, 11524.98it/s] 43%|████▎     | 173832/400000 [00:15<00:19, 11612.29it/s] 44%|████▎     | 174996/400000 [00:15<00:19, 11469.23it/s] 44%|████▍     | 176146/400000 [00:15<00:20, 11017.44it/s] 44%|████▍     | 177319/400000 [00:15<00:19, 11219.59it/s] 45%|████▍     | 178446/400000 [00:15<00:20, 10956.66it/s] 45%|████▍     | 179547/400000 [00:15<00:20, 10944.32it/s] 45%|████▌     | 180729/400000 [00:16<00:19, 11189.94it/s] 45%|████▌     | 181852/400000 [00:16<00:19, 11127.57it/s] 46%|████▌     | 183036/400000 [00:16<00:19, 11330.41it/s] 46%|████▌     | 184224/400000 [00:16<00:18, 11488.58it/s] 46%|████▋     | 185376/400000 [00:16<00:18, 11492.04it/s] 47%|████▋     | 186594/400000 [00:16<00:18, 11686.87it/s] 47%|████▋     | 187765/400000 [00:16<00:18, 11519.87it/s] 47%|████▋     | 188932/400000 [00:16<00:18, 11563.35it/s] 48%|████▊     | 190090/400000 [00:16<00:18, 11498.06it/s] 48%|████▊     | 191242/400000 [00:16<00:18, 11503.62it/s] 48%|████▊     | 192436/400000 [00:17<00:17, 11629.69it/s] 48%|████▊     | 193600/400000 [00:17<00:17, 11497.00it/s] 49%|████▊     | 194751/400000 [00:17<00:18, 10883.67it/s] 49%|████▉     | 195847/400000 [00:17<00:19, 10686.70it/s] 49%|████▉     | 197014/400000 [00:17<00:18, 10962.35it/s] 50%|████▉     | 198214/400000 [00:17<00:17, 11252.71it/s] 50%|████▉     | 199346/400000 [00:17<00:17, 11230.53it/s] 50%|█████     | 200510/400000 [00:17<00:17, 11350.29it/s] 50%|█████     | 201675/400000 [00:17<00:17, 11438.57it/s] 51%|█████     | 202847/400000 [00:17<00:17, 11519.72it/s] 51%|█████     | 204027/400000 [00:18<00:16, 11599.87it/s] 51%|█████▏    | 205189/400000 [00:18<00:17, 11366.21it/s] 52%|█████▏    | 206328/400000 [00:18<00:17, 11207.15it/s] 52%|█████▏    | 207518/400000 [00:18<00:16, 11405.88it/s] 52%|█████▏    | 208692/400000 [00:18<00:16, 11503.47it/s] 52%|█████▏    | 209905/400000 [00:18<00:16, 11682.59it/s] 53%|█████▎    | 211076/400000 [00:18<00:16, 11550.90it/s] 53%|█████▎    | 212298/400000 [00:18<00:15, 11742.01it/s] 53%|█████▎    | 213475/400000 [00:18<00:15, 11722.99it/s] 54%|█████▎    | 214649/400000 [00:18<00:16, 11528.84it/s] 54%|█████▍    | 215804/400000 [00:19<00:16, 11386.17it/s] 54%|█████▍    | 216945/400000 [00:19<00:16, 11346.30it/s] 55%|█████▍    | 218142/400000 [00:19<00:15, 11524.91it/s] 55%|█████▍    | 219357/400000 [00:19<00:15, 11705.13it/s] 55%|█████▌    | 220556/400000 [00:19<00:15, 11786.80it/s] 55%|█████▌    | 221774/400000 [00:19<00:14, 11901.73it/s] 56%|█████▌    | 222966/400000 [00:19<00:15, 11760.95it/s] 56%|█████▌    | 224185/400000 [00:19<00:14, 11886.34it/s] 56%|█████▋    | 225413/400000 [00:19<00:14, 11999.69it/s] 57%|█████▋    | 226626/400000 [00:19<00:14, 12035.95it/s] 57%|█████▋    | 227831/400000 [00:20<00:14, 11977.83it/s] 57%|█████▋    | 229030/400000 [00:20<00:14, 11801.34it/s] 58%|█████▊    | 230258/400000 [00:20<00:14, 11940.06it/s] 58%|█████▊    | 231455/400000 [00:20<00:14, 11947.09it/s] 58%|█████▊    | 232651/400000 [00:20<00:14, 11767.27it/s] 58%|█████▊    | 233829/400000 [00:20<00:14, 11656.62it/s] 59%|█████▊    | 234996/400000 [00:20<00:14, 11311.16it/s] 59%|█████▉    | 236131/400000 [00:20<00:14, 10995.18it/s] 59%|█████▉    | 237295/400000 [00:20<00:14, 11178.28it/s] 60%|█████▉    | 238514/400000 [00:21<00:14, 11461.93it/s] 60%|█████▉    | 239715/400000 [00:21<00:13, 11618.67it/s] 60%|██████    | 240881/400000 [00:21<00:13, 11507.18it/s] 61%|██████    | 242035/400000 [00:21<00:14, 11257.46it/s] 61%|██████    | 243246/400000 [00:21<00:13, 11499.29it/s] 61%|██████    | 244461/400000 [00:21<00:13, 11685.60it/s] 61%|██████▏   | 245678/400000 [00:21<00:13, 11824.87it/s] 62%|██████▏   | 246864/400000 [00:21<00:13, 11326.19it/s] 62%|██████▏   | 248063/400000 [00:21<00:13, 11516.88it/s] 62%|██████▏   | 249277/400000 [00:21<00:12, 11694.73it/s] 63%|██████▎   | 250451/400000 [00:22<00:12, 11593.64it/s] 63%|██████▎   | 251614/400000 [00:22<00:12, 11477.12it/s] 63%|██████▎   | 252814/400000 [00:22<00:12, 11627.63it/s] 63%|██████▎   | 253979/400000 [00:22<00:12, 11593.47it/s] 64%|██████▍   | 255140/400000 [00:22<00:12, 11347.53it/s] 64%|██████▍   | 256278/400000 [00:22<00:12, 11144.15it/s] 64%|██████▍   | 257463/400000 [00:22<00:12, 11345.32it/s] 65%|██████▍   | 258633/400000 [00:22<00:12, 11448.31it/s] 65%|██████▍   | 259845/400000 [00:22<00:12, 11641.82it/s] 65%|██████▌   | 261066/400000 [00:22<00:11, 11806.63it/s] 66%|██████▌   | 262260/400000 [00:23<00:11, 11845.03it/s] 66%|██████▌   | 263447/400000 [00:23<00:11, 11521.61it/s] 66%|██████▌   | 264605/400000 [00:23<00:11, 11538.08it/s] 66%|██████▋   | 265798/400000 [00:23<00:11, 11652.07it/s] 67%|██████▋   | 267024/400000 [00:23<00:11, 11825.80it/s] 67%|██████▋   | 268211/400000 [00:23<00:11, 11836.49it/s] 67%|██████▋   | 269396/400000 [00:23<00:11, 11668.79it/s] 68%|██████▊   | 270565/400000 [00:23<00:11, 11656.03it/s] 68%|██████▊   | 271749/400000 [00:23<00:10, 11708.35it/s] 68%|██████▊   | 272921/400000 [00:23<00:10, 11579.88it/s] 69%|██████▊   | 274080/400000 [00:24<00:11, 11411.59it/s] 69%|██████▉   | 275223/400000 [00:24<00:10, 11387.75it/s] 69%|██████▉   | 276363/400000 [00:24<00:10, 11327.77it/s] 69%|██████▉   | 277497/400000 [00:24<00:10, 11172.54it/s] 70%|██████▉   | 278657/400000 [00:24<00:10, 11296.77it/s] 70%|██████▉   | 279880/400000 [00:24<00:10, 11560.99it/s] 70%|███████   | 281039/400000 [00:24<00:10, 11505.83it/s] 71%|███████   | 282249/400000 [00:24<00:10, 11677.59it/s] 71%|███████   | 283472/400000 [00:24<00:09, 11836.63it/s] 71%|███████   | 284669/400000 [00:25<00:09, 11874.94it/s] 71%|███████▏  | 285890/400000 [00:25<00:09, 11970.87it/s] 72%|███████▏  | 287089/400000 [00:25<00:09, 11712.46it/s] 72%|███████▏  | 288305/400000 [00:25<00:09, 11841.29it/s] 72%|███████▏  | 289527/400000 [00:25<00:09, 11949.98it/s] 73%|███████▎  | 290724/400000 [00:25<00:09, 11572.16it/s] 73%|███████▎  | 291885/400000 [00:25<00:09, 11487.42it/s] 73%|███████▎  | 293037/400000 [00:25<00:09, 11252.13it/s] 74%|███████▎  | 294197/400000 [00:25<00:09, 11351.20it/s] 74%|███████▍  | 295402/400000 [00:25<00:09, 11549.92it/s] 74%|███████▍  | 296618/400000 [00:26<00:08, 11726.40it/s] 74%|███████▍  | 297824/400000 [00:26<00:08, 11822.43it/s] 75%|███████▍  | 299009/400000 [00:26<00:08, 11668.37it/s] 75%|███████▌  | 300178/400000 [00:26<00:08, 11158.15it/s] 75%|███████▌  | 301300/400000 [00:26<00:09, 10273.19it/s] 76%|███████▌  | 302346/400000 [00:26<00:09, 10041.69it/s] 76%|███████▌  | 303364/400000 [00:26<00:09, 9725.43it/s]  76%|███████▌  | 304429/400000 [00:26<00:09, 9984.23it/s] 76%|███████▋  | 305638/400000 [00:26<00:08, 10532.48it/s] 77%|███████▋  | 306711/400000 [00:27<00:08, 10588.01it/s] 77%|███████▋  | 307869/400000 [00:27<00:08, 10866.82it/s] 77%|███████▋  | 309071/400000 [00:27<00:08, 11186.69it/s] 78%|███████▊  | 310211/400000 [00:27<00:07, 11248.09it/s] 78%|███████▊  | 311343/400000 [00:27<00:07, 11185.45it/s] 78%|███████▊  | 312467/400000 [00:27<00:09, 9507.94it/s]  78%|███████▊  | 313491/400000 [00:27<00:08, 9714.81it/s] 79%|███████▊  | 314497/400000 [00:27<00:08, 9731.77it/s] 79%|███████▉  | 315526/400000 [00:27<00:08, 9890.96it/s] 79%|███████▉  | 316549/400000 [00:27<00:08, 9988.50it/s] 79%|███████▉  | 317592/400000 [00:28<00:08, 10114.42it/s] 80%|███████▉  | 318780/400000 [00:28<00:07, 10586.21it/s] 80%|███████▉  | 319877/400000 [00:28<00:07, 10697.27it/s] 80%|████████  | 321015/400000 [00:28<00:07, 10891.98it/s] 81%|████████  | 322202/400000 [00:28<00:06, 11167.73it/s] 81%|████████  | 323422/400000 [00:28<00:06, 11458.26it/s] 81%|████████  | 324637/400000 [00:28<00:06, 11654.92it/s] 81%|████████▏ | 325810/400000 [00:28<00:06, 11677.31it/s] 82%|████████▏ | 327004/400000 [00:28<00:06, 11753.42it/s] 82%|████████▏ | 328221/400000 [00:28<00:06, 11873.24it/s] 82%|████████▏ | 329411/400000 [00:29<00:05, 11845.80it/s] 83%|████████▎ | 330598/400000 [00:29<00:06, 11465.76it/s] 83%|████████▎ | 331749/400000 [00:29<00:06, 11345.64it/s] 83%|████████▎ | 332954/400000 [00:29<00:05, 11546.86it/s] 84%|████████▎ | 334121/400000 [00:29<00:05, 11581.11it/s] 84%|████████▍ | 335339/400000 [00:29<00:05, 11752.79it/s] 84%|████████▍ | 336517/400000 [00:29<00:05, 11748.03it/s] 84%|████████▍ | 337694/400000 [00:29<00:05, 11753.57it/s] 85%|████████▍ | 338899/400000 [00:29<00:05, 11839.38it/s] 85%|████████▌ | 340104/400000 [00:29<00:05, 11899.02it/s] 85%|████████▌ | 341295/400000 [00:30<00:04, 11866.98it/s] 86%|████████▌ | 342483/400000 [00:30<00:04, 11640.46it/s] 86%|████████▌ | 343656/400000 [00:30<00:04, 11664.29it/s] 86%|████████▌ | 344824/400000 [00:30<00:04, 11646.08it/s] 86%|████████▋ | 345990/400000 [00:30<00:04, 11301.15it/s] 87%|████████▋ | 347123/400000 [00:30<00:04, 11182.22it/s] 87%|████████▋ | 348288/400000 [00:30<00:04, 11316.14it/s] 87%|████████▋ | 349422/400000 [00:30<00:04, 11037.78it/s] 88%|████████▊ | 350529/400000 [00:30<00:04, 10981.76it/s] 88%|████████▊ | 351720/400000 [00:31<00:04, 11242.64it/s] 88%|████████▊ | 352911/400000 [00:31<00:04, 11434.42it/s] 89%|████████▊ | 354107/400000 [00:31<00:03, 11586.41it/s] 89%|████████▉ | 355287/400000 [00:31<00:03, 11647.97it/s] 89%|████████▉ | 356454/400000 [00:31<00:03, 11574.70it/s] 89%|████████▉ | 357621/400000 [00:31<00:03, 11601.54it/s] 90%|████████▉ | 358828/400000 [00:31<00:03, 11736.93it/s] 90%|█████████ | 360028/400000 [00:31<00:03, 11812.65it/s] 90%|█████████ | 361211/400000 [00:31<00:03, 11634.73it/s] 91%|█████████ | 362376/400000 [00:31<00:03, 11541.80it/s] 91%|█████████ | 363567/400000 [00:32<00:03, 11649.77it/s] 91%|█████████ | 364733/400000 [00:32<00:03, 11617.81it/s] 91%|█████████▏| 365934/400000 [00:32<00:02, 11729.25it/s] 92%|█████████▏| 367110/400000 [00:32<00:02, 11736.17it/s] 92%|█████████▏| 368285/400000 [00:32<00:02, 11537.93it/s] 92%|█████████▏| 369440/400000 [00:32<00:02, 10349.65it/s] 93%|█████████▎| 370499/400000 [00:32<00:02, 10383.55it/s] 93%|█████████▎| 371667/400000 [00:32<00:02, 10739.16it/s] 93%|█████████▎| 372788/400000 [00:32<00:02, 10875.49it/s] 93%|█████████▎| 373998/400000 [00:32<00:02, 11215.58it/s] 94%|█████████▍| 375221/400000 [00:33<00:02, 11500.70it/s] 94%|█████████▍| 376413/400000 [00:33<00:02, 11622.84it/s] 94%|█████████▍| 377582/400000 [00:33<00:01, 11484.74it/s] 95%|█████████▍| 378736/400000 [00:33<00:01, 11266.71it/s] 95%|█████████▍| 379868/400000 [00:33<00:01, 11217.39it/s] 95%|█████████▌| 380993/400000 [00:33<00:01, 11222.22it/s] 96%|█████████▌| 382118/400000 [00:33<00:01, 11187.44it/s] 96%|█████████▌| 383295/400000 [00:33<00:01, 11353.73it/s] 96%|█████████▌| 384433/400000 [00:33<00:01, 11201.67it/s] 96%|█████████▋| 385633/400000 [00:33<00:01, 11426.38it/s] 97%|█████████▋| 386844/400000 [00:34<00:01, 11621.47it/s] 97%|█████████▋| 388009/400000 [00:34<00:01, 11490.00it/s] 97%|█████████▋| 389161/400000 [00:34<00:00, 11384.86it/s] 98%|█████████▊| 390302/400000 [00:34<00:00, 11273.46it/s] 98%|█████████▊| 391476/400000 [00:34<00:00, 11409.12it/s] 98%|█████████▊| 392679/400000 [00:34<00:00, 11587.30it/s] 98%|█████████▊| 393907/400000 [00:34<00:00, 11786.02it/s] 99%|█████████▉| 395124/400000 [00:34<00:00, 11897.47it/s] 99%|█████████▉| 396316/400000 [00:34<00:00, 11665.27it/s] 99%|█████████▉| 397485/400000 [00:35<00:00, 10800.81it/s]100%|█████████▉| 398580/400000 [00:35<00:00, 10158.38it/s]100%|█████████▉| 399614/400000 [00:35<00:00, 9781.65it/s] 100%|█████████▉| 399999/400000 [00:35<00:00, 11324.06it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f19ae6db4e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011512399657771993 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011303154122470613 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15869 out of table with 15822 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15869 out of table with 15822 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

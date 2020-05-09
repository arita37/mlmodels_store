
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fec42c18f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 17:12:05.868112
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 17:12:05.871909
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 17:12:05.875126
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 17:12:05.878527
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fec4e9dd400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357936.5312
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 286230.6562
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 191289.2969
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 114118.1250
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 65658.6328
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 38662.4062
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 23941.3809
Epoch 8/10

1/1 [==============================] - 0s 88ms/step - loss: 15748.6904
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 10964.6162
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 8002.5752

  #### Inference Need return ypred, ytrue ######################### 
[[-5.76938987e-02 -2.72636712e-01 -3.07743728e-01 -8.77719998e-01
   1.65536547e+00  6.82593286e-02  6.65500641e-01 -1.65632379e+00
  -5.83916605e-01  9.92402613e-01  4.17613596e-01 -2.14875787e-01
   1.73515916e-01 -1.60773921e+00  9.82077956e-01  1.62509680e-01
   1.35375834e+00  2.14533329e-01 -1.07821667e+00 -9.95897770e-01
   9.84495223e-01  2.67174423e-01  4.67243582e-01  3.79608214e-01
  -9.32170510e-01 -5.34443498e-01  6.63944662e-01 -7.75240123e-01
   1.11262870e+00 -4.71594393e-01  1.16162789e+00 -5.96262813e-02
  -2.56034553e-01  6.49768114e-02  1.15908301e+00  7.79172659e-01
   3.88713151e-01  2.65588164e-01  3.94475907e-02 -6.19876444e-01
   3.37066770e-01  1.89758337e+00 -3.40323001e-01 -2.19321823e+00
  -9.17613506e-04 -9.31772232e-01 -6.11913562e-01 -1.47520685e+00
  -4.55008745e-02 -1.22683048e+00  5.83507419e-01 -1.50513160e+00
  -2.40359828e-01  1.64727271e-01 -4.92896736e-01  1.79957294e+00
  -8.48630428e-01 -3.68202746e-01 -1.67514968e+00  7.36712754e-01
  -2.26867601e-01  7.13253784e+00  7.68167925e+00  7.47314119e+00
   7.49256325e+00  7.93493080e+00  7.83604240e+00  7.75097752e+00
   6.72972441e+00  7.13015556e+00  8.25880527e+00  6.81122732e+00
   7.37364531e+00  8.51663876e+00  6.67386675e+00  7.57259178e+00
   6.73810005e+00  5.67502069e+00  7.86643410e+00  6.83398342e+00
   7.88888311e+00  8.24184608e+00  7.75010967e+00  8.39975643e+00
   6.48136044e+00  7.00163794e+00  6.15799809e+00  7.75596189e+00
   7.03624725e+00  7.35480976e+00  7.58539152e+00  7.53599691e+00
   7.34507418e+00  7.94099522e+00  7.13340187e+00  7.85779238e+00
   5.66236019e+00  6.61913300e+00  8.23365021e+00  7.17747402e+00
   7.31627750e+00  6.79282999e+00  6.88873291e+00  8.64415646e+00
   7.28418636e+00  8.17760658e+00  7.51712132e+00  6.57546520e+00
   5.05670404e+00  8.11093330e+00  6.99552107e+00  5.74510288e+00
   5.93663216e+00  7.38171005e+00  7.19821739e+00  8.17679596e+00
   8.09955311e+00  8.99342728e+00  6.41461945e+00  7.01886654e+00
  -8.82668078e-01 -1.05386019e+00  8.47852588e-01 -2.24151760e-01
  -8.68576765e-01  3.97730768e-02  1.15780902e+00 -7.37783730e-01
   1.03113663e+00 -7.97384381e-01  6.23871565e-01  6.84796095e-01
   1.53738618e+00  1.12922513e+00 -7.70698607e-01  1.07118416e+00
   4.00246143e-01  5.81778109e-01  1.15187490e+00 -8.16206992e-01
  -2.03600571e-01 -1.50921488e+00 -7.47959793e-01 -8.28618526e-01
   1.53311121e+00  6.57494307e-01 -9.42248821e-01 -7.79315233e-02
  -7.26638138e-01  2.97967464e-01  1.34298325e-01  1.61222696e+00
  -1.30986321e+00  6.96831524e-01  1.05336726e-01  6.32789731e-02
   7.29249001e-01  8.86523500e-02  6.06338978e-02  3.65890563e-01
  -9.61867809e-01 -1.34598541e+00 -5.56778431e-01 -2.05014646e-01
  -8.37535262e-01 -8.69344294e-01  9.95635986e-01  1.38388610e+00
   1.19340658e-01  1.06219649e+00 -1.19659126e+00 -7.75429249e-01
   1.58466911e+00  4.70327228e-01 -1.07348454e+00 -9.13004339e-01
  -1.05534725e-01  1.01124561e+00 -4.25900757e-01  6.79804087e-02
   1.55837286e+00  9.23463345e-01  1.21623623e+00  1.61392927e+00
   3.75710726e-01  4.47448075e-01  1.77129364e+00  1.36627924e+00
   1.87015271e+00  4.55101550e-01  9.83277142e-01  9.32651699e-01
   4.09507394e-01  7.28380322e-01  4.94082212e-01  2.24974573e-01
   1.80146670e+00  7.57189393e-01  1.84823418e+00  7.94239342e-01
   2.13347721e+00  1.28216696e+00  1.55794847e+00  1.67699480e+00
   6.64462566e-01  7.10991621e-01  1.13129139e-01  1.98962891e+00
   1.04674256e+00  8.89735699e-01  4.16934133e-01  1.35877037e+00
   1.36938000e+00  3.03164482e-01  9.61842835e-01  1.65280843e+00
   5.02690494e-01  1.16400278e+00  8.97769213e-01  2.74071360e+00
   1.40793324e+00  1.31347454e+00  8.13036382e-01  2.39302969e+00
   2.37744689e-01  1.16932833e+00  9.98823404e-01  2.01104283e-01
   1.63673472e+00  8.36787283e-01  1.14967656e+00  1.90233564e+00
   1.38274980e+00  5.23045897e-01  2.80482590e-01  4.28106070e-01
   1.17799354e+00  2.21213222e+00  1.27381349e+00  1.90841711e+00
   4.49648499e-02  7.73340321e+00  7.81907034e+00  7.86504745e+00
   7.43596268e+00  8.37507629e+00  8.05081081e+00  8.19149208e+00
   7.06580067e+00  7.98731613e+00  6.75456238e+00  8.57798862e+00
   6.83373833e+00  8.59787273e+00  6.83566427e+00  7.90276766e+00
   8.86942768e+00  8.73336601e+00  7.47255325e+00  8.54014206e+00
   7.83326912e+00  7.88415194e+00  8.87150478e+00  7.89785862e+00
   8.67758846e+00  6.15331936e+00  7.07779980e+00  7.82127380e+00
   6.81412840e+00  7.70189428e+00  9.22627926e+00  7.91034937e+00
   6.33566141e+00  7.62372351e+00  7.98221064e+00  7.65858698e+00
   7.64872074e+00  8.33609200e+00  9.03074169e+00  7.72477245e+00
   6.91606140e+00  8.71663857e+00  8.94541645e+00  6.61303806e+00
   7.61083460e+00  6.82984447e+00  6.85546446e+00  8.34781170e+00
   7.11887503e+00  8.60118198e+00  5.56963491e+00  7.58878899e+00
   8.23802662e+00  8.17411900e+00  7.78956127e+00  8.02994442e+00
   7.99502802e+00  7.36792183e+00  8.75620842e+00  7.18384027e+00
   1.33728158e+00  1.75059795e-01  9.57152247e-01  1.20586336e+00
   2.17698669e+00  4.48313832e-01  3.47031474e-01  1.06797147e+00
   1.70893025e+00  1.55980170e+00  1.29069209e-01  1.10743332e+00
   1.88458323e-01  7.73737371e-01  7.62340009e-01  5.31910360e-01
   3.05327415e+00  1.68831038e+00  9.14787650e-01  4.83188272e-01
   8.36941898e-01  1.73155642e+00  8.40471148e-01  3.79123688e-01
   9.54200566e-01  1.20771408e+00  2.02414274e-01  1.00000691e+00
   3.12362194e+00  5.83811939e-01  1.84865260e+00  6.44186854e-01
   6.77594543e-01  4.19227600e-01  2.28751707e+00  2.35647798e-01
   2.04189777e+00  1.18125415e+00  6.80221617e-01  1.07337165e+00
   1.93086934e+00  9.53484118e-01  1.58404398e+00  1.01154709e+00
   1.93866146e+00  1.52089953e-01  1.65041709e+00  2.19635296e+00
   4.73926008e-01  2.51605868e-01  6.74626231e-01  3.41787577e-01
   1.35894728e+00  9.60979223e-01  9.33375716e-01  2.22687650e+00
   1.21050060e+00  1.82715464e+00  3.60902905e-01  4.81539607e-01
  -1.01768160e+01  6.90780401e+00 -6.13345814e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 17:12:16.514582
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6454
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 17:12:16.518224
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    8983.2
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 17:12:16.521370
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6162
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 17:12:16.524407
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -803.509
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140652349769040
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140651408315168
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140651408315672
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140651408316176
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140651408316680
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140651408317184

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fec448b6f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.561734
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.523651
grad_step = 000002, loss = 0.497582
grad_step = 000003, loss = 0.471679
grad_step = 000004, loss = 0.444023
grad_step = 000005, loss = 0.417898
grad_step = 000006, loss = 0.400926
grad_step = 000007, loss = 0.397321
grad_step = 000008, loss = 0.385975
grad_step = 000009, loss = 0.366095
grad_step = 000010, loss = 0.348238
grad_step = 000011, loss = 0.334673
grad_step = 000012, loss = 0.323336
grad_step = 000013, loss = 0.312269
grad_step = 000014, loss = 0.300746
grad_step = 000015, loss = 0.288673
grad_step = 000016, loss = 0.276469
grad_step = 000017, loss = 0.264711
grad_step = 000018, loss = 0.253923
grad_step = 000019, loss = 0.244888
grad_step = 000020, loss = 0.236860
grad_step = 000021, loss = 0.227705
grad_step = 000022, loss = 0.216351
grad_step = 000023, loss = 0.205175
grad_step = 000024, loss = 0.195797
grad_step = 000025, loss = 0.187879
grad_step = 000026, loss = 0.180303
grad_step = 000027, loss = 0.172386
grad_step = 000028, loss = 0.164064
grad_step = 000029, loss = 0.156076
grad_step = 000030, loss = 0.149018
grad_step = 000031, loss = 0.142536
grad_step = 000032, loss = 0.135888
grad_step = 000033, loss = 0.129035
grad_step = 000034, loss = 0.122720
grad_step = 000035, loss = 0.117021
grad_step = 000036, loss = 0.111316
grad_step = 000037, loss = 0.105536
grad_step = 000038, loss = 0.099902
grad_step = 000039, loss = 0.094583
grad_step = 000040, loss = 0.089547
grad_step = 000041, loss = 0.084707
grad_step = 000042, loss = 0.080037
grad_step = 000043, loss = 0.075510
grad_step = 000044, loss = 0.071174
grad_step = 000045, loss = 0.067048
grad_step = 000046, loss = 0.063122
grad_step = 000047, loss = 0.059297
grad_step = 000048, loss = 0.055627
grad_step = 000049, loss = 0.052123
grad_step = 000050, loss = 0.048742
grad_step = 000051, loss = 0.045518
grad_step = 000052, loss = 0.042491
grad_step = 000053, loss = 0.039533
grad_step = 000054, loss = 0.036695
grad_step = 000055, loss = 0.034067
grad_step = 000056, loss = 0.031576
grad_step = 000057, loss = 0.029142
grad_step = 000058, loss = 0.026889
grad_step = 000059, loss = 0.024835
grad_step = 000060, loss = 0.022845
grad_step = 000061, loss = 0.020951
grad_step = 000062, loss = 0.019236
grad_step = 000063, loss = 0.017647
grad_step = 000064, loss = 0.016137
grad_step = 000065, loss = 0.014780
grad_step = 000066, loss = 0.013552
grad_step = 000067, loss = 0.012364
grad_step = 000068, loss = 0.011218
grad_step = 000069, loss = 0.010225
grad_step = 000070, loss = 0.009349
grad_step = 000071, loss = 0.008513
grad_step = 000072, loss = 0.007715
grad_step = 000073, loss = 0.007010
grad_step = 000074, loss = 0.006418
grad_step = 000075, loss = 0.005855
grad_step = 000076, loss = 0.005322
grad_step = 000077, loss = 0.004884
grad_step = 000078, loss = 0.004515
grad_step = 000079, loss = 0.004162
grad_step = 000080, loss = 0.003836
grad_step = 000081, loss = 0.003584
grad_step = 000082, loss = 0.003380
grad_step = 000083, loss = 0.003188
grad_step = 000084, loss = 0.003011
grad_step = 000085, loss = 0.002871
grad_step = 000086, loss = 0.002766
grad_step = 000087, loss = 0.002673
grad_step = 000088, loss = 0.002579
grad_step = 000089, loss = 0.002501
grad_step = 000090, loss = 0.002447
grad_step = 000091, loss = 0.002406
grad_step = 000092, loss = 0.002369
grad_step = 000093, loss = 0.002330
grad_step = 000094, loss = 0.002297
grad_step = 000095, loss = 0.002273
grad_step = 000096, loss = 0.002256
grad_step = 000097, loss = 0.002243
grad_step = 000098, loss = 0.002232
grad_step = 000099, loss = 0.002218
grad_step = 000100, loss = 0.002202
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002187
grad_step = 000102, loss = 0.002172
grad_step = 000103, loss = 0.002162
grad_step = 000104, loss = 0.002152
grad_step = 000105, loss = 0.002142
grad_step = 000106, loss = 0.002133
grad_step = 000107, loss = 0.002123
grad_step = 000108, loss = 0.002112
grad_step = 000109, loss = 0.002104
grad_step = 000110, loss = 0.002097
grad_step = 000111, loss = 0.002094
grad_step = 000112, loss = 0.002097
grad_step = 000113, loss = 0.002103
grad_step = 000114, loss = 0.002117
grad_step = 000115, loss = 0.002125
grad_step = 000116, loss = 0.002125
grad_step = 000117, loss = 0.002104
grad_step = 000118, loss = 0.002070
grad_step = 000119, loss = 0.002040
grad_step = 000120, loss = 0.002023
grad_step = 000121, loss = 0.002022
grad_step = 000122, loss = 0.002033
grad_step = 000123, loss = 0.002044
grad_step = 000124, loss = 0.002044
grad_step = 000125, loss = 0.002027
grad_step = 000126, loss = 0.002002
grad_step = 000127, loss = 0.001983
grad_step = 000128, loss = 0.001978
grad_step = 000129, loss = 0.001984
grad_step = 000130, loss = 0.001992
grad_step = 000131, loss = 0.001996
grad_step = 000132, loss = 0.001995
grad_step = 000133, loss = 0.001994
grad_step = 000134, loss = 0.001997
grad_step = 000135, loss = 0.001997
grad_step = 000136, loss = 0.001995
grad_step = 000137, loss = 0.001983
grad_step = 000138, loss = 0.001967
grad_step = 000139, loss = 0.001950
grad_step = 000140, loss = 0.001938
grad_step = 000141, loss = 0.001932
grad_step = 000142, loss = 0.001929
grad_step = 000143, loss = 0.001927
grad_step = 000144, loss = 0.001926
grad_step = 000145, loss = 0.001925
grad_step = 000146, loss = 0.001926
grad_step = 000147, loss = 0.001934
grad_step = 000148, loss = 0.001953
grad_step = 000149, loss = 0.001997
grad_step = 000150, loss = 0.002067
grad_step = 000151, loss = 0.002172
grad_step = 000152, loss = 0.002196
grad_step = 000153, loss = 0.002121
grad_step = 000154, loss = 0.001962
grad_step = 000155, loss = 0.001911
grad_step = 000156, loss = 0.001991
grad_step = 000157, loss = 0.002045
grad_step = 000158, loss = 0.001990
grad_step = 000159, loss = 0.001895
grad_step = 000160, loss = 0.001902
grad_step = 000161, loss = 0.001975
grad_step = 000162, loss = 0.001980
grad_step = 000163, loss = 0.001911
grad_step = 000164, loss = 0.001867
grad_step = 000165, loss = 0.001896
grad_step = 000166, loss = 0.001932
grad_step = 000167, loss = 0.001910
grad_step = 000168, loss = 0.001867
grad_step = 000169, loss = 0.001862
grad_step = 000170, loss = 0.001890
grad_step = 000171, loss = 0.001903
grad_step = 000172, loss = 0.001879
grad_step = 000173, loss = 0.001851
grad_step = 000174, loss = 0.001848
grad_step = 000175, loss = 0.001863
grad_step = 000176, loss = 0.001869
grad_step = 000177, loss = 0.001855
grad_step = 000178, loss = 0.001835
grad_step = 000179, loss = 0.001827
grad_step = 000180, loss = 0.001832
grad_step = 000181, loss = 0.001840
grad_step = 000182, loss = 0.001841
grad_step = 000183, loss = 0.001833
grad_step = 000184, loss = 0.001822
grad_step = 000185, loss = 0.001815
grad_step = 000186, loss = 0.001817
grad_step = 000187, loss = 0.001829
grad_step = 000188, loss = 0.001853
grad_step = 000189, loss = 0.001890
grad_step = 000190, loss = 0.001952
grad_step = 000191, loss = 0.001978
grad_step = 000192, loss = 0.001962
grad_step = 000193, loss = 0.001857
grad_step = 000194, loss = 0.001812
grad_step = 000195, loss = 0.001855
grad_step = 000196, loss = 0.001886
grad_step = 000197, loss = 0.001850
grad_step = 000198, loss = 0.001797
grad_step = 000199, loss = 0.001815
grad_step = 000200, loss = 0.001852
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001828
grad_step = 000202, loss = 0.001804
grad_step = 000203, loss = 0.001836
grad_step = 000204, loss = 0.001889
grad_step = 000205, loss = 0.001915
grad_step = 000206, loss = 0.001972
grad_step = 000207, loss = 0.002037
grad_step = 000208, loss = 0.002066
grad_step = 000209, loss = 0.001954
grad_step = 000210, loss = 0.001819
grad_step = 000211, loss = 0.001762
grad_step = 000212, loss = 0.001800
grad_step = 000213, loss = 0.001869
grad_step = 000214, loss = 0.001896
grad_step = 000215, loss = 0.001869
grad_step = 000216, loss = 0.001788
grad_step = 000217, loss = 0.001738
grad_step = 000218, loss = 0.001756
grad_step = 000219, loss = 0.001805
grad_step = 000220, loss = 0.001828
grad_step = 000221, loss = 0.001800
grad_step = 000222, loss = 0.001758
grad_step = 000223, loss = 0.001731
grad_step = 000224, loss = 0.001732
grad_step = 000225, loss = 0.001752
grad_step = 000226, loss = 0.001770
grad_step = 000227, loss = 0.001769
grad_step = 000228, loss = 0.001746
grad_step = 000229, loss = 0.001721
grad_step = 000230, loss = 0.001710
grad_step = 000231, loss = 0.001714
grad_step = 000232, loss = 0.001725
grad_step = 000233, loss = 0.001733
grad_step = 000234, loss = 0.001734
grad_step = 000235, loss = 0.001728
grad_step = 000236, loss = 0.001717
grad_step = 000237, loss = 0.001703
grad_step = 000238, loss = 0.001694
grad_step = 000239, loss = 0.001689
grad_step = 000240, loss = 0.001688
grad_step = 000241, loss = 0.001689
grad_step = 000242, loss = 0.001693
grad_step = 000243, loss = 0.001702
grad_step = 000244, loss = 0.001721
grad_step = 000245, loss = 0.001756
grad_step = 000246, loss = 0.001806
grad_step = 000247, loss = 0.001883
grad_step = 000248, loss = 0.001939
grad_step = 000249, loss = 0.001974
grad_step = 000250, loss = 0.001892
grad_step = 000251, loss = 0.001770
grad_step = 000252, loss = 0.001676
grad_step = 000253, loss = 0.001677
grad_step = 000254, loss = 0.001744
grad_step = 000255, loss = 0.001793
grad_step = 000256, loss = 0.001782
grad_step = 000257, loss = 0.001718
grad_step = 000258, loss = 0.001667
grad_step = 000259, loss = 0.001658
grad_step = 000260, loss = 0.001683
grad_step = 000261, loss = 0.001714
grad_step = 000262, loss = 0.001720
grad_step = 000263, loss = 0.001704
grad_step = 000264, loss = 0.001676
grad_step = 000265, loss = 0.001654
grad_step = 000266, loss = 0.001645
grad_step = 000267, loss = 0.001648
grad_step = 000268, loss = 0.001655
grad_step = 000269, loss = 0.001658
grad_step = 000270, loss = 0.001658
grad_step = 000271, loss = 0.001653
grad_step = 000272, loss = 0.001646
grad_step = 000273, loss = 0.001637
grad_step = 000274, loss = 0.001629
grad_step = 000275, loss = 0.001620
grad_step = 000276, loss = 0.001612
grad_step = 000277, loss = 0.001608
grad_step = 000278, loss = 0.001608
grad_step = 000279, loss = 0.001612
grad_step = 000280, loss = 0.001619
grad_step = 000281, loss = 0.001627
grad_step = 000282, loss = 0.001635
grad_step = 000283, loss = 0.001644
grad_step = 000284, loss = 0.001652
grad_step = 000285, loss = 0.001669
grad_step = 000286, loss = 0.001691
grad_step = 000287, loss = 0.001728
grad_step = 000288, loss = 0.001761
grad_step = 000289, loss = 0.001798
grad_step = 000290, loss = 0.001791
grad_step = 000291, loss = 0.001753
grad_step = 000292, loss = 0.001670
grad_step = 000293, loss = 0.001594
grad_step = 000294, loss = 0.001560
grad_step = 000295, loss = 0.001575
grad_step = 000296, loss = 0.001612
grad_step = 000297, loss = 0.001637
grad_step = 000298, loss = 0.001641
grad_step = 000299, loss = 0.001620
grad_step = 000300, loss = 0.001596
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001576
grad_step = 000302, loss = 0.001571
grad_step = 000303, loss = 0.001573
grad_step = 000304, loss = 0.001572
grad_step = 000305, loss = 0.001555
grad_step = 000306, loss = 0.001535
grad_step = 000307, loss = 0.001525
grad_step = 000308, loss = 0.001531
grad_step = 000309, loss = 0.001545
grad_step = 000310, loss = 0.001551
grad_step = 000311, loss = 0.001547
grad_step = 000312, loss = 0.001533
grad_step = 000313, loss = 0.001525
grad_step = 000314, loss = 0.001528
grad_step = 000315, loss = 0.001542
grad_step = 000316, loss = 0.001554
grad_step = 000317, loss = 0.001566
grad_step = 000318, loss = 0.001569
grad_step = 000319, loss = 0.001580
grad_step = 000320, loss = 0.001589
grad_step = 000321, loss = 0.001606
grad_step = 000322, loss = 0.001615
grad_step = 000323, loss = 0.001619
grad_step = 000324, loss = 0.001597
grad_step = 000325, loss = 0.001562
grad_step = 000326, loss = 0.001511
grad_step = 000327, loss = 0.001473
grad_step = 000328, loss = 0.001453
grad_step = 000329, loss = 0.001448
grad_step = 000330, loss = 0.001450
grad_step = 000331, loss = 0.001455
grad_step = 000332, loss = 0.001469
grad_step = 000333, loss = 0.001489
grad_step = 000334, loss = 0.001518
grad_step = 000335, loss = 0.001542
grad_step = 000336, loss = 0.001570
grad_step = 000337, loss = 0.001574
grad_step = 000338, loss = 0.001579
grad_step = 000339, loss = 0.001554
grad_step = 000340, loss = 0.001525
grad_step = 000341, loss = 0.001480
grad_step = 000342, loss = 0.001441
grad_step = 000343, loss = 0.001414
grad_step = 000344, loss = 0.001410
grad_step = 000345, loss = 0.001424
grad_step = 000346, loss = 0.001444
grad_step = 000347, loss = 0.001460
grad_step = 000348, loss = 0.001462
grad_step = 000349, loss = 0.001458
grad_step = 000350, loss = 0.001441
grad_step = 000351, loss = 0.001426
grad_step = 000352, loss = 0.001410
grad_step = 000353, loss = 0.001401
grad_step = 000354, loss = 0.001393
grad_step = 000355, loss = 0.001387
grad_step = 000356, loss = 0.001382
grad_step = 000357, loss = 0.001379
grad_step = 000358, loss = 0.001379
grad_step = 000359, loss = 0.001382
grad_step = 000360, loss = 0.001387
grad_step = 000361, loss = 0.001393
grad_step = 000362, loss = 0.001401
grad_step = 000363, loss = 0.001413
grad_step = 000364, loss = 0.001439
grad_step = 000365, loss = 0.001476
grad_step = 000366, loss = 0.001545
grad_step = 000367, loss = 0.001605
grad_step = 000368, loss = 0.001694
grad_step = 000369, loss = 0.001680
grad_step = 000370, loss = 0.001631
grad_step = 000371, loss = 0.001515
grad_step = 000372, loss = 0.001422
grad_step = 000373, loss = 0.001385
grad_step = 000374, loss = 0.001393
grad_step = 000375, loss = 0.001432
grad_step = 000376, loss = 0.001468
grad_step = 000377, loss = 0.001470
grad_step = 000378, loss = 0.001429
grad_step = 000379, loss = 0.001365
grad_step = 000380, loss = 0.001336
grad_step = 000381, loss = 0.001351
grad_step = 000382, loss = 0.001378
grad_step = 000383, loss = 0.001387
grad_step = 000384, loss = 0.001376
grad_step = 000385, loss = 0.001373
grad_step = 000386, loss = 0.001390
grad_step = 000387, loss = 0.001401
grad_step = 000388, loss = 0.001405
grad_step = 000389, loss = 0.001385
grad_step = 000390, loss = 0.001373
grad_step = 000391, loss = 0.001365
grad_step = 000392, loss = 0.001357
grad_step = 000393, loss = 0.001341
grad_step = 000394, loss = 0.001321
grad_step = 000395, loss = 0.001307
grad_step = 000396, loss = 0.001303
grad_step = 000397, loss = 0.001307
grad_step = 000398, loss = 0.001308
grad_step = 000399, loss = 0.001304
grad_step = 000400, loss = 0.001298
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001292
grad_step = 000402, loss = 0.001291
grad_step = 000403, loss = 0.001299
grad_step = 000404, loss = 0.001311
grad_step = 000405, loss = 0.001334
grad_step = 000406, loss = 0.001360
grad_step = 000407, loss = 0.001428
grad_step = 000408, loss = 0.001531
grad_step = 000409, loss = 0.001737
grad_step = 000410, loss = 0.001858
grad_step = 000411, loss = 0.001947
grad_step = 000412, loss = 0.001661
grad_step = 000413, loss = 0.001385
grad_step = 000414, loss = 0.001327
grad_step = 000415, loss = 0.001461
grad_step = 000416, loss = 0.001544
grad_step = 000417, loss = 0.001419
grad_step = 000418, loss = 0.001312
grad_step = 000419, loss = 0.001339
grad_step = 000420, loss = 0.001375
grad_step = 000421, loss = 0.001330
grad_step = 000422, loss = 0.001264
grad_step = 000423, loss = 0.001293
grad_step = 000424, loss = 0.001364
grad_step = 000425, loss = 0.001314
grad_step = 000426, loss = 0.001244
grad_step = 000427, loss = 0.001232
grad_step = 000428, loss = 0.001285
grad_step = 000429, loss = 0.001313
grad_step = 000430, loss = 0.001259
grad_step = 000431, loss = 0.001218
grad_step = 000432, loss = 0.001206
grad_step = 000433, loss = 0.001235
grad_step = 000434, loss = 0.001266
grad_step = 000435, loss = 0.001228
grad_step = 000436, loss = 0.001200
grad_step = 000437, loss = 0.001188
grad_step = 000438, loss = 0.001198
grad_step = 000439, loss = 0.001225
grad_step = 000440, loss = 0.001229
grad_step = 000441, loss = 0.001222
grad_step = 000442, loss = 0.001185
grad_step = 000443, loss = 0.001164
grad_step = 000444, loss = 0.001155
grad_step = 000445, loss = 0.001160
grad_step = 000446, loss = 0.001188
grad_step = 000447, loss = 0.001224
grad_step = 000448, loss = 0.001297
grad_step = 000449, loss = 0.001348
grad_step = 000450, loss = 0.001398
grad_step = 000451, loss = 0.001334
grad_step = 000452, loss = 0.001257
grad_step = 000453, loss = 0.001252
grad_step = 000454, loss = 0.001248
grad_step = 000455, loss = 0.001249
grad_step = 000456, loss = 0.001151
grad_step = 000457, loss = 0.001078
grad_step = 000458, loss = 0.001089
grad_step = 000459, loss = 0.001143
grad_step = 000460, loss = 0.001199
grad_step = 000461, loss = 0.001268
grad_step = 000462, loss = 0.001456
grad_step = 000463, loss = 0.001704
grad_step = 000464, loss = 0.001579
grad_step = 000465, loss = 0.001298
grad_step = 000466, loss = 0.001073
grad_step = 000467, loss = 0.001152
grad_step = 000468, loss = 0.001340
grad_step = 000469, loss = 0.001348
grad_step = 000470, loss = 0.001247
grad_step = 000471, loss = 0.001189
grad_step = 000472, loss = 0.001160
grad_step = 000473, loss = 0.001137
grad_step = 000474, loss = 0.001163
grad_step = 000475, loss = 0.001205
grad_step = 000476, loss = 0.001133
grad_step = 000477, loss = 0.000998
grad_step = 000478, loss = 0.001031
grad_step = 000479, loss = 0.001138
grad_step = 000480, loss = 0.001087
grad_step = 000481, loss = 0.001082
grad_step = 000482, loss = 0.001117
grad_step = 000483, loss = 0.000999
grad_step = 000484, loss = 0.000927
grad_step = 000485, loss = 0.000965
grad_step = 000486, loss = 0.000962
grad_step = 000487, loss = 0.000906
grad_step = 000488, loss = 0.000904
grad_step = 000489, loss = 0.000920
grad_step = 000490, loss = 0.000907
grad_step = 000491, loss = 0.000892
grad_step = 000492, loss = 0.000899
grad_step = 000493, loss = 0.000917
grad_step = 000494, loss = 0.000962
grad_step = 000495, loss = 0.001107
grad_step = 000496, loss = 0.001315
grad_step = 000497, loss = 0.001236
grad_step = 000498, loss = 0.001085
grad_step = 000499, loss = 0.000862
grad_step = 000500, loss = 0.000823
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000968
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

  date_run                              2020-05-09 17:12:34.821658
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.199593
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 17:12:34.827273
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.103562
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 17:12:34.835009
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.116567
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 17:12:34.839922
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.573667
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
0   2020-05-09 17:12:05.868112  ...    mean_absolute_error
1   2020-05-09 17:12:05.871909  ...     mean_squared_error
2   2020-05-09 17:12:05.875126  ...  median_absolute_error
3   2020-05-09 17:12:05.878527  ...               r2_score
4   2020-05-09 17:12:16.514582  ...    mean_absolute_error
5   2020-05-09 17:12:16.518224  ...     mean_squared_error
6   2020-05-09 17:12:16.521370  ...  median_absolute_error
7   2020-05-09 17:12:16.524407  ...               r2_score
8   2020-05-09 17:12:34.821658  ...    mean_absolute_error
9   2020-05-09 17:12:34.827273  ...     mean_squared_error
10  2020-05-09 17:12:34.835009  ...  median_absolute_error
11  2020-05-09 17:12:34.839922  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  7%|▋         | 737280/9912422 [00:00<00:01, 7041619.41it/s] 24%|██▍       | 2416640/9912422 [00:00<00:00, 8502796.08it/s] 63%|██████▎   | 6258688/9912422 [00:00<00:00, 11084745.23it/s]9920512it [00:00, 20315839.64it/s]                             
0it [00:00, ?it/s]32768it [00:00, 735246.82it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 152421.70it/s]1654784it [00:00, 11183152.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 219763.08it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb242004780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb1df748ac8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb241fbbe48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb1df748da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb241fbbe48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb242004e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb242004780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb1f49b6cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb1df748ac8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb1f49b6cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb241fbbe48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f91169a1208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=000d670e457402fab5b0c5eefd79e4423eb1b80fee7d616711ce97b8db08ac2c
  Stored in directory: /tmp/pip-ephem-wheel-cache-usipy1lo/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f90ae586048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1245184/17464789 [=>............................] - ETA: 0s
 5079040/17464789 [=======>......................] - ETA: 0s
11960320/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 17:14:00.783077: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 17:14:00.787539: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-09 17:14:00.787673: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e509640dc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 17:14:00.787685: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0653 - accuracy: 0.4740
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5976 - accuracy: 0.5045
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6268 - accuracy: 0.5026
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6379 - accuracy: 0.5019
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6375 - accuracy: 0.5019
11000/25000 [============>.................] - ETA: 3s - loss: 7.6694 - accuracy: 0.4998
12000/25000 [=============>................] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6867 - accuracy: 0.4987
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6955 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6947 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6873 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 7s 278us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 17:14:14.292270
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 17:14:14.292270  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 17:14:20.208743: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 17:14:20.214572: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-09 17:14:20.215201: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56076133a960 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 17:14:20.215533: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f5f25585b38> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1564 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.1031 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5f2cce30b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.1266 - accuracy: 0.4700
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6022 - accuracy: 0.5042
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6841 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6496 - accuracy: 0.5011
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6084 - accuracy: 0.5038
11000/25000 [============>.................] - ETA: 3s - loss: 7.5900 - accuracy: 0.5050
12000/25000 [=============>................] - ETA: 3s - loss: 7.6232 - accuracy: 0.5028
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6242 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6637 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f5ebc249d68> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<47:54:33, 5.00kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<33:46:19, 7.09kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:41:39, 10.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:35:15, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.58M/862M [00:02<11:34:41, 20.6kB/s].vector_cache/glove.6B.zip:   1%|          | 7.76M/862M [00:02<8:04:01, 29.4kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:02<5:37:13, 42.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.4M/862M [00:02<3:54:57, 60.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:02<2:43:45, 85.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:02<1:54:04, 122kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:02<1:19:36, 174kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.8M/862M [00:02<55:30, 249kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.7M/862M [00:03<38:47, 354kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:03<27:04, 505kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:03<18:58, 716kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:03<13:17, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:03<10:42, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:05<09:22, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<08:26, 1.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:05<06:17, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:05<04:33, 2.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<22:12, 602kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<17:14, 775kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<12:24, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:09<11:22, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:09<10:55, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<08:23, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<06:00, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:11<16:34, 798kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<13:01, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:11<09:26, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:13<09:37, 1.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:13<09:26, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<07:09, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:13<05:12, 2.52MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:15<08:58, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<07:38, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<05:40, 2.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:17<07:01, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<07:35, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<05:53, 2.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<04:15, 3.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:42, 1.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<9:27:11, 22.8kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<6:37:24, 32.6kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<4:37:35, 46.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:21<3:19:24, 64.6kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:21<2:23:17, 89.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<1:41:07, 127kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:21<1:10:45, 181kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:23<54:54, 233kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:23<40:07, 319kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<28:26, 450kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:25<22:17, 572kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<17:21, 735kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<12:29, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<08:52, 1.43MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<26:41, 475kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<20:24, 621kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<14:38, 865kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<12:38, 997kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<10:34, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<07:45, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<07:50, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<08:59, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<07:02, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:08, 2.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<07:04, 1.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:39, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:01, 2.48MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<05:54, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<05:49, 2.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<04:26, 2.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<05:28, 2.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<05:31, 2.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:13, 2.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<05:19, 2.31MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:23, 2.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:08, 2.96MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<05:14, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:05, 2.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:12, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:18, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<04:07, 2.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<07:07, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:42, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<04:11, 2.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:52, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:08, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:20, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:02, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<05:51, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<04:26, 2.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:23, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<05:23, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<04:10, 2.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:10, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<05:13, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<03:59, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<02:55, 4.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<48:46, 240kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<35:44, 328kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<25:18, 462kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<17:47, 655kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<2:08:03, 90.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<1:31:11, 128kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<1:04:04, 181kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<46:54, 247kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<34:24, 336kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<24:26, 473kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<19:14, 598kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<15:03, 764kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<10:54, 1.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:48, 1.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:25, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<06:14, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:31, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:50, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<04:28, 2.53MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<03:16, 3.46MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<16:57, 666kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<13:24, 842kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<09:41, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:53, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<1:40:35, 112kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<1:11:55, 156kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<50:38, 221kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<37:26, 298kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<27:43, 402kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<19:42, 565kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<15:52, 698kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<14:09, 783kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:32, 1.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<07:33, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<08:15, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:16, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:23, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<03:54, 2.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<17:36, 622kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<13:50, 791kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<10:02, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<09:05, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<07:51, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:48, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<06:08, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:46, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:24, 2.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:07, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:03, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<03:50, 2.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<02:49, 3.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<17:09, 622kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<13:27, 793kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<09:46, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<08:50, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:38, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:38, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:58, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:38, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:14, 2.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:04, 3.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<30:35, 342kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<22:50, 458kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<16:16, 642kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<13:21, 779kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<10:45, 966kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<07:52, 1.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<07:28, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:36, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:56, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<03:59, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<7:20:50, 23.3kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<5:09:00, 33.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<3:35:37, 47.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<2:35:08, 65.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<1:51:21, 91.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<1:18:28, 130kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<54:55, 185kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<41:41, 243kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<30:32, 331kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<21:38, 467kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<17:01, 591kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<13:16, 757kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<09:35, 1.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<08:36, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<07:23, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:30, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<05:44, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<05:22, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<04:03, 2.44MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<04:43, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<04:39, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<03:35, 2.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<04:22, 2.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<04:23, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<03:22, 2.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<04:13, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<04:15, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<03:15, 2.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<04:08, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<04:12, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<03:14, 2.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<04:06, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<04:10, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<03:12, 2.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<04:04, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<04:09, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<03:11, 2.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<04:02, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<04:07, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:00<03:09, 2.98MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<04:05, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<03:10, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<03:59, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:03, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<03:07, 2.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<03:57, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<04:01, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<03:05, 2.98MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:55, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<04:00, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<03:03, 2.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:14, 4.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<31:44, 286kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<23:27, 387kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<16:41, 543kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<13:21, 675kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<10:22, 869kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<07:29, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:24, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<07:38, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<06:34, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<04:50, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:28, 2.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<28:59, 306kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<21:30, 413kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<15:19, 578kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<12:21, 713kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:51, 894kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<07:10, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<06:41, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:50, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:22, 1.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:48, 1.49MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<04:39, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:23, 2.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:03, 1.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:25, 1.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<04:02, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:54, 2.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<53:00, 161kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<38:15, 223kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<26:58, 316kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<18:53, 448kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<28:04, 302kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<20:47, 407kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:28<14:46, 571kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<10:23, 809kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<4:17:07, 32.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<3:01:01, 46.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<2:06:42, 66.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<1:29:49, 92.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<1:03:56, 130kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<44:56, 185kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<32:52, 251kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<24:08, 342kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<17:08, 481kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<13:30, 607kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<10:25, 786kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:28, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<06:56, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<07:04, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:25, 1.50MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:55, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:56, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:33, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:25, 2.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:55, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:50, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<02:57, 2.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<03:34, 2.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<03:34, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<02:46, 2.85MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<03:26, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:28, 2.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<02:41, 2.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:22, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:25, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<02:36, 2.97MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<01:56, 3.98MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<06:28, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:16, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:53, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:49, 2.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<07:53, 968kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<07:44, 987kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:52, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:11, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<06:04, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:17, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<03:57, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:08, 2.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<5:19:40, 23.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<3:43:59, 33.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<2:36:04, 47.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<1:52:17, 66.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<1:20:44, 92.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<56:52, 131kB/s]   .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<39:48, 186kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<29:40, 248kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<21:46, 338kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<15:24, 476kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<10:49, 675kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<20:22, 358kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<15:06, 483kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<10:43, 678kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<09:00, 803kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<08:16, 873kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<06:14, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:28, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<06:07, 1.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<05:15, 1.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:53, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<04:04, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<03:50, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<02:52, 2.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:04, 3.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<39:33, 177kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<28:28, 246kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<20:04, 348kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<14:08, 493kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<12:52, 540kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<09:57, 698kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<07:08, 969kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<06:18, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<05:21, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:58, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<04:04, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<03:46, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<02:52, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<03:17, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<03:12, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:25, 2.77MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<01:45, 3.78MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<45:32, 147kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<32:46, 204kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<23:04, 288kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<17:19, 381kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<12:53, 512kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<09:10, 716kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<07:46, 841kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<07:13, 904kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<05:30, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:55, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<05:31, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:25<04:45, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:30, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:30, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<3:12:22, 33.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<2:15:26, 47.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<1:34:44, 67.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<1:07:03, 94.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<47:47, 132kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<33:32, 188kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<24:31, 255kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<18:00, 347kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<12:46, 488kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<10:04, 615kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<07:46, 796kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:35, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<05:09, 1.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<05:22, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<04:05, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:05, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:16, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<03:07, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:23, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:48, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<02:47, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:09, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:37, 2.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:00, 2.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:32, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<02:32, 2.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<01:56, 2.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:28, 2.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<02:30, 2.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<01:55, 2.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:24, 4.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<20:56, 272kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<15:25, 369kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<10:55, 520kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<08:40, 649kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<06:50, 823kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<04:55, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:29, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<1:53:14, 49.1kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<1:19:58, 69.5kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<55:58, 99.0kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<39:54, 138kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<28:40, 191kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<20:11, 271kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<15:03, 360kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<11:17, 480kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<08:03, 670kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<06:37, 809kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:21, 998kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<03:53, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:43, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:19, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:30, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:44, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:37, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:00, 2.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:23, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:48, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:18, 3.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<10:28, 485kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<08:31, 595kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:15, 809kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:24, 1.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:59, 716kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<05:34, 897kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:02, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:45, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:18, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:28, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:38, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:11, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:34, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<01:51, 2.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:18, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:58, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:14, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:27, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:21, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:47, 2.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:08, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:06, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:38, 2.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:00, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:02, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:33, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<01:57, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:59, 2.28MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:30, 2.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:06, 4.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<08:22, 532kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<06:28, 687kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:40, 950kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<04:04, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<03:27, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:32, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:59, 2.17MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<3:07:06, 23.1kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<2:10:58, 32.9kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<1:30:54, 47.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<1:05:12, 65.3kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<46:51, 90.8kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<32:58, 129kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<23:00, 183kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<17:09, 244kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<12:34, 333kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<08:53, 468kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<06:56, 593kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<05:25, 758kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:55, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<03:29, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:59, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:17, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:08, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:37, 2.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:52, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:50, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:23, 2.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:42, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:43, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:19, 2.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<01:38, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<01:39, 2.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:17, 2.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<01:36, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<01:37, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:14, 2.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<01:33, 2.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:35, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:13, 2.96MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<01:31, 2.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:33, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:11, 2.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<00:51, 4.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<3:29:10, 16.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<2:26:41, 23.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<1:42:12, 33.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<1:11:17, 48.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<50:48, 67.4kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<35:40, 95.7kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<24:48, 136kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<18:12, 184kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<13:10, 254kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<09:16, 359kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<07:02, 466kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<05:22, 611kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:50, 851kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:41, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<29:17, 110kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<20:55, 154kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<14:44, 217kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<10:17, 308kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<08:05, 389kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<06:05, 516kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<04:19, 723kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:01, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<14:22, 214kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<10:27, 294kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<07:22, 414kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<05:39, 531kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<04:22, 688kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<03:08, 951kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:43, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<02:18, 1.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:41, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:43, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<01:35, 1.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:12, 2.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:22, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:20, 2.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:01, 2.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:13, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:14, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<00:56, 2.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:40, 3.92MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:39, 729kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<02:54, 912kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<02:06, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:29, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<04:08, 628kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:11, 811kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<02:17, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:06, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:10, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<01:39, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:11, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:31, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:23, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:10, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:08, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<00:52, 2.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:02, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:03, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:47, 2.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:34, 3.92MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<04:44, 475kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<03:36, 622kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:34, 866kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:10, 999kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:49, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:19, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:56, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<08:04, 261kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<06:13, 339kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<04:27, 470kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<03:06, 665kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:50, 720kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:15, 901kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:37, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:31<01:08, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<04:51, 406kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<03:51, 510kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:47, 703kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:56, 990kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<02:05, 907kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:43, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:15, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:52, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<03:39, 502kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<02:48, 653kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:59, 908kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:23, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<04:44, 372kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<03:32, 496kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:30, 692kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<02:02, 831kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:37, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:09, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:59, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:43, 2.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:49, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:47, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:35, 2.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:41, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:41, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:31, 2.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:25, 3.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<1:01:39, 23.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<42:57, 32.9kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<29:15, 47.0kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<20:45, 65.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<14:53, 90.7kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<10:26, 128kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<07:07, 183kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<05:27, 235kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<03:59, 321kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<02:47, 452kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<02:06, 574kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:38, 737kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:10, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:00, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:51, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:37, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:37, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:35, 1.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:25, 2.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:28, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:21, 2.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<00:25, 2.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:25, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:18, 2.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:22, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:22, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:17, 2.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:12, 4.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<01:28, 545kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<01:07, 704kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:47, 973kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:40, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:33, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:24, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:23, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:27, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:21, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:14, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:19, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:18, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:13, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:14, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:14, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:10, 2.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:12, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:12, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:08, 2.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:10, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:19<00:07, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:10, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:08, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:05, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:07, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:05, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:04, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:03, 2.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:01, 2.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.31MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.92MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 861/400000 [00:00<00:46, 8601.09it/s]  0%|          | 1746/400000 [00:00<00:45, 8672.57it/s]  1%|          | 2617/400000 [00:00<00:45, 8682.79it/s]  1%|          | 3477/400000 [00:00<00:45, 8657.64it/s]  1%|          | 4361/400000 [00:00<00:45, 8709.46it/s]  1%|▏         | 5242/400000 [00:00<00:45, 8738.99it/s]  2%|▏         | 6155/400000 [00:00<00:44, 8850.21it/s]  2%|▏         | 7079/400000 [00:00<00:43, 8963.06it/s]  2%|▏         | 8007/400000 [00:00<00:43, 9053.76it/s]  2%|▏         | 8940/400000 [00:01<00:42, 9134.48it/s]  2%|▏         | 9834/400000 [00:01<00:43, 9073.43it/s]  3%|▎         | 10731/400000 [00:01<00:43, 9037.97it/s]  3%|▎         | 11624/400000 [00:01<00:43, 8982.59it/s]  3%|▎         | 12528/400000 [00:01<00:43, 8997.16it/s]  3%|▎         | 13423/400000 [00:01<00:43, 8856.17it/s]  4%|▎         | 14306/400000 [00:01<00:43, 8773.56it/s]  4%|▍         | 15182/400000 [00:01<00:43, 8763.86it/s]  4%|▍         | 16057/400000 [00:01<00:43, 8757.84it/s]  4%|▍         | 16932/400000 [00:01<00:43, 8750.32it/s]  4%|▍         | 17807/400000 [00:02<00:43, 8703.23it/s]  5%|▍         | 18677/400000 [00:02<00:43, 8695.11it/s]  5%|▍         | 19558/400000 [00:02<00:43, 8728.59it/s]  5%|▌         | 20431/400000 [00:02<00:43, 8707.71it/s]  5%|▌         | 21309/400000 [00:02<00:43, 8728.10it/s]  6%|▌         | 22184/400000 [00:02<00:43, 8733.89it/s]  6%|▌         | 23058/400000 [00:02<00:43, 8714.56it/s]  6%|▌         | 23930/400000 [00:02<00:43, 8708.13it/s]  6%|▌         | 24801/400000 [00:02<00:43, 8639.48it/s]  6%|▋         | 25671/400000 [00:02<00:43, 8657.54it/s]  7%|▋         | 26537/400000 [00:03<00:43, 8650.84it/s]  7%|▋         | 27416/400000 [00:03<00:42, 8690.78it/s]  7%|▋         | 28298/400000 [00:03<00:42, 8726.57it/s]  7%|▋         | 29171/400000 [00:03<00:42, 8722.55it/s]  8%|▊         | 30044/400000 [00:03<00:42, 8718.28it/s]  8%|▊         | 30927/400000 [00:03<00:42, 8748.73it/s]  8%|▊         | 31802/400000 [00:03<00:42, 8721.90it/s]  8%|▊         | 32675/400000 [00:03<00:42, 8704.14it/s]  8%|▊         | 33546/400000 [00:03<00:42, 8699.63it/s]  9%|▊         | 34416/400000 [00:03<00:42, 8676.28it/s]  9%|▉         | 35284/400000 [00:04<00:42, 8675.65it/s]  9%|▉         | 36152/400000 [00:04<00:42, 8567.46it/s]  9%|▉         | 37010/400000 [00:04<00:42, 8550.94it/s]  9%|▉         | 37878/400000 [00:04<00:42, 8585.86it/s] 10%|▉         | 38752/400000 [00:04<00:41, 8631.15it/s] 10%|▉         | 39616/400000 [00:04<00:41, 8611.69it/s] 10%|█         | 40479/400000 [00:04<00:41, 8616.61it/s] 10%|█         | 41341/400000 [00:04<00:41, 8610.47it/s] 11%|█         | 42211/400000 [00:04<00:41, 8635.73it/s] 11%|█         | 43085/400000 [00:04<00:41, 8663.94it/s] 11%|█         | 43952/400000 [00:05<00:41, 8644.36it/s] 11%|█         | 44825/400000 [00:05<00:40, 8668.83it/s] 11%|█▏        | 45692/400000 [00:05<00:40, 8664.16it/s] 12%|█▏        | 46559/400000 [00:05<00:40, 8621.87it/s] 12%|█▏        | 47423/400000 [00:05<00:40, 8625.00it/s] 12%|█▏        | 48286/400000 [00:05<00:40, 8591.75it/s] 12%|█▏        | 49153/400000 [00:05<00:40, 8612.48it/s] 13%|█▎        | 50018/400000 [00:05<00:40, 8622.40it/s] 13%|█▎        | 50893/400000 [00:05<00:40, 8657.85it/s] 13%|█▎        | 51768/400000 [00:05<00:40, 8685.07it/s] 13%|█▎        | 52640/400000 [00:06<00:39, 8695.49it/s] 13%|█▎        | 53510/400000 [00:06<00:39, 8688.09it/s] 14%|█▎        | 54379/400000 [00:06<00:39, 8652.15it/s] 14%|█▍        | 55255/400000 [00:06<00:39, 8681.97it/s] 14%|█▍        | 56130/400000 [00:06<00:39, 8701.24it/s] 14%|█▍        | 57005/400000 [00:06<00:39, 8713.12it/s] 14%|█▍        | 57880/400000 [00:06<00:39, 8723.30it/s] 15%|█▍        | 58753/400000 [00:06<00:39, 8706.00it/s] 15%|█▍        | 59634/400000 [00:06<00:38, 8734.21it/s] 15%|█▌        | 60540/400000 [00:06<00:38, 8828.75it/s] 15%|█▌        | 61424/400000 [00:07<00:39, 8658.97it/s] 16%|█▌        | 62320/400000 [00:07<00:38, 8745.14it/s] 16%|█▌        | 63216/400000 [00:07<00:38, 8808.46it/s] 16%|█▌        | 64150/400000 [00:07<00:37, 8960.75it/s] 16%|█▋        | 65048/400000 [00:07<00:37, 8944.13it/s] 16%|█▋        | 65944/400000 [00:07<00:37, 8876.15it/s] 17%|█▋        | 66867/400000 [00:07<00:37, 8979.07it/s] 17%|█▋        | 67826/400000 [00:07<00:36, 9152.11it/s] 17%|█▋        | 68767/400000 [00:07<00:35, 9225.29it/s] 17%|█▋        | 69691/400000 [00:07<00:35, 9210.11it/s] 18%|█▊        | 70613/400000 [00:08<00:35, 9180.44it/s] 18%|█▊        | 71543/400000 [00:08<00:35, 9214.87it/s] 18%|█▊        | 72465/400000 [00:08<00:35, 9159.31it/s] 18%|█▊        | 73382/400000 [00:08<00:35, 9151.76it/s] 19%|█▊        | 74328/400000 [00:08<00:35, 9241.59it/s] 19%|█▉        | 75269/400000 [00:08<00:34, 9290.12it/s] 19%|█▉        | 76199/400000 [00:08<00:35, 9127.52it/s] 19%|█▉        | 77124/400000 [00:08<00:35, 9162.86it/s] 20%|█▉        | 78041/400000 [00:08<00:35, 9125.95it/s] 20%|█▉        | 78955/400000 [00:08<00:35, 9070.11it/s] 20%|█▉        | 79863/400000 [00:09<00:35, 9035.59it/s] 20%|██        | 80767/400000 [00:09<00:35, 8994.22it/s] 20%|██        | 81667/400000 [00:09<00:36, 8765.90it/s] 21%|██        | 82548/400000 [00:09<00:36, 8776.23it/s] 21%|██        | 83458/400000 [00:09<00:35, 8870.15it/s] 21%|██        | 84346/400000 [00:09<00:35, 8852.30it/s] 21%|██▏       | 85271/400000 [00:09<00:35, 8965.17it/s] 22%|██▏       | 86173/400000 [00:09<00:34, 8980.29it/s] 22%|██▏       | 87072/400000 [00:09<00:34, 8943.95it/s] 22%|██▏       | 87967/400000 [00:09<00:34, 8922.80it/s] 22%|██▏       | 88860/400000 [00:10<00:35, 8868.08it/s] 22%|██▏       | 89748/400000 [00:10<00:35, 8823.55it/s] 23%|██▎       | 90631/400000 [00:10<00:35, 8605.11it/s] 23%|██▎       | 91512/400000 [00:10<00:35, 8664.38it/s] 23%|██▎       | 92392/400000 [00:10<00:35, 8704.15it/s] 23%|██▎       | 93276/400000 [00:10<00:35, 8742.65it/s] 24%|██▎       | 94159/400000 [00:10<00:34, 8767.58it/s] 24%|██▍       | 95037/400000 [00:10<00:34, 8745.38it/s] 24%|██▍       | 95922/400000 [00:10<00:34, 8776.50it/s] 24%|██▍       | 96809/400000 [00:10<00:34, 8802.50it/s] 24%|██▍       | 97724/400000 [00:11<00:33, 8901.82it/s] 25%|██▍       | 98658/400000 [00:11<00:33, 9028.13it/s] 25%|██▍       | 99562/400000 [00:11<00:33, 8954.06it/s] 25%|██▌       | 100459/400000 [00:11<00:33, 8824.49it/s] 25%|██▌       | 101343/400000 [00:11<00:33, 8824.23it/s] 26%|██▌       | 102227/400000 [00:11<00:33, 8804.74it/s] 26%|██▌       | 103108/400000 [00:11<00:33, 8735.46it/s] 26%|██▌       | 103982/400000 [00:11<00:33, 8715.41it/s] 26%|██▌       | 104866/400000 [00:11<00:33, 8751.71it/s] 26%|██▋       | 105752/400000 [00:12<00:33, 8783.72it/s] 27%|██▋       | 106637/400000 [00:12<00:33, 8802.81it/s] 27%|██▋       | 107518/400000 [00:12<00:33, 8786.51it/s] 27%|██▋       | 108397/400000 [00:12<00:33, 8699.78it/s] 27%|██▋       | 109275/400000 [00:12<00:33, 8722.29it/s] 28%|██▊       | 110160/400000 [00:12<00:33, 8760.17it/s] 28%|██▊       | 111046/400000 [00:12<00:32, 8788.12it/s] 28%|██▊       | 111937/400000 [00:12<00:32, 8822.18it/s] 28%|██▊       | 112820/400000 [00:12<00:33, 8649.43it/s] 28%|██▊       | 113698/400000 [00:12<00:32, 8687.32it/s] 29%|██▊       | 114573/400000 [00:13<00:32, 8705.16it/s] 29%|██▉       | 115460/400000 [00:13<00:32, 8753.69it/s] 29%|██▉       | 116361/400000 [00:13<00:32, 8826.10it/s] 29%|██▉       | 117245/400000 [00:13<00:32, 8822.27it/s] 30%|██▉       | 118141/400000 [00:13<00:31, 8862.13it/s] 30%|██▉       | 119059/400000 [00:13<00:31, 8952.83it/s] 30%|██▉       | 119999/400000 [00:13<00:30, 9080.11it/s] 30%|███       | 120965/400000 [00:13<00:30, 9244.76it/s] 30%|███       | 121891/400000 [00:13<00:30, 9172.45it/s] 31%|███       | 122810/400000 [00:13<00:30, 9098.04it/s] 31%|███       | 123721/400000 [00:14<00:30, 8986.72it/s] 31%|███       | 124621/400000 [00:14<00:31, 8740.33it/s] 31%|███▏      | 125527/400000 [00:14<00:31, 8833.52it/s] 32%|███▏      | 126418/400000 [00:14<00:30, 8855.40it/s] 32%|███▏      | 127314/400000 [00:14<00:30, 8885.80it/s] 32%|███▏      | 128204/400000 [00:14<00:30, 8889.99it/s] 32%|███▏      | 129094/400000 [00:14<00:30, 8834.41it/s] 32%|███▏      | 129978/400000 [00:14<00:30, 8835.15it/s] 33%|███▎      | 130875/400000 [00:14<00:30, 8874.34it/s] 33%|███▎      | 131776/400000 [00:14<00:30, 8913.42it/s] 33%|███▎      | 132675/400000 [00:15<00:29, 8934.15it/s] 33%|███▎      | 133587/400000 [00:15<00:29, 8989.06it/s] 34%|███▎      | 134487/400000 [00:15<00:29, 8885.96it/s] 34%|███▍      | 135379/400000 [00:15<00:29, 8893.92it/s] 34%|███▍      | 136269/400000 [00:15<00:29, 8892.52it/s] 34%|███▍      | 137184/400000 [00:15<00:29, 8965.54it/s] 35%|███▍      | 138081/400000 [00:15<00:29, 8898.44it/s] 35%|███▍      | 139020/400000 [00:15<00:28, 9039.79it/s] 35%|███▍      | 139925/400000 [00:15<00:28, 8991.40it/s] 35%|███▌      | 140825/400000 [00:15<00:29, 8911.36it/s] 35%|███▌      | 141717/400000 [00:16<00:29, 8856.44it/s] 36%|███▌      | 142604/400000 [00:16<00:29, 8832.64it/s] 36%|███▌      | 143489/400000 [00:16<00:29, 8837.60it/s] 36%|███▌      | 144384/400000 [00:16<00:28, 8868.97it/s] 36%|███▋      | 145272/400000 [00:16<00:28, 8834.74it/s] 37%|███▋      | 146156/400000 [00:16<00:28, 8789.78it/s] 37%|███▋      | 147036/400000 [00:16<00:28, 8779.72it/s] 37%|███▋      | 147915/400000 [00:16<00:28, 8772.42it/s] 37%|███▋      | 148793/400000 [00:16<00:28, 8769.75it/s] 37%|███▋      | 149671/400000 [00:16<00:28, 8763.80it/s] 38%|███▊      | 150548/400000 [00:17<00:28, 8755.14it/s] 38%|███▊      | 151424/400000 [00:17<00:28, 8750.85it/s] 38%|███▊      | 152300/400000 [00:17<00:28, 8743.68it/s] 38%|███▊      | 153193/400000 [00:17<00:28, 8798.65it/s] 39%|███▊      | 154073/400000 [00:17<00:28, 8537.92it/s] 39%|███▊      | 154952/400000 [00:17<00:28, 8610.30it/s] 39%|███▉      | 155834/400000 [00:17<00:28, 8671.88it/s] 39%|███▉      | 156716/400000 [00:17<00:27, 8713.12it/s] 39%|███▉      | 157596/400000 [00:17<00:27, 8737.57it/s] 40%|███▉      | 158471/400000 [00:17<00:27, 8740.89it/s] 40%|███▉      | 159346/400000 [00:18<00:27, 8740.54it/s] 40%|████      | 160221/400000 [00:18<00:27, 8598.82it/s] 40%|████      | 161082/400000 [00:18<00:27, 8582.01it/s] 40%|████      | 161954/400000 [00:18<00:27, 8622.66it/s] 41%|████      | 162828/400000 [00:18<00:27, 8656.15it/s] 41%|████      | 163697/400000 [00:18<00:27, 8664.99it/s] 41%|████      | 164579/400000 [00:18<00:27, 8710.86it/s] 41%|████▏     | 165463/400000 [00:18<00:26, 8749.17it/s] 42%|████▏     | 166339/400000 [00:18<00:26, 8720.66it/s] 42%|████▏     | 167215/400000 [00:18<00:26, 8732.08it/s] 42%|████▏     | 168092/400000 [00:19<00:26, 8741.47it/s] 42%|████▏     | 168968/400000 [00:19<00:26, 8745.74it/s] 42%|████▏     | 169852/400000 [00:19<00:26, 8773.36it/s] 43%|████▎     | 170737/400000 [00:19<00:26, 8795.90it/s] 43%|████▎     | 171652/400000 [00:19<00:25, 8897.70it/s] 43%|████▎     | 172558/400000 [00:19<00:25, 8945.48it/s] 43%|████▎     | 173482/400000 [00:19<00:25, 9031.02it/s] 44%|████▎     | 174412/400000 [00:19<00:24, 9107.10it/s] 44%|████▍     | 175324/400000 [00:19<00:24, 9042.46it/s] 44%|████▍     | 176229/400000 [00:19<00:25, 8837.21it/s] 44%|████▍     | 177115/400000 [00:20<00:25, 8805.08it/s] 45%|████▍     | 178010/400000 [00:20<00:25, 8845.51it/s] 45%|████▍     | 178905/400000 [00:20<00:24, 8874.53it/s] 45%|████▍     | 179793/400000 [00:20<00:24, 8869.26it/s] 45%|████▌     | 180684/400000 [00:20<00:24, 8881.38it/s] 45%|████▌     | 181584/400000 [00:20<00:24, 8915.34it/s] 46%|████▌     | 182501/400000 [00:20<00:24, 8989.99it/s] 46%|████▌     | 183401/400000 [00:20<00:24, 8989.39it/s] 46%|████▌     | 184301/400000 [00:20<00:24, 8935.47it/s] 46%|████▋     | 185195/400000 [00:20<00:24, 8886.21it/s] 47%|████▋     | 186084/400000 [00:21<00:24, 8664.55it/s] 47%|████▋     | 186952/400000 [00:21<00:24, 8601.08it/s] 47%|████▋     | 187829/400000 [00:21<00:24, 8649.35it/s] 47%|████▋     | 188738/400000 [00:21<00:24, 8775.98it/s] 47%|████▋     | 189617/400000 [00:21<00:23, 8776.43it/s] 48%|████▊     | 190496/400000 [00:21<00:23, 8762.90it/s] 48%|████▊     | 191394/400000 [00:21<00:23, 8825.33it/s] 48%|████▊     | 192277/400000 [00:21<00:23, 8826.06it/s] 48%|████▊     | 193175/400000 [00:21<00:23, 8871.04it/s] 49%|████▊     | 194125/400000 [00:22<00:22, 9049.69it/s] 49%|████▉     | 195032/400000 [00:22<00:22, 9042.26it/s] 49%|████▉     | 195937/400000 [00:22<00:22, 8972.78it/s] 49%|████▉     | 196844/400000 [00:22<00:22, 9000.78it/s] 49%|████▉     | 197745/400000 [00:22<00:22, 8998.62it/s] 50%|████▉     | 198647/400000 [00:22<00:22, 9004.83it/s] 50%|████▉     | 199548/400000 [00:22<00:22, 8997.52it/s] 50%|█████     | 200471/400000 [00:22<00:22, 9064.54it/s] 50%|█████     | 201378/400000 [00:22<00:22, 8989.82it/s] 51%|█████     | 202278/400000 [00:22<00:22, 8940.36it/s] 51%|█████     | 203177/400000 [00:23<00:21, 8954.90it/s] 51%|█████     | 204086/400000 [00:23<00:21, 8992.94it/s] 51%|█████     | 204988/400000 [00:23<00:21, 8999.24it/s] 51%|█████▏    | 205889/400000 [00:23<00:21, 8969.82it/s] 52%|█████▏    | 206787/400000 [00:23<00:21, 8967.76it/s] 52%|█████▏    | 207695/400000 [00:23<00:21, 8998.16it/s] 52%|█████▏    | 208621/400000 [00:23<00:21, 9073.49it/s] 52%|█████▏    | 209529/400000 [00:23<00:21, 8977.78it/s] 53%|█████▎    | 210428/400000 [00:23<00:21, 8945.29it/s] 53%|█████▎    | 211323/400000 [00:23<00:21, 8942.63it/s] 53%|█████▎    | 212218/400000 [00:24<00:21, 8892.31it/s] 53%|█████▎    | 213108/400000 [00:24<00:21, 8861.79it/s] 53%|█████▎    | 213995/400000 [00:24<00:21, 8836.60it/s] 54%|█████▎    | 214879/400000 [00:24<00:21, 8812.73it/s] 54%|█████▍    | 215761/400000 [00:24<00:20, 8784.98it/s] 54%|█████▍    | 216642/400000 [00:24<00:20, 8792.23it/s] 54%|█████▍    | 217522/400000 [00:24<00:20, 8779.38it/s] 55%|█████▍    | 218400/400000 [00:24<00:20, 8755.45it/s] 55%|█████▍    | 219276/400000 [00:24<00:20, 8754.92it/s] 55%|█████▌    | 220152/400000 [00:24<00:20, 8735.64it/s] 55%|█████▌    | 221036/400000 [00:25<00:20, 8765.74it/s] 55%|█████▌    | 221916/400000 [00:25<00:20, 8775.38it/s] 56%|█████▌    | 222819/400000 [00:25<00:20, 8848.76it/s] 56%|█████▌    | 223737/400000 [00:25<00:19, 8942.81it/s] 56%|█████▌    | 224666/400000 [00:25<00:19, 9044.00it/s] 56%|█████▋    | 225579/400000 [00:25<00:19, 9069.22it/s] 57%|█████▋    | 226487/400000 [00:25<00:19, 9020.69it/s] 57%|█████▋    | 227397/400000 [00:25<00:19, 9044.21it/s] 57%|█████▋    | 228306/400000 [00:25<00:18, 9057.07it/s] 57%|█████▋    | 229233/400000 [00:25<00:18, 9118.74it/s] 58%|█████▊    | 230146/400000 [00:26<00:18, 9015.08it/s] 58%|█████▊    | 231048/400000 [00:26<00:18, 8960.50it/s] 58%|█████▊    | 231945/400000 [00:26<00:18, 8933.41it/s] 58%|█████▊    | 232839/400000 [00:26<00:19, 8783.53it/s] 58%|█████▊    | 233719/400000 [00:26<00:18, 8769.85it/s] 59%|█████▊    | 234625/400000 [00:26<00:18, 8852.16it/s] 59%|█████▉    | 235516/400000 [00:26<00:18, 8868.15it/s] 59%|█████▉    | 236405/400000 [00:26<00:18, 8872.56it/s] 59%|█████▉    | 237305/400000 [00:26<00:18, 8908.03it/s] 60%|█████▉    | 238198/400000 [00:26<00:18, 8911.74it/s] 60%|█████▉    | 239090/400000 [00:27<00:18, 8869.44it/s] 60%|█████▉    | 239978/400000 [00:27<00:18, 8831.88it/s] 60%|██████    | 240862/400000 [00:27<00:18, 8814.60it/s] 60%|██████    | 241744/400000 [00:27<00:17, 8806.29it/s] 61%|██████    | 242625/400000 [00:27<00:17, 8804.71it/s] 61%|██████    | 243506/400000 [00:27<00:17, 8778.92it/s] 61%|██████    | 244384/400000 [00:27<00:17, 8766.26it/s] 61%|██████▏   | 245261/400000 [00:27<00:17, 8741.12it/s] 62%|██████▏   | 246136/400000 [00:27<00:17, 8742.19it/s] 62%|██████▏   | 247016/400000 [00:27<00:17, 8757.75it/s] 62%|██████▏   | 247914/400000 [00:28<00:17, 8821.29it/s] 62%|██████▏   | 248815/400000 [00:28<00:17, 8874.67it/s] 62%|██████▏   | 249707/400000 [00:28<00:16, 8886.81it/s] 63%|██████▎   | 250610/400000 [00:28<00:16, 8926.46it/s] 63%|██████▎   | 251505/400000 [00:28<00:16, 8930.53it/s] 63%|██████▎   | 252401/400000 [00:28<00:16, 8938.02it/s] 63%|██████▎   | 253310/400000 [00:28<00:16, 8982.46it/s] 64%|██████▎   | 254244/400000 [00:28<00:16, 9086.78it/s] 64%|██████▍   | 255154/400000 [00:28<00:16, 9018.26it/s] 64%|██████▍   | 256057/400000 [00:28<00:16, 8965.93it/s] 64%|██████▍   | 256954/400000 [00:29<00:16, 8916.87it/s] 64%|██████▍   | 257846/400000 [00:29<00:15, 8917.15it/s] 65%|██████▍   | 258763/400000 [00:29<00:15, 8989.98it/s] 65%|██████▍   | 259687/400000 [00:29<00:15, 9060.96it/s] 65%|██████▌   | 260594/400000 [00:29<00:15, 9007.52it/s] 65%|██████▌   | 261517/400000 [00:29<00:15, 9070.63it/s] 66%|██████▌   | 262425/400000 [00:29<00:15, 8992.47it/s] 66%|██████▌   | 263325/400000 [00:29<00:15, 8969.00it/s] 66%|██████▌   | 264230/400000 [00:29<00:15, 8992.85it/s] 66%|██████▋   | 265130/400000 [00:29<00:15, 8948.64it/s] 67%|██████▋   | 266026/400000 [00:30<00:15, 8894.73it/s] 67%|██████▋   | 266916/400000 [00:30<00:15, 8828.81it/s] 67%|██████▋   | 267800/400000 [00:30<00:14, 8824.17it/s] 67%|██████▋   | 268692/400000 [00:30<00:14, 8851.70it/s] 67%|██████▋   | 269578/400000 [00:30<00:14, 8823.87it/s] 68%|██████▊   | 270471/400000 [00:30<00:14, 8854.73it/s] 68%|██████▊   | 271357/400000 [00:30<00:14, 8838.30it/s] 68%|██████▊   | 272241/400000 [00:30<00:14, 8770.92it/s] 68%|██████▊   | 273120/400000 [00:30<00:14, 8774.92it/s] 68%|██████▊   | 274000/400000 [00:30<00:14, 8781.33it/s] 69%|██████▊   | 274879/400000 [00:31<00:14, 8752.77it/s] 69%|██████▉   | 275755/400000 [00:31<00:14, 8731.63it/s] 69%|██████▉   | 276629/400000 [00:31<00:14, 8728.19it/s] 69%|██████▉   | 277505/400000 [00:31<00:14, 8734.85it/s] 70%|██████▉   | 278386/400000 [00:31<00:13, 8754.53it/s] 70%|██████▉   | 279262/400000 [00:31<00:13, 8741.82it/s] 70%|███████   | 280139/400000 [00:31<00:13, 8747.35it/s] 70%|███████   | 281018/400000 [00:31<00:13, 8759.15it/s] 70%|███████   | 281899/400000 [00:31<00:13, 8771.35it/s] 71%|███████   | 282779/400000 [00:31<00:13, 8778.08it/s] 71%|███████   | 283657/400000 [00:32<00:13, 8764.56it/s] 71%|███████   | 284536/400000 [00:32<00:13, 8771.19it/s] 71%|███████▏  | 285440/400000 [00:32<00:12, 8849.20it/s] 72%|███████▏  | 286387/400000 [00:32<00:12, 9023.99it/s] 72%|███████▏  | 287300/400000 [00:32<00:12, 9054.33it/s] 72%|███████▏  | 288207/400000 [00:32<00:12, 8991.00it/s] 72%|███████▏  | 289107/400000 [00:32<00:12, 8912.29it/s] 73%|███████▎  | 290004/400000 [00:32<00:12, 8927.63it/s] 73%|███████▎  | 290940/400000 [00:32<00:12, 9052.13it/s] 73%|███████▎  | 291846/400000 [00:32<00:11, 9045.43it/s] 73%|███████▎  | 292752/400000 [00:33<00:11, 9015.43it/s] 73%|███████▎  | 293676/400000 [00:33<00:11, 9080.98it/s] 74%|███████▎  | 294594/400000 [00:33<00:11, 9109.09it/s] 74%|███████▍  | 295512/400000 [00:33<00:11, 9128.08it/s] 74%|███████▍  | 296426/400000 [00:33<00:11, 9097.12it/s] 74%|███████▍  | 297336/400000 [00:33<00:11, 9079.05it/s] 75%|███████▍  | 298245/400000 [00:33<00:11, 8971.99it/s] 75%|███████▍  | 299143/400000 [00:33<00:11, 8954.78it/s] 75%|███████▌  | 300039/400000 [00:33<00:11, 8852.10it/s] 75%|███████▌  | 300925/400000 [00:33<00:11, 8845.96it/s] 75%|███████▌  | 301810/400000 [00:34<00:11, 8842.68it/s] 76%|███████▌  | 302695/400000 [00:34<00:11, 8797.47it/s] 76%|███████▌  | 303575/400000 [00:34<00:10, 8793.43it/s] 76%|███████▌  | 304459/400000 [00:34<00:10, 8805.02it/s] 76%|███████▋  | 305343/400000 [00:34<00:10, 8813.32it/s] 77%|███████▋  | 306228/400000 [00:34<00:10, 8822.17it/s] 77%|███████▋  | 307111/400000 [00:34<00:10, 8549.74it/s] 77%|███████▋  | 307991/400000 [00:34<00:10, 8621.20it/s] 77%|███████▋  | 308856/400000 [00:34<00:10, 8628.64it/s] 77%|███████▋  | 309728/400000 [00:35<00:10, 8654.87it/s] 78%|███████▊  | 310600/400000 [00:35<00:10, 8673.31it/s] 78%|███████▊  | 311468/400000 [00:35<00:10, 8650.98it/s] 78%|███████▊  | 312349/400000 [00:35<00:10, 8696.13it/s] 78%|███████▊  | 313224/400000 [00:35<00:09, 8709.93it/s] 79%|███████▊  | 314104/400000 [00:35<00:09, 8736.29it/s] 79%|███████▊  | 314982/400000 [00:35<00:09, 8747.55it/s] 79%|███████▉  | 315857/400000 [00:35<00:09, 8701.20it/s] 79%|███████▉  | 316728/400000 [00:35<00:09, 8697.86it/s] 79%|███████▉  | 317613/400000 [00:35<00:09, 8742.09it/s] 80%|███████▉  | 318496/400000 [00:36<00:09, 8767.25it/s] 80%|███████▉  | 319376/400000 [00:36<00:09, 8776.75it/s] 80%|████████  | 320259/400000 [00:36<00:09, 8791.97it/s] 80%|████████  | 321142/400000 [00:36<00:08, 8802.54it/s] 81%|████████  | 322029/400000 [00:36<00:08, 8820.50it/s] 81%|████████  | 322912/400000 [00:36<00:08, 8813.50it/s] 81%|████████  | 323797/400000 [00:36<00:08, 8822.19it/s] 81%|████████  | 324683/400000 [00:36<00:08, 8832.88it/s] 81%|████████▏ | 325567/400000 [00:36<00:08, 8832.30it/s] 82%|████████▏ | 326451/400000 [00:36<00:08, 8815.13it/s] 82%|████████▏ | 327335/400000 [00:37<00:08, 8819.95it/s] 82%|████████▏ | 328218/400000 [00:37<00:08, 8819.40it/s] 82%|████████▏ | 329103/400000 [00:37<00:08, 8826.83it/s] 82%|████████▏ | 329986/400000 [00:37<00:07, 8820.73it/s] 83%|████████▎ | 330869/400000 [00:37<00:07, 8818.57it/s] 83%|████████▎ | 331755/400000 [00:37<00:07, 8828.09it/s] 83%|████████▎ | 332640/400000 [00:37<00:07, 8833.30it/s] 83%|████████▎ | 333526/400000 [00:37<00:07, 8839.56it/s] 84%|████████▎ | 334410/400000 [00:37<00:07, 8794.99it/s] 84%|████████▍ | 335291/400000 [00:37<00:07, 8799.08it/s] 84%|████████▍ | 336179/400000 [00:38<00:07, 8821.34it/s] 84%|████████▍ | 337066/400000 [00:38<00:07, 8833.39it/s] 84%|████████▍ | 337952/400000 [00:38<00:07, 8840.14it/s] 85%|████████▍ | 338837/400000 [00:38<00:06, 8835.98it/s] 85%|████████▍ | 339723/400000 [00:38<00:06, 8841.89it/s] 85%|████████▌ | 340610/400000 [00:38<00:06, 8847.86it/s] 85%|████████▌ | 341497/400000 [00:38<00:06, 8853.25it/s] 86%|████████▌ | 342383/400000 [00:38<00:06, 8854.89it/s] 86%|████████▌ | 343269/400000 [00:38<00:06, 8856.08it/s] 86%|████████▌ | 344155/400000 [00:38<00:06, 8856.54it/s] 86%|████████▋ | 345041/400000 [00:39<00:06, 8856.71it/s] 86%|████████▋ | 345927/400000 [00:39<00:06, 8848.00it/s] 87%|████████▋ | 346812/400000 [00:39<00:06, 8847.76it/s] 87%|████████▋ | 347697/400000 [00:39<00:05, 8837.49it/s] 87%|████████▋ | 348582/400000 [00:39<00:05, 8838.56it/s] 87%|████████▋ | 349468/400000 [00:39<00:05, 8843.99it/s] 88%|████████▊ | 350353/400000 [00:39<00:05, 8841.73it/s] 88%|████████▊ | 351238/400000 [00:39<00:05, 8842.99it/s] 88%|████████▊ | 352123/400000 [00:39<00:05, 8818.31it/s] 88%|████████▊ | 353005/400000 [00:39<00:05, 8817.80it/s] 88%|████████▊ | 353887/400000 [00:40<00:05, 8816.39it/s] 89%|████████▊ | 354769/400000 [00:40<00:05, 8784.63it/s] 89%|████████▉ | 355653/400000 [00:40<00:05, 8798.34it/s] 89%|████████▉ | 356537/400000 [00:40<00:04, 8809.10it/s] 89%|████████▉ | 357423/400000 [00:40<00:04, 8822.26it/s] 90%|████████▉ | 358309/400000 [00:40<00:04, 8832.46it/s] 90%|████████▉ | 359195/400000 [00:40<00:04, 8840.35it/s] 90%|█████████ | 360080/400000 [00:40<00:04, 8826.09it/s] 90%|█████████ | 360963/400000 [00:40<00:04, 8817.18it/s] 90%|█████████ | 361848/400000 [00:40<00:04, 8826.48it/s] 91%|█████████ | 362735/400000 [00:41<00:04, 8838.09it/s] 91%|█████████ | 363619/400000 [00:41<00:04, 8829.75it/s] 91%|█████████ | 364502/400000 [00:41<00:04, 8822.49it/s] 91%|█████████▏| 365385/400000 [00:41<00:03, 8820.06it/s] 92%|█████████▏| 366273/400000 [00:41<00:03, 8837.38it/s] 92%|█████████▏| 367175/400000 [00:41<00:03, 8889.07it/s] 92%|█████████▏| 368112/400000 [00:41<00:03, 9025.50it/s] 92%|█████████▏| 369039/400000 [00:41<00:03, 9095.06it/s] 92%|█████████▏| 369950/400000 [00:41<00:03, 9068.35it/s] 93%|█████████▎| 370858/400000 [00:41<00:03, 9046.98it/s] 93%|█████████▎| 371787/400000 [00:42<00:03, 9115.62it/s] 93%|█████████▎| 372699/400000 [00:42<00:03, 9069.74it/s] 93%|█████████▎| 373607/400000 [00:42<00:02, 9061.99it/s] 94%|█████████▎| 374530/400000 [00:42<00:02, 9110.63it/s] 94%|█████████▍| 375442/400000 [00:42<00:02, 8974.32it/s] 94%|█████████▍| 376341/400000 [00:42<00:02, 8942.28it/s] 94%|█████████▍| 377236/400000 [00:42<00:02, 8919.34it/s] 95%|█████████▍| 378129/400000 [00:42<00:02, 8914.96it/s] 95%|█████████▍| 379023/400000 [00:42<00:02, 8919.77it/s] 95%|█████████▍| 379916/400000 [00:42<00:02, 8913.73it/s] 95%|█████████▌| 380808/400000 [00:43<00:02, 8882.21it/s] 95%|█████████▌| 381697/400000 [00:43<00:02, 8843.94it/s] 96%|█████████▌| 382582/400000 [00:43<00:01, 8839.84it/s] 96%|█████████▌| 383467/400000 [00:43<00:01, 8800.05it/s] 96%|█████████▌| 384348/400000 [00:43<00:01, 8792.00it/s] 96%|█████████▋| 385231/400000 [00:43<00:01, 8801.61it/s] 97%|█████████▋| 386112/400000 [00:43<00:01, 8790.87it/s] 97%|█████████▋| 386992/400000 [00:43<00:01, 8754.13it/s] 97%|█████████▋| 387868/400000 [00:43<00:01, 8750.53it/s] 97%|█████████▋| 388748/400000 [00:43<00:01, 8762.95it/s] 97%|█████████▋| 389631/400000 [00:44<00:01, 8780.46it/s] 98%|█████████▊| 390510/400000 [00:44<00:01, 8778.89it/s] 98%|█████████▊| 391393/400000 [00:44<00:00, 8793.98it/s] 98%|█████████▊| 392277/400000 [00:44<00:00, 8805.60it/s] 98%|█████████▊| 393160/400000 [00:44<00:00, 8810.59it/s] 99%|█████████▊| 394042/400000 [00:44<00:00, 8804.51it/s] 99%|█████████▊| 394923/400000 [00:44<00:00, 8790.84it/s] 99%|█████████▉| 395808/400000 [00:44<00:00, 8807.51it/s] 99%|█████████▉| 396692/400000 [00:44<00:00, 8814.50it/s] 99%|█████████▉| 397574/400000 [00:44<00:00, 8813.11it/s]100%|█████████▉| 398456/400000 [00:45<00:00, 8806.62it/s]100%|█████████▉| 399337/400000 [00:45<00:00, 8800.98it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8844.75it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5f31102f28> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011217477580849535 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010984155047298674 	 Accuracy: 62

  model saves at 62% accuracy 

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

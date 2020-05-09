
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f0e940bd470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 18:13:47.821565
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 18:13:47.825663
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 18:13:47.830110
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 18:13:47.835086
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0e6d13d940> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 359083.1250
Epoch 2/10

1/1 [==============================] - 0s 99ms/step - loss: 282845.2812
Epoch 3/10

1/1 [==============================] - 0s 106ms/step - loss: 186649.1406
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 116558.3672
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 68192.2188
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 38950.8008
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 23231.8691
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 15012.3926
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 10462.0977
Epoch 10/10

1/1 [==============================] - 0s 108ms/step - loss: 7776.6841

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.08085343  8.295476    6.8303185   7.3100786   8.490483    7.6433005
   7.241297    7.9839063   8.283101    7.7955337   7.8837013   8.288125
   6.6872144   6.7352443   8.733541    7.5388403   8.15667     8.144166
   7.241594    7.092708    8.220198    7.2181835   7.3289623   7.0609007
   7.820239    7.999194    6.078749    7.702414    6.423861    7.413558
   8.49561     8.995223    8.965975    6.226088    7.6246533   8.1445675
   7.5828767   7.887521    6.7943816   7.203119    7.448255    9.340083
   6.609598    8.132931    7.1297965   8.985517    8.020164    7.867134
   8.3934145   7.024079    6.30971     8.849324    7.802168    8.186996
   7.586252    6.997376    8.080239    7.7813716   7.299515    7.039211
   0.26073265 -0.91100794  0.97494966 -0.04914958 -0.29200506 -0.89067644
  -0.7629268  -0.45580593  1.7510415  -0.6475477   0.52730757  0.805568
   0.26492992  1.0926185   0.38914946 -1.0109261   0.77158505  0.6645328
   0.3627621  -0.7736126   0.03827814  0.21774709 -0.2497795   1.1106277
   1.58577     1.0209618  -0.22391216  0.37220052  0.5720785  -0.82843053
   0.17603853  0.56693554  1.0732946  -0.5242819   0.62117857  0.24176542
  -1.1425545   0.28527638  0.27603167 -0.19240579  0.7736615   1.4514045
  -0.38871682  0.828983    0.10257265 -0.19580504  1.2718326   0.4515838
   1.0705504   0.64762986  1.1029434  -0.08108479 -0.31714296  0.65977645
  -0.13938808 -0.07271376 -0.685169   -0.47885525 -0.55222434 -0.09796
  -0.975462   -0.5708461  -0.6990131  -0.12309217 -1.8149546  -1.5979259
  -0.630996    0.14739965 -1.5938565  -0.589488   -1.24121     0.07024105
   0.02377689 -1.3996868   0.5103048   0.33794427  1.1055877   0.65015954
  -0.11940521  0.16182956 -0.71730816 -0.8153978  -0.6048069  -0.19326758
   0.5174554  -0.20150578 -0.60236883  0.5982539   0.41930175  0.8135961
   0.3501333   1.2439737   0.25846067 -0.5109085   0.24903408  1.5187405
   1.6907881   0.9936749  -0.55100584 -0.20415612 -0.93975306 -0.35052785
  -0.56267965  0.7698899  -0.31946573  0.27730498  0.8035526  -0.5362369
  -0.5322552  -1.8887937  -1.65681    -0.25961936 -0.6677916   0.65321255
   0.42508522  0.5168319  -0.14832371  0.5868211  -0.6040678   0.01859975
   0.07745445  8.244377    8.131869    8.5432205   8.39594     7.1776323
   6.829596    8.587678    6.6121173   7.319827    7.574623    8.245416
   8.808816    8.413069    7.7141037   7.4789085   7.964821    8.08325
   9.3838215   8.403062    8.340309    7.269773    7.281628    7.997279
   8.718874    7.082743    7.7653685   8.030451    9.121412    7.911446
   8.498522    6.1717424   8.659148    7.7739925   8.776548    6.972072
   8.686359    7.8267674   7.2585773   6.815136    7.346656    7.6055145
   6.908721    7.823841    8.879983    6.546142    6.377183    8.064327
   6.6189756   7.2802157   7.6727943   8.285672    6.5247664   8.758088
   7.253994    7.1025004   8.83876     7.7694783   6.857999    7.6478243
   0.47101182  2.002192    1.4826165   0.42269003  2.368163    0.782555
   1.3722768   0.5932659   0.5143706   2.031211    0.9895217   0.75631285
   0.9217922   1.0255995   0.35638452  2.7166398   0.6879945   0.74353075
   0.23035026  1.7791352   0.8324466   0.3439772   1.6308284   0.85327744
   1.0631831   1.4349356   0.845222    0.5284178   0.6538512   0.74367934
   1.3373348   2.125391    1.0080417   0.27616858  0.4973737   0.95084846
   0.51158816  2.2537775   1.702425    0.56544524  1.7531033   2.5410545
   1.0875816   1.7398489   1.4710451   1.1686381   0.41981184  1.654489
   1.1884539   0.35838902  1.5419981   0.81557935  1.0610809   0.24747169
   1.4573677   0.4789474   2.089839    0.31343883  1.1320112   1.1236333
   1.323751    1.6191442   0.6883404   1.8222095   0.45728374  1.0322933
   1.235686    2.046829    1.0434649   0.29870296  0.40922755  3.111177
   1.1180911   0.63770056  1.6622981   0.58401024  0.38911724  0.40784293
   0.21665621  2.1695495   0.899621    1.35569     1.8847812   1.0954845
   1.5536796   1.3549027   0.42170614  0.8733164   0.3762604   0.13705671
   0.9490723   1.0874203   1.0785847   1.2379591   0.51001817  1.3464401
   0.47763062  0.74374413  1.0663548   0.18245476  0.3984149   1.8947749
   0.18261719  1.5401155   0.70340955  2.1987052   1.6499072   0.47918397
   1.3214269   1.7854284   1.5889062   2.2361093   1.6503124   0.9139898
   0.21701914  2.0547223   1.0345081   0.36977684  0.76559734  0.44423258
   4.4444995  -1.5503145  -5.953247  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 18:13:56.657347
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.3345
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 18:13:56.661698
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8923.93
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 18:13:56.665423
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6527
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 18:13:56.669026
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -798.201
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139699918378040
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139698959516504
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139698959517008
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139698959517512
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139698959518016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139698959518520

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f0e87b1f0f0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.577085
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.533728
grad_step = 000002, loss = 0.504830
grad_step = 000003, loss = 0.475036
grad_step = 000004, loss = 0.442350
grad_step = 000005, loss = 0.412141
grad_step = 000006, loss = 0.393376
grad_step = 000007, loss = 0.387570
grad_step = 000008, loss = 0.375207
grad_step = 000009, loss = 0.355136
grad_step = 000010, loss = 0.340117
grad_step = 000011, loss = 0.331229
grad_step = 000012, loss = 0.323744
grad_step = 000013, loss = 0.314286
grad_step = 000014, loss = 0.302654
grad_step = 000015, loss = 0.290634
grad_step = 000016, loss = 0.280234
grad_step = 000017, loss = 0.271501
grad_step = 000018, loss = 0.262466
grad_step = 000019, loss = 0.252730
grad_step = 000020, loss = 0.243524
grad_step = 000021, loss = 0.235297
grad_step = 000022, loss = 0.227589
grad_step = 000023, loss = 0.219805
grad_step = 000024, loss = 0.211646
grad_step = 000025, loss = 0.203136
grad_step = 000026, loss = 0.194665
grad_step = 000027, loss = 0.186860
grad_step = 000028, loss = 0.179686
grad_step = 000029, loss = 0.172293
grad_step = 000030, loss = 0.164689
grad_step = 000031, loss = 0.157541
grad_step = 000032, loss = 0.150889
grad_step = 000033, loss = 0.144358
grad_step = 000034, loss = 0.137863
grad_step = 000035, loss = 0.131527
grad_step = 000036, loss = 0.125311
grad_step = 000037, loss = 0.119293
grad_step = 000038, loss = 0.113479
grad_step = 000039, loss = 0.107838
grad_step = 000040, loss = 0.102413
grad_step = 000041, loss = 0.097231
grad_step = 000042, loss = 0.092255
grad_step = 000043, loss = 0.087447
grad_step = 000044, loss = 0.082727
grad_step = 000045, loss = 0.077135
grad_step = 000046, loss = 0.071708
grad_step = 000047, loss = 0.066860
grad_step = 000048, loss = 0.062624
grad_step = 000049, loss = 0.058863
grad_step = 000050, loss = 0.055413
grad_step = 000051, loss = 0.052043
grad_step = 000052, loss = 0.048771
grad_step = 000053, loss = 0.045755
grad_step = 000054, loss = 0.042914
grad_step = 000055, loss = 0.040097
grad_step = 000056, loss = 0.037347
grad_step = 000057, loss = 0.034757
grad_step = 000058, loss = 0.032339
grad_step = 000059, loss = 0.030137
grad_step = 000060, loss = 0.028178
grad_step = 000061, loss = 0.026355
grad_step = 000062, loss = 0.024603
grad_step = 000063, loss = 0.022946
grad_step = 000064, loss = 0.021342
grad_step = 000065, loss = 0.019803
grad_step = 000066, loss = 0.018388
grad_step = 000067, loss = 0.017066
grad_step = 000068, loss = 0.015738
grad_step = 000069, loss = 0.014590
grad_step = 000070, loss = 0.013520
grad_step = 000071, loss = 0.012578
grad_step = 000072, loss = 0.011697
grad_step = 000073, loss = 0.010813
grad_step = 000074, loss = 0.010016
grad_step = 000075, loss = 0.009231
grad_step = 000076, loss = 0.008537
grad_step = 000077, loss = 0.007905
grad_step = 000078, loss = 0.007300
grad_step = 000079, loss = 0.006759
grad_step = 000080, loss = 0.006255
grad_step = 000081, loss = 0.005824
grad_step = 000082, loss = 0.005419
grad_step = 000083, loss = 0.005030
grad_step = 000084, loss = 0.004678
grad_step = 000085, loss = 0.004349
grad_step = 000086, loss = 0.004077
grad_step = 000087, loss = 0.003823
grad_step = 000088, loss = 0.003597
grad_step = 000089, loss = 0.003390
grad_step = 000090, loss = 0.003207
grad_step = 000091, loss = 0.003054
grad_step = 000092, loss = 0.002909
grad_step = 000093, loss = 0.002785
grad_step = 000094, loss = 0.002670
grad_step = 000095, loss = 0.002574
grad_step = 000096, loss = 0.002490
grad_step = 000097, loss = 0.002419
grad_step = 000098, loss = 0.002359
grad_step = 000099, loss = 0.002304
grad_step = 000100, loss = 0.002261
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002220
grad_step = 000102, loss = 0.002186
grad_step = 000103, loss = 0.002153
grad_step = 000104, loss = 0.002127
grad_step = 000105, loss = 0.002104
grad_step = 000106, loss = 0.002085
grad_step = 000107, loss = 0.002068
grad_step = 000108, loss = 0.002053
grad_step = 000109, loss = 0.002039
grad_step = 000110, loss = 0.002027
grad_step = 000111, loss = 0.002017
grad_step = 000112, loss = 0.002006
grad_step = 000113, loss = 0.001997
grad_step = 000114, loss = 0.001989
grad_step = 000115, loss = 0.001982
grad_step = 000116, loss = 0.001975
grad_step = 000117, loss = 0.001968
grad_step = 000118, loss = 0.001961
grad_step = 000119, loss = 0.001954
grad_step = 000120, loss = 0.001948
grad_step = 000121, loss = 0.001942
grad_step = 000122, loss = 0.001936
grad_step = 000123, loss = 0.001929
grad_step = 000124, loss = 0.001923
grad_step = 000125, loss = 0.001916
grad_step = 000126, loss = 0.001910
grad_step = 000127, loss = 0.001903
grad_step = 000128, loss = 0.001896
grad_step = 000129, loss = 0.001889
grad_step = 000130, loss = 0.001882
grad_step = 000131, loss = 0.001875
grad_step = 000132, loss = 0.001867
grad_step = 000133, loss = 0.001860
grad_step = 000134, loss = 0.001852
grad_step = 000135, loss = 0.001845
grad_step = 000136, loss = 0.001837
grad_step = 000137, loss = 0.001829
grad_step = 000138, loss = 0.001821
grad_step = 000139, loss = 0.001812
grad_step = 000140, loss = 0.001804
grad_step = 000141, loss = 0.001796
grad_step = 000142, loss = 0.001790
grad_step = 000143, loss = 0.001793
grad_step = 000144, loss = 0.001833
grad_step = 000145, loss = 0.001923
grad_step = 000146, loss = 0.001953
grad_step = 000147, loss = 0.001765
grad_step = 000148, loss = 0.001837
grad_step = 000149, loss = 0.001895
grad_step = 000150, loss = 0.001747
grad_step = 000151, loss = 0.001842
grad_step = 000152, loss = 0.001828
grad_step = 000153, loss = 0.001732
grad_step = 000154, loss = 0.001827
grad_step = 000155, loss = 0.001752
grad_step = 000156, loss = 0.001724
grad_step = 000157, loss = 0.001790
grad_step = 000158, loss = 0.001692
grad_step = 000159, loss = 0.001726
grad_step = 000160, loss = 0.001731
grad_step = 000161, loss = 0.001669
grad_step = 000162, loss = 0.001706
grad_step = 000163, loss = 0.001686
grad_step = 000164, loss = 0.001651
grad_step = 000165, loss = 0.001677
grad_step = 000166, loss = 0.001654
grad_step = 000167, loss = 0.001630
grad_step = 000168, loss = 0.001650
grad_step = 000169, loss = 0.001627
grad_step = 000170, loss = 0.001611
grad_step = 000171, loss = 0.001620
grad_step = 000172, loss = 0.001609
grad_step = 000173, loss = 0.001588
grad_step = 000174, loss = 0.001589
grad_step = 000175, loss = 0.001590
grad_step = 000176, loss = 0.001572
grad_step = 000177, loss = 0.001560
grad_step = 000178, loss = 0.001559
grad_step = 000179, loss = 0.001558
grad_step = 000180, loss = 0.001546
grad_step = 000181, loss = 0.001531
grad_step = 000182, loss = 0.001525
grad_step = 000183, loss = 0.001522
grad_step = 000184, loss = 0.001521
grad_step = 000185, loss = 0.001518
grad_step = 000186, loss = 0.001515
grad_step = 000187, loss = 0.001512
grad_step = 000188, loss = 0.001512
grad_step = 000189, loss = 0.001517
grad_step = 000190, loss = 0.001536
grad_step = 000191, loss = 0.001570
grad_step = 000192, loss = 0.001650
grad_step = 000193, loss = 0.001694
grad_step = 000194, loss = 0.001710
grad_step = 000195, loss = 0.001561
grad_step = 000196, loss = 0.001448
grad_step = 000197, loss = 0.001462
grad_step = 000198, loss = 0.001546
grad_step = 000199, loss = 0.001585
grad_step = 000200, loss = 0.001491
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001420
grad_step = 000202, loss = 0.001438
grad_step = 000203, loss = 0.001491
grad_step = 000204, loss = 0.001507
grad_step = 000205, loss = 0.001446
grad_step = 000206, loss = 0.001398
grad_step = 000207, loss = 0.001402
grad_step = 000208, loss = 0.001434
grad_step = 000209, loss = 0.001454
grad_step = 000210, loss = 0.001428
grad_step = 000211, loss = 0.001392
grad_step = 000212, loss = 0.001369
grad_step = 000213, loss = 0.001372
grad_step = 000214, loss = 0.001390
grad_step = 000215, loss = 0.001405
grad_step = 000216, loss = 0.001412
grad_step = 000217, loss = 0.001399
grad_step = 000218, loss = 0.001381
grad_step = 000219, loss = 0.001358
grad_step = 000220, loss = 0.001342
grad_step = 000221, loss = 0.001332
grad_step = 000222, loss = 0.001328
grad_step = 000223, loss = 0.001330
grad_step = 000224, loss = 0.001336
grad_step = 000225, loss = 0.001352
grad_step = 000226, loss = 0.001384
grad_step = 000227, loss = 0.001454
grad_step = 000228, loss = 0.001542
grad_step = 000229, loss = 0.001674
grad_step = 000230, loss = 0.001630
grad_step = 000231, loss = 0.001493
grad_step = 000232, loss = 0.001316
grad_step = 000233, loss = 0.001329
grad_step = 000234, loss = 0.001452
grad_step = 000235, loss = 0.001441
grad_step = 000236, loss = 0.001331
grad_step = 000237, loss = 0.001286
grad_step = 000238, loss = 0.001351
grad_step = 000239, loss = 0.001385
grad_step = 000240, loss = 0.001313
grad_step = 000241, loss = 0.001271
grad_step = 000242, loss = 0.001310
grad_step = 000243, loss = 0.001337
grad_step = 000244, loss = 0.001303
grad_step = 000245, loss = 0.001259
grad_step = 000246, loss = 0.001267
grad_step = 000247, loss = 0.001298
grad_step = 000248, loss = 0.001297
grad_step = 000249, loss = 0.001268
grad_step = 000250, loss = 0.001242
grad_step = 000251, loss = 0.001242
grad_step = 000252, loss = 0.001259
grad_step = 000253, loss = 0.001268
grad_step = 000254, loss = 0.001262
grad_step = 000255, loss = 0.001242
grad_step = 000256, loss = 0.001225
grad_step = 000257, loss = 0.001219
grad_step = 000258, loss = 0.001222
grad_step = 000259, loss = 0.001230
grad_step = 000260, loss = 0.001234
grad_step = 000261, loss = 0.001235
grad_step = 000262, loss = 0.001228
grad_step = 000263, loss = 0.001221
grad_step = 000264, loss = 0.001211
grad_step = 000265, loss = 0.001203
grad_step = 000266, loss = 0.001196
grad_step = 000267, loss = 0.001190
grad_step = 000268, loss = 0.001186
grad_step = 000269, loss = 0.001182
grad_step = 000270, loss = 0.001179
grad_step = 000271, loss = 0.001176
grad_step = 000272, loss = 0.001174
grad_step = 000273, loss = 0.001173
grad_step = 000274, loss = 0.001176
grad_step = 000275, loss = 0.001189
grad_step = 000276, loss = 0.001222
grad_step = 000277, loss = 0.001321
grad_step = 000278, loss = 0.001483
grad_step = 000279, loss = 0.001826
grad_step = 000280, loss = 0.001800
grad_step = 000281, loss = 0.001615
grad_step = 000282, loss = 0.001202
grad_step = 000283, loss = 0.001261
grad_step = 000284, loss = 0.001549
grad_step = 000285, loss = 0.001330
grad_step = 000286, loss = 0.001160
grad_step = 000287, loss = 0.001343
grad_step = 000288, loss = 0.001311
grad_step = 000289, loss = 0.001152
grad_step = 000290, loss = 0.001214
grad_step = 000291, loss = 0.001269
grad_step = 000292, loss = 0.001179
grad_step = 000293, loss = 0.001144
grad_step = 000294, loss = 0.001217
grad_step = 000295, loss = 0.001216
grad_step = 000296, loss = 0.001120
grad_step = 000297, loss = 0.001163
grad_step = 000298, loss = 0.001210
grad_step = 000299, loss = 0.001136
grad_step = 000300, loss = 0.001113
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001158
grad_step = 000302, loss = 0.001149
grad_step = 000303, loss = 0.001101
grad_step = 000304, loss = 0.001101
grad_step = 000305, loss = 0.001129
grad_step = 000306, loss = 0.001115
grad_step = 000307, loss = 0.001082
grad_step = 000308, loss = 0.001083
grad_step = 000309, loss = 0.001101
grad_step = 000310, loss = 0.001090
grad_step = 000311, loss = 0.001066
grad_step = 000312, loss = 0.001061
grad_step = 000313, loss = 0.001071
grad_step = 000314, loss = 0.001073
grad_step = 000315, loss = 0.001057
grad_step = 000316, loss = 0.001043
grad_step = 000317, loss = 0.001042
grad_step = 000318, loss = 0.001047
grad_step = 000319, loss = 0.001047
grad_step = 000320, loss = 0.001037
grad_step = 000321, loss = 0.001026
grad_step = 000322, loss = 0.001017
grad_step = 000323, loss = 0.001016
grad_step = 000324, loss = 0.001019
grad_step = 000325, loss = 0.001019
grad_step = 000326, loss = 0.001016
grad_step = 000327, loss = 0.001009
grad_step = 000328, loss = 0.001002
grad_step = 000329, loss = 0.000995
grad_step = 000330, loss = 0.000987
grad_step = 000331, loss = 0.000982
grad_step = 000332, loss = 0.000977
grad_step = 000333, loss = 0.000972
grad_step = 000334, loss = 0.000967
grad_step = 000335, loss = 0.000963
grad_step = 000336, loss = 0.000960
grad_step = 000337, loss = 0.000958
grad_step = 000338, loss = 0.000957
grad_step = 000339, loss = 0.000963
grad_step = 000340, loss = 0.000983
grad_step = 000341, loss = 0.001019
grad_step = 000342, loss = 0.001112
grad_step = 000343, loss = 0.001207
grad_step = 000344, loss = 0.001371
grad_step = 000345, loss = 0.001275
grad_step = 000346, loss = 0.001125
grad_step = 000347, loss = 0.000939
grad_step = 000348, loss = 0.000954
grad_step = 000349, loss = 0.001087
grad_step = 000350, loss = 0.001067
grad_step = 000351, loss = 0.000951
grad_step = 000352, loss = 0.000903
grad_step = 000353, loss = 0.000973
grad_step = 000354, loss = 0.001022
grad_step = 000355, loss = 0.000945
grad_step = 000356, loss = 0.000884
grad_step = 000357, loss = 0.000899
grad_step = 000358, loss = 0.000948
grad_step = 000359, loss = 0.000965
grad_step = 000360, loss = 0.000910
grad_step = 000361, loss = 0.000864
grad_step = 000362, loss = 0.000863
grad_step = 000363, loss = 0.000893
grad_step = 000364, loss = 0.000916
grad_step = 000365, loss = 0.000896
grad_step = 000366, loss = 0.000858
grad_step = 000367, loss = 0.000837
grad_step = 000368, loss = 0.000843
grad_step = 000369, loss = 0.000863
grad_step = 000370, loss = 0.000861
grad_step = 000371, loss = 0.000847
grad_step = 000372, loss = 0.000825
grad_step = 000373, loss = 0.000814
grad_step = 000374, loss = 0.000812
grad_step = 000375, loss = 0.000819
grad_step = 000376, loss = 0.000825
grad_step = 000377, loss = 0.000823
grad_step = 000378, loss = 0.000815
grad_step = 000379, loss = 0.000803
grad_step = 000380, loss = 0.000793
grad_step = 000381, loss = 0.000784
grad_step = 000382, loss = 0.000780
grad_step = 000383, loss = 0.000778
grad_step = 000384, loss = 0.000779
grad_step = 000385, loss = 0.000778
grad_step = 000386, loss = 0.000779
grad_step = 000387, loss = 0.000779
grad_step = 000388, loss = 0.000779
grad_step = 000389, loss = 0.000779
grad_step = 000390, loss = 0.000778
grad_step = 000391, loss = 0.000776
grad_step = 000392, loss = 0.000773
grad_step = 000393, loss = 0.000770
grad_step = 000394, loss = 0.000765
grad_step = 000395, loss = 0.000762
grad_step = 000396, loss = 0.000756
grad_step = 000397, loss = 0.000752
grad_step = 000398, loss = 0.000746
grad_step = 000399, loss = 0.000743
grad_step = 000400, loss = 0.000737
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000734
grad_step = 000402, loss = 0.000729
grad_step = 000403, loss = 0.000728
grad_step = 000404, loss = 0.000725
grad_step = 000405, loss = 0.000726
grad_step = 000406, loss = 0.000727
grad_step = 000407, loss = 0.000735
grad_step = 000408, loss = 0.000742
grad_step = 000409, loss = 0.000763
grad_step = 000410, loss = 0.000782
grad_step = 000411, loss = 0.000824
grad_step = 000412, loss = 0.000830
grad_step = 000413, loss = 0.000834
grad_step = 000414, loss = 0.000783
grad_step = 000415, loss = 0.000728
grad_step = 000416, loss = 0.000682
grad_step = 000417, loss = 0.000675
grad_step = 000418, loss = 0.000699
grad_step = 000419, loss = 0.000724
grad_step = 000420, loss = 0.000736
grad_step = 000421, loss = 0.000711
grad_step = 000422, loss = 0.000684
grad_step = 000423, loss = 0.000658
grad_step = 000424, loss = 0.000651
grad_step = 000425, loss = 0.000660
grad_step = 000426, loss = 0.000672
grad_step = 000427, loss = 0.000684
grad_step = 000428, loss = 0.000679
grad_step = 000429, loss = 0.000669
grad_step = 000430, loss = 0.000650
grad_step = 000431, loss = 0.000635
grad_step = 000432, loss = 0.000627
grad_step = 000433, loss = 0.000628
grad_step = 000434, loss = 0.000632
grad_step = 000435, loss = 0.000637
grad_step = 000436, loss = 0.000640
grad_step = 000437, loss = 0.000638
grad_step = 000438, loss = 0.000634
grad_step = 000439, loss = 0.000627
grad_step = 000440, loss = 0.000619
grad_step = 000441, loss = 0.000611
grad_step = 000442, loss = 0.000605
grad_step = 000443, loss = 0.000600
grad_step = 000444, loss = 0.000595
grad_step = 000445, loss = 0.000592
grad_step = 000446, loss = 0.000589
grad_step = 000447, loss = 0.000587
grad_step = 000448, loss = 0.000585
grad_step = 000449, loss = 0.000583
grad_step = 000450, loss = 0.000582
grad_step = 000451, loss = 0.000583
grad_step = 000452, loss = 0.000585
grad_step = 000453, loss = 0.000591
grad_step = 000454, loss = 0.000602
grad_step = 000455, loss = 0.000629
grad_step = 000456, loss = 0.000669
grad_step = 000457, loss = 0.000760
grad_step = 000458, loss = 0.000844
grad_step = 000459, loss = 0.000996
grad_step = 000460, loss = 0.000955
grad_step = 000461, loss = 0.000858
grad_step = 000462, loss = 0.000645
grad_step = 000463, loss = 0.000566
grad_step = 000464, loss = 0.000646
grad_step = 000465, loss = 0.000734
grad_step = 000466, loss = 0.000711
grad_step = 000467, loss = 0.000587
grad_step = 000468, loss = 0.000546
grad_step = 000469, loss = 0.000606
grad_step = 000470, loss = 0.000654
grad_step = 000471, loss = 0.000644
grad_step = 000472, loss = 0.000579
grad_step = 000473, loss = 0.000536
grad_step = 000474, loss = 0.000542
grad_step = 000475, loss = 0.000580
grad_step = 000476, loss = 0.000603
grad_step = 000477, loss = 0.000571
grad_step = 000478, loss = 0.000534
grad_step = 000479, loss = 0.000521
grad_step = 000480, loss = 0.000529
grad_step = 000481, loss = 0.000547
grad_step = 000482, loss = 0.000549
grad_step = 000483, loss = 0.000531
grad_step = 000484, loss = 0.000509
grad_step = 000485, loss = 0.000506
grad_step = 000486, loss = 0.000514
grad_step = 000487, loss = 0.000520
grad_step = 000488, loss = 0.000521
grad_step = 000489, loss = 0.000513
grad_step = 000490, loss = 0.000500
grad_step = 000491, loss = 0.000492
grad_step = 000492, loss = 0.000492
grad_step = 000493, loss = 0.000495
grad_step = 000494, loss = 0.000497
grad_step = 000495, loss = 0.000497
grad_step = 000496, loss = 0.000492
grad_step = 000497, loss = 0.000485
grad_step = 000498, loss = 0.000479
grad_step = 000499, loss = 0.000476
grad_step = 000500, loss = 0.000475
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000476
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

  date_run                              2020-05-09 18:14:20.579854
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.265629
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 18:14:20.586649
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.199055
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 18:14:20.593948
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138859
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 18:14:20.599537
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.02471
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
0   2020-05-09 18:13:47.821565  ...    mean_absolute_error
1   2020-05-09 18:13:47.825663  ...     mean_squared_error
2   2020-05-09 18:13:47.830110  ...  median_absolute_error
3   2020-05-09 18:13:47.835086  ...               r2_score
4   2020-05-09 18:13:56.657347  ...    mean_absolute_error
5   2020-05-09 18:13:56.661698  ...     mean_squared_error
6   2020-05-09 18:13:56.665423  ...  median_absolute_error
7   2020-05-09 18:13:56.669026  ...               r2_score
8   2020-05-09 18:14:20.579854  ...    mean_absolute_error
9   2020-05-09 18:14:20.586649  ...     mean_squared_error
10  2020-05-09 18:14:20.593948  ...  median_absolute_error
11  2020-05-09 18:14:20.599537  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 318062.77it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 410831.29it/s]  5%|▍         | 458752/9912422 [00:00<00:17, 539711.68it/s]  8%|▊         | 770048/9912422 [00:00<00:12, 706289.59it/s] 11%|█         | 1097728/9912422 [00:00<00:09, 907945.55it/s] 14%|█▍        | 1433600/9912422 [00:01<00:07, 1136205.57it/s] 18%|█▊        | 1769472/9912422 [00:01<00:05, 1408817.36it/s] 21%|██▏       | 2121728/9912422 [00:01<00:04, 1624065.40it/s] 25%|██▍       | 2473984/9912422 [00:01<00:03, 1923135.79it/s] 29%|██▊       | 2834432/9912422 [00:01<00:03, 2143243.97it/s] 32%|███▏      | 3203072/9912422 [00:01<00:02, 2347888.33it/s] 36%|███▌      | 3579904/9912422 [00:01<00:02, 2528667.35it/s] 40%|███▉      | 3964928/9912422 [00:01<00:02, 2688349.16it/s] 44%|████▍     | 4349952/9912422 [00:01<00:01, 2801529.56it/s] 48%|████▊     | 4743168/9912422 [00:02<00:01, 2920150.81it/s] 52%|█████▏    | 5144576/9912422 [00:02<00:01, 3020472.08it/s] 56%|█████▌    | 5545984/9912422 [00:02<00:01, 3081021.22it/s] 60%|██████    | 5955584/9912422 [00:02<00:01, 3175949.01it/s] 64%|██████▍   | 6365184/9912422 [00:02<00:01, 3220599.53it/s] 68%|██████▊   | 6782976/9912422 [00:02<00:00, 3276772.73it/s] 73%|███████▎  | 7200768/9912422 [00:02<00:00, 3317492.85it/s] 77%|███████▋  | 7618560/9912422 [00:02<00:00, 3349304.33it/s] 81%|████████  | 8044544/9912422 [00:03<00:00, 3387126.39it/s] 85%|████████▌ | 8470528/9912422 [00:03<00:00, 3417413.17it/s] 90%|████████▉ | 8904704/9912422 [00:03<00:00, 3458520.94it/s] 94%|█████████▍| 9330688/9912422 [00:03<00:00, 3452762.37it/s] 99%|█████████▊| 9764864/9912422 [00:03<00:00, 3498957.25it/s]9920512it [00:03, 2764254.88it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:01<?, ?it/s]32768it [00:01, 18724.38it/s]            
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 314029.90it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 406042.78it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 562071.96it/s]1654784it [00:00, 2811937.98it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 39043.29it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8424b37828> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f83c227aa90> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8424aeee48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f83c227ae48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8424aeee48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8424b37a90> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8424b37828> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f83cb59b400> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f83d74ecc88> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f83c227ae48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f83d74ecc88> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5bf7213208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f9bd1c2f4b8c2b2314bf91280c56a8c31214e457381bc1298dbd1538b34cfbe6
  Stored in directory: /tmp/pip-ephem-wheel-cache-qyfonkj1/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5b8feeff60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
  106496/17464789 [..............................] - ETA: 32s
  196608/17464789 [..............................] - ETA: 23s
  417792/17464789 [..............................] - ETA: 13s
  835584/17464789 [>.............................] - ETA: 8s 
 1679360/17464789 [=>............................] - ETA: 4s
 3334144/17464789 [====>.........................] - ETA: 2s
 6397952/17464789 [=========>....................] - ETA: 1s
 9445376/17464789 [===============>..............] - ETA: 0s
12509184/17464789 [====================>.........] - ETA: 0s
15589376/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 18:15:55.498646: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 18:15:55.503279: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-09 18:15:55.503427: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56285dd86920 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 18:15:55.503447: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7893 - accuracy: 0.4920
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7701 - accuracy: 0.4933
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6394 - accuracy: 0.5018
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6743 - accuracy: 0.4995
11000/25000 [============>.................] - ETA: 4s - loss: 7.6764 - accuracy: 0.4994
12000/25000 [=============>................] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6867 - accuracy: 0.4987
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6521 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6601 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6701 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 350us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 18:16:11.820196
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 18:16:11.820196  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 18:16:18.979678: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 18:16:18.984469: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-09 18:16:18.984621: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56087eef1a80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 18:16:18.984640: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc1c3221cc0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6514 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.6065 - val_crf_viterbi_accuracy: 0.3467

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc19ef617f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9426 - accuracy: 0.4820
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8736 - accuracy: 0.4865 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8557 - accuracy: 0.4877
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7471 - accuracy: 0.4947
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7433 - accuracy: 0.4950
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8174 - accuracy: 0.4902
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7871 - accuracy: 0.4921
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7797 - accuracy: 0.4926
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7774 - accuracy: 0.4928
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7448 - accuracy: 0.4949
11000/25000 [============>.................] - ETA: 4s - loss: 7.7405 - accuracy: 0.4952
12000/25000 [=============>................] - ETA: 3s - loss: 7.7395 - accuracy: 0.4952
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7622 - accuracy: 0.4938
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7477 - accuracy: 0.4947
15000/25000 [=================>............] - ETA: 2s - loss: 7.7515 - accuracy: 0.4945
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7510 - accuracy: 0.4945
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7595 - accuracy: 0.4939
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7654 - accuracy: 0.4936
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7602 - accuracy: 0.4939
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7471 - accuracy: 0.4947
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7192 - accuracy: 0.4966
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7147 - accuracy: 0.4969
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6993 - accuracy: 0.4979
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 9s 353us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc163c58c88> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<13:44:40, 17.4kB/s].vector_cache/glove.6B.zip:   0%|          | 426k/862M [00:00<9:38:11, 24.8kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.28M/862M [00:00<6:42:03, 35.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.8M/862M [00:00<4:39:02, 50.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:00<3:13:29, 72.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:00<2:14:10, 103kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:01<1:33:00, 148kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:01<1:04:25, 211kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:01<45:10, 299kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<31:48, 423kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<12:13:53, 18.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:03<8:33:14, 26.2kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<6:00:39, 37.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<4:13:26, 52.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:58:34, 74.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<2:05:36, 106kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:29:40, 148kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<1:03:21, 209kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<46:15, 285kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<33:03, 398kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<25:08, 521kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<18:18, 715kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<14:49, 879kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<11:04, 1.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<09:47, 1.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:17<07:31, 1.72MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:18<07:18, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<05:48, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:20<06:05, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<05:06, 2.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<05:30, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:23<04:34, 2.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<05:11, 2.44MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<04:18, 2.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<05:01, 2.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<04:11, 3.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<03:27, 3.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<7:06:26, 29.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:28<4:57:26, 42.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<3:34:05, 58.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<2:30:50, 82.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<1:46:55, 116kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<1:15:26, 165kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<54:32, 227kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<38:47, 318kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<29:00, 424kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<20:56, 587kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<16:32, 739kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<12:13, 999kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<10:28, 1.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<07:58, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:29, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:54, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<06:01, 2.00MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:51, 2.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<05:17, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:20, 2.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<04:55, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:04, 2.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<04:43, 2.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<03:56, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<04:37, 2.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<04:06, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<03:20, 3.51MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<6:34:59, 29.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:53<4:35:20, 42.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<3:19:20, 58.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<2:21:14, 82.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:55<1:38:25, 117kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<1:39:57, 116kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<1:10:41, 163kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<50:58, 225kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<36:28, 315kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<27:07, 421kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<19:34, 583kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<15:25, 736kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<11:27, 990kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:03<08:06, 1.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<14:22, 785kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<10:40, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<09:12, 1.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<07:29, 1.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<06:51, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:41, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<05:37, 1.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:52, 2.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<05:01, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:04, 2.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:34, 2.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:48, 2.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<04:23, 2.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<03:38, 2.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<03:00, 3.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<6:05:08, 29.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<4:15:31, 42.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<2:59:35, 59.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<2:06:36, 84.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<1:29:06, 120kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<1:03:44, 167kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<45:07, 235kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<33:06, 319kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<23:43, 444kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<18:11, 576kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<13:16, 789kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<10:52, 957kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<08:10, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<07:20, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<05:40, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<05:36, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:27, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:53, 2.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:33, 2.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<04:05, 2.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<03:23, 2.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:57, 2.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<03:19, 3.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<02:43, 3.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<5:44:46, 28.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:43<4:00:06, 41.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<2:54:01, 56.8kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<2:02:30, 80.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<1:26:41, 113kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<1:01:09, 160kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<44:06, 221kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<31:21, 310kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<23:22, 414kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<16:50, 574kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<13:14, 726kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<09:45, 984kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<08:17, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<06:20, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:53, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<04:44, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<04:43, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<04:11, 2.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:26, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<03:52, 2.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:10, 2.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:42, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:18, 2.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:39, 2.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:03, 2.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<03:34, 2.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<02:58, 3.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<02:26, 3.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<5:13:59, 28.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:10<3:38:39, 41.0kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<2:37:06, 56.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<1:50:36, 80.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<1:18:13, 113kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<55:11, 161kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<39:45, 221kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<28:12, 312kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<21:03, 415kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<15:07, 577kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<11:56, 725kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<08:45, 989kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<07:29, 1.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:38, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<05:19, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:10, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:20, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:42, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:50, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<03:04, 2.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:28, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:52, 2.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:19, 2.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:46, 2.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:13, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:37, 3.12MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<03:10, 2.56MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:34, 3.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<03:07, 2.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:36, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:05, 2.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:35, 3.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<03:04, 2.58MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<02:29, 3.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:02, 2.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<02:32, 3.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<02:03, 3.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<5:00:56, 25.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:45<3:29:18, 36.9kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<2:30:33, 51.2kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<1:45:54, 72.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<1:14:41, 102kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<52:42, 145kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<37:45, 201kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<26:48, 282kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<19:49, 379kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<14:15, 526kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<11:05, 671kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<08:09, 911kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<06:50, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<05:10, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:45, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<03:40, 1.98MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:42, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:58, 2.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:38, 2.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:57, 2.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:22, 2.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:49, 2.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:17, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:44, 2.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:17, 3.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<02:41, 2.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:25, 2.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:42, 2.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:26, 2.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:42, 2.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:25, 2.78MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<02:41, 2.49MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<02:24, 2.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:39, 2.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:24, 2.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<01:54, 3.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<4:15:28, 25.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<2:58:03, 36.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<2:05:34, 51.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<1:28:19, 73.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<1:02:12, 103kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<44:00, 146kB/s]  .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<31:28, 202kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<22:31, 282kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<16:33, 380kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<12:05, 519kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<09:18, 668kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<07:00, 886kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<05:46, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<04:32, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:03, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:19, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:12, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:43, 2.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:46, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:26, 2.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:34, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:15, 2.60MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:26, 2.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:10, 2.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:21, 2.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:06, 2.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:18, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:05, 2.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:17, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<02:02, 2.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:49<02:14, 2.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<02:01, 2.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<01:36, 3.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<3:30:53, 26.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<2:26:47, 37.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:43:32, 52.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:13:11, 73.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:53<50:52, 105kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<37:43, 142kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<26:47, 199kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<19:20, 273kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<13:55, 378kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<10:25, 499kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<07:42, 674kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<06:06, 841kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<04:40, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:59, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<03:11, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:57, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:28, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:26, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:25, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:07<01:42, 2.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<13:45, 353kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<10:00, 485kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<07:38, 627kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<05:43, 834kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<04:40, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:38, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<03:12, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:36, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<02:29, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:06, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<02:08, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<01:51, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<01:27, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<2:45:50, 26.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<1:55:16, 38.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<1:21:12, 54.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<57:06, 76.8kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<40:04, 108kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<28:26, 152kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<19:48, 216kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<15:07, 281kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<10:54, 390kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<08:09, 514kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<06:09, 680kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:28<04:18, 961kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<04:51, 847kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:43, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:10, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<02:33, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:21, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:58, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:56, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:41, 2.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:44, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<01:37, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:27, 2.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:40<01:01, 3.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<14:02, 264kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<10:07, 365kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<07:30, 484kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<05:32, 655kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<04:20, 821kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:19, 1.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<02:48, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<02:15, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<01:41, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<2:08:26, 26.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<1:29:14, 38.3kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<1:02:29, 54.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<43:55, 76.8kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<30:41, 108kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<21:41, 152kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<15:23, 211kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<11:00, 294kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<08:02, 395kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<05:51, 540kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<04:12, 747kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<03:34, 868kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<02:42, 1.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:19, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:43, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:27, 2.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:13, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:16, 2.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:07, 2.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:10, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:02, 2.64MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:07, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:00, 2.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:04, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:57, 2.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:02, 2.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:55, 2.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:00, 2.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:54, 2.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<00:42, 3.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<1:31:06, 26.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<1:02:49, 38.2kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<44:05, 53.7kB/s]  .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<30:57, 76.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<21:27, 107kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<15:09, 151kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<10:39, 209kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<07:37, 292kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<05:30, 392kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<04:00, 536kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<03:02, 687kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<02:17, 910kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:51, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:27, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:16, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:03, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:59, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:50, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:44, 2.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:45, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:40, 2.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:42, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:37, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:39, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:35, 2.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:37, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:33, 2.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<00:26, 3.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<53:45, 27.8kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<36:51, 39.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<25:29, 55.8kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<17:52, 79.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<12:10, 111kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<08:34, 157kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<05:55, 217kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<04:13, 303kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<02:59, 406kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<02:10, 555kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:37, 708kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<01:12, 942kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:57, 1.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:45, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:39, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:31, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:29, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:24, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:24, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:20, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:20, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:18, 2.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:18, 2.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:15, 2.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:16, 2.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:14, 2.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:14, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:13, 2.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:12, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<00:11, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:08, 3.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<17:47, 26.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<11:44, 38.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<07:33, 54.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<05:13, 77.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<03:09, 108kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<02:11, 152kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:17, 211kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:53, 294kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:30, 395kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:21, 542kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:11, 693kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:08, 918kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:03, 1.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.39MB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 725/400000 [00:00<00:55, 7242.58it/s]  0%|          | 1480/400000 [00:00<00:54, 7329.08it/s]  1%|          | 2151/400000 [00:00<00:55, 7131.61it/s]  1%|          | 2902/400000 [00:00<00:54, 7238.45it/s]  1%|          | 3648/400000 [00:00<00:54, 7303.51it/s]  1%|          | 4399/400000 [00:00<00:53, 7363.14it/s]  1%|▏         | 5087/400000 [00:00<00:54, 7209.65it/s]  1%|▏         | 5821/400000 [00:00<00:54, 7246.11it/s]  2%|▏         | 6596/400000 [00:00<00:53, 7386.08it/s]  2%|▏         | 7424/400000 [00:01<00:51, 7631.68it/s]  2%|▏         | 8197/400000 [00:01<00:51, 7658.74it/s]  2%|▏         | 8951/400000 [00:01<00:51, 7555.73it/s]  2%|▏         | 9699/400000 [00:01<00:52, 7370.68it/s]  3%|▎         | 10432/400000 [00:01<00:53, 7294.30it/s]  3%|▎         | 11159/400000 [00:01<00:53, 7256.89it/s]  3%|▎         | 11962/400000 [00:01<00:51, 7472.06it/s]  3%|▎         | 12764/400000 [00:01<00:50, 7626.99it/s]  3%|▎         | 13528/400000 [00:01<00:51, 7496.01it/s]  4%|▎         | 14279/400000 [00:01<00:52, 7292.41it/s]  4%|▍         | 15011/400000 [00:02<00:52, 7298.09it/s]  4%|▍         | 15743/400000 [00:02<00:52, 7274.80it/s]  4%|▍         | 16476/400000 [00:02<00:52, 7289.56it/s]  4%|▍         | 17220/400000 [00:02<00:52, 7333.82it/s]  5%|▍         | 18005/400000 [00:02<00:51, 7481.24it/s]  5%|▍         | 18830/400000 [00:02<00:49, 7695.09it/s]  5%|▍         | 19644/400000 [00:02<00:48, 7821.69it/s]  5%|▌         | 20429/400000 [00:02<00:48, 7793.43it/s]  5%|▌         | 21240/400000 [00:02<00:48, 7885.38it/s]  6%|▌         | 22030/400000 [00:02<00:47, 7884.26it/s]  6%|▌         | 22820/400000 [00:03<00:49, 7624.00it/s]  6%|▌         | 23585/400000 [00:03<00:49, 7541.95it/s]  6%|▌         | 24342/400000 [00:03<00:49, 7518.17it/s]  6%|▋         | 25111/400000 [00:03<00:49, 7566.74it/s]  6%|▋         | 25884/400000 [00:03<00:49, 7612.60it/s]  7%|▋         | 26674/400000 [00:03<00:48, 7694.66it/s]  7%|▋         | 27466/400000 [00:03<00:48, 7759.60it/s]  7%|▋         | 28243/400000 [00:03<00:48, 7688.61it/s]  7%|▋         | 29013/400000 [00:03<00:48, 7682.86it/s]  7%|▋         | 29782/400000 [00:03<00:48, 7604.84it/s]  8%|▊         | 30543/400000 [00:04<00:48, 7587.19it/s]  8%|▊         | 31361/400000 [00:04<00:47, 7753.45it/s]  8%|▊         | 32138/400000 [00:04<00:48, 7624.30it/s]  8%|▊         | 32908/400000 [00:04<00:48, 7645.42it/s]  8%|▊         | 33690/400000 [00:04<00:47, 7695.15it/s]  9%|▊         | 34523/400000 [00:04<00:46, 7873.77it/s]  9%|▉         | 35312/400000 [00:04<00:46, 7770.19it/s]  9%|▉         | 36091/400000 [00:04<00:47, 7728.39it/s]  9%|▉         | 36889/400000 [00:04<00:46, 7800.67it/s]  9%|▉         | 37670/400000 [00:04<00:46, 7715.92it/s] 10%|▉         | 38448/400000 [00:05<00:46, 7734.67it/s] 10%|▉         | 39257/400000 [00:05<00:46, 7835.12it/s] 10%|█         | 40042/400000 [00:05<00:47, 7655.75it/s] 10%|█         | 40809/400000 [00:05<00:47, 7625.46it/s] 10%|█         | 41573/400000 [00:05<00:47, 7586.54it/s] 11%|█         | 42333/400000 [00:05<00:47, 7574.52it/s] 11%|█         | 43091/400000 [00:05<00:47, 7503.34it/s] 11%|█         | 43842/400000 [00:05<00:48, 7403.49it/s] 11%|█         | 44584/400000 [00:05<00:48, 7390.97it/s] 11%|█▏        | 45360/400000 [00:05<00:47, 7496.53it/s] 12%|█▏        | 46184/400000 [00:06<00:45, 7703.62it/s] 12%|█▏        | 46973/400000 [00:06<00:45, 7756.52it/s] 12%|█▏        | 47751/400000 [00:06<00:46, 7555.81it/s] 12%|█▏        | 48509/400000 [00:06<00:46, 7503.62it/s] 12%|█▏        | 49269/400000 [00:06<00:46, 7531.64it/s] 13%|█▎        | 50024/400000 [00:06<00:47, 7415.29it/s] 13%|█▎        | 50767/400000 [00:06<00:48, 7245.08it/s] 13%|█▎        | 51494/400000 [00:06<00:48, 7210.04it/s] 13%|█▎        | 52252/400000 [00:06<00:47, 7315.02it/s] 13%|█▎        | 52985/400000 [00:07<00:47, 7311.22it/s] 13%|█▎        | 53744/400000 [00:07<00:46, 7390.67it/s] 14%|█▎        | 54520/400000 [00:07<00:46, 7496.97it/s] 14%|█▍        | 55271/400000 [00:07<00:46, 7419.76it/s] 14%|█▍        | 56083/400000 [00:07<00:45, 7616.02it/s] 14%|█▍        | 56877/400000 [00:07<00:44, 7709.04it/s] 14%|█▍        | 57650/400000 [00:07<00:45, 7516.77it/s] 15%|█▍        | 58404/400000 [00:07<00:45, 7487.49it/s] 15%|█▍        | 59155/400000 [00:07<00:45, 7469.93it/s] 15%|█▍        | 59946/400000 [00:07<00:44, 7596.27it/s] 15%|█▌        | 60707/400000 [00:08<00:45, 7535.30it/s] 15%|█▌        | 61462/400000 [00:08<00:45, 7415.79it/s] 16%|█▌        | 62205/400000 [00:08<00:45, 7410.28it/s] 16%|█▌        | 62950/400000 [00:08<00:45, 7421.40it/s] 16%|█▌        | 63693/400000 [00:08<00:45, 7400.28it/s] 16%|█▌        | 64447/400000 [00:08<00:45, 7440.95it/s] 16%|█▋        | 65192/400000 [00:08<00:45, 7406.28it/s] 16%|█▋        | 65981/400000 [00:08<00:44, 7544.30it/s] 17%|█▋        | 66737/400000 [00:08<00:44, 7503.81it/s] 17%|█▋        | 67541/400000 [00:08<00:43, 7655.75it/s] 17%|█▋        | 68318/400000 [00:09<00:43, 7688.86it/s] 17%|█▋        | 69088/400000 [00:09<00:44, 7472.05it/s] 17%|█▋        | 69838/400000 [00:09<00:44, 7381.73it/s] 18%|█▊        | 70578/400000 [00:09<00:45, 7261.52it/s] 18%|█▊        | 71306/400000 [00:09<00:46, 7020.10it/s] 18%|█▊        | 72011/400000 [00:09<00:47, 6972.97it/s] 18%|█▊        | 72711/400000 [00:09<00:48, 6747.18it/s] 18%|█▊        | 73457/400000 [00:09<00:47, 6944.12it/s] 19%|█▊        | 74184/400000 [00:09<00:46, 7036.49it/s] 19%|█▊        | 74911/400000 [00:10<00:45, 7101.53it/s] 19%|█▉        | 75693/400000 [00:10<00:44, 7300.08it/s] 19%|█▉        | 76462/400000 [00:10<00:43, 7411.54it/s] 19%|█▉        | 77239/400000 [00:10<00:42, 7512.91it/s] 19%|█▉        | 77993/400000 [00:10<00:43, 7381.02it/s] 20%|█▉        | 78734/400000 [00:10<00:43, 7360.17it/s] 20%|█▉        | 79472/400000 [00:10<00:43, 7363.03it/s] 20%|██        | 80210/400000 [00:10<00:43, 7337.67it/s] 20%|██        | 80955/400000 [00:10<00:43, 7370.79it/s] 20%|██        | 81693/400000 [00:10<00:43, 7357.43it/s] 21%|██        | 82464/400000 [00:11<00:42, 7459.10it/s] 21%|██        | 83252/400000 [00:11<00:41, 7577.90it/s] 21%|██        | 84023/400000 [00:11<00:41, 7616.26it/s] 21%|██        | 84786/400000 [00:11<00:41, 7592.48it/s] 21%|██▏       | 85546/400000 [00:11<00:42, 7483.63it/s] 22%|██▏       | 86317/400000 [00:11<00:41, 7549.70it/s] 22%|██▏       | 87073/400000 [00:11<00:41, 7486.74it/s] 22%|██▏       | 87823/400000 [00:11<00:42, 7411.77it/s] 22%|██▏       | 88586/400000 [00:11<00:41, 7475.26it/s] 22%|██▏       | 89335/400000 [00:11<00:41, 7456.37it/s] 23%|██▎       | 90082/400000 [00:12<00:41, 7399.09it/s] 23%|██▎       | 90828/400000 [00:12<00:41, 7416.77it/s] 23%|██▎       | 91614/400000 [00:12<00:40, 7543.45it/s] 23%|██▎       | 92370/400000 [00:12<00:40, 7539.95it/s] 23%|██▎       | 93125/400000 [00:12<00:41, 7433.42it/s] 23%|██▎       | 93870/400000 [00:12<00:41, 7427.34it/s] 24%|██▎       | 94614/400000 [00:12<00:41, 7422.59it/s] 24%|██▍       | 95357/400000 [00:12<00:41, 7369.62it/s] 24%|██▍       | 96145/400000 [00:12<00:40, 7514.99it/s] 24%|██▍       | 96898/400000 [00:12<00:40, 7444.84it/s] 24%|██▍       | 97677/400000 [00:13<00:40, 7543.51it/s] 25%|██▍       | 98441/400000 [00:13<00:39, 7569.38it/s] 25%|██▍       | 99199/400000 [00:13<00:40, 7436.15it/s] 25%|██▍       | 99944/400000 [00:13<00:40, 7419.41it/s] 25%|██▌       | 100687/400000 [00:13<00:40, 7402.90it/s] 25%|██▌       | 101433/400000 [00:13<00:40, 7417.83it/s] 26%|██▌       | 102196/400000 [00:13<00:39, 7478.16it/s] 26%|██▌       | 102990/400000 [00:13<00:39, 7609.75it/s] 26%|██▌       | 103752/400000 [00:13<00:39, 7513.43it/s] 26%|██▌       | 104505/400000 [00:13<00:39, 7439.51it/s] 26%|██▋       | 105250/400000 [00:14<00:39, 7386.76it/s] 27%|██▋       | 106023/400000 [00:14<00:39, 7484.88it/s] 27%|██▋       | 106773/400000 [00:14<00:39, 7471.49it/s] 27%|██▋       | 107521/400000 [00:14<00:39, 7427.02it/s] 27%|██▋       | 108265/400000 [00:14<00:39, 7366.94it/s] 27%|██▋       | 109003/400000 [00:14<00:39, 7349.39it/s] 27%|██▋       | 109739/400000 [00:14<00:39, 7311.17it/s] 28%|██▊       | 110501/400000 [00:14<00:39, 7399.10it/s] 28%|██▊       | 111273/400000 [00:14<00:38, 7490.07it/s] 28%|██▊       | 112023/400000 [00:14<00:38, 7455.65it/s] 28%|██▊       | 112789/400000 [00:15<00:38, 7514.60it/s] 28%|██▊       | 113563/400000 [00:15<00:37, 7577.55it/s] 29%|██▊       | 114357/400000 [00:15<00:37, 7679.25it/s] 29%|██▉       | 115126/400000 [00:15<00:37, 7606.65it/s] 29%|██▉       | 115888/400000 [00:15<00:37, 7524.28it/s] 29%|██▉       | 116666/400000 [00:15<00:37, 7596.80it/s] 29%|██▉       | 117427/400000 [00:15<00:37, 7495.99it/s] 30%|██▉       | 118178/400000 [00:15<00:37, 7422.25it/s] 30%|██▉       | 118936/400000 [00:15<00:37, 7462.49it/s] 30%|██▉       | 119683/400000 [00:15<00:37, 7382.27it/s] 30%|███       | 120462/400000 [00:16<00:37, 7497.71it/s] 30%|███       | 121213/400000 [00:16<00:37, 7472.08it/s] 30%|███       | 121961/400000 [00:16<00:37, 7383.82it/s] 31%|███       | 122741/400000 [00:16<00:36, 7501.50it/s] 31%|███       | 123493/400000 [00:16<00:37, 7466.92it/s] 31%|███       | 124278/400000 [00:16<00:36, 7577.50it/s] 31%|███▏      | 125094/400000 [00:16<00:35, 7741.87it/s] 31%|███▏      | 125870/400000 [00:16<00:35, 7709.27it/s] 32%|███▏      | 126642/400000 [00:16<00:35, 7612.84it/s] 32%|███▏      | 127405/400000 [00:17<00:36, 7528.65it/s] 32%|███▏      | 128159/400000 [00:17<00:36, 7394.85it/s] 32%|███▏      | 128900/400000 [00:17<00:36, 7384.99it/s] 32%|███▏      | 129640/400000 [00:17<00:36, 7347.31it/s] 33%|███▎      | 130378/400000 [00:17<00:36, 7354.37it/s] 33%|███▎      | 131114/400000 [00:17<00:36, 7299.39it/s] 33%|███▎      | 131893/400000 [00:17<00:36, 7439.93it/s] 33%|███▎      | 132638/400000 [00:17<00:36, 7402.74it/s] 33%|███▎      | 133379/400000 [00:17<00:36, 7268.90it/s] 34%|███▎      | 134107/400000 [00:17<00:37, 7153.31it/s] 34%|███▎      | 134854/400000 [00:18<00:36, 7244.10it/s] 34%|███▍      | 135661/400000 [00:18<00:35, 7471.35it/s] 34%|███▍      | 136456/400000 [00:18<00:34, 7606.52it/s] 34%|███▍      | 137274/400000 [00:18<00:33, 7768.55it/s] 35%|███▍      | 138054/400000 [00:18<00:34, 7598.51it/s] 35%|███▍      | 138824/400000 [00:18<00:34, 7628.40it/s] 35%|███▍      | 139646/400000 [00:18<00:33, 7794.48it/s] 35%|███▌      | 140460/400000 [00:18<00:32, 7893.00it/s] 35%|███▌      | 141252/400000 [00:18<00:33, 7696.54it/s] 36%|███▌      | 142025/400000 [00:18<00:33, 7626.16it/s] 36%|███▌      | 142795/400000 [00:19<00:33, 7647.19it/s] 36%|███▌      | 143602/400000 [00:19<00:33, 7768.76it/s] 36%|███▌      | 144387/400000 [00:19<00:32, 7790.27it/s] 36%|███▋      | 145182/400000 [00:19<00:32, 7834.38it/s] 36%|███▋      | 145967/400000 [00:19<00:33, 7690.68it/s] 37%|███▋      | 146738/400000 [00:19<00:33, 7636.14it/s] 37%|███▋      | 147522/400000 [00:19<00:32, 7695.12it/s] 37%|███▋      | 148352/400000 [00:19<00:31, 7865.95it/s] 37%|███▋      | 149198/400000 [00:19<00:31, 8034.41it/s] 38%|███▊      | 150039/400000 [00:19<00:30, 8143.31it/s] 38%|███▊      | 150856/400000 [00:20<00:31, 7947.96it/s] 38%|███▊      | 151654/400000 [00:20<00:31, 7870.94it/s] 38%|███▊      | 152483/400000 [00:20<00:30, 7990.58it/s] 38%|███▊      | 153284/400000 [00:20<00:30, 7991.20it/s] 39%|███▊      | 154085/400000 [00:20<00:30, 7955.75it/s] 39%|███▊      | 154882/400000 [00:20<00:31, 7728.47it/s] 39%|███▉      | 155657/400000 [00:20<00:31, 7682.16it/s] 39%|███▉      | 156455/400000 [00:20<00:31, 7766.70it/s] 39%|███▉      | 157256/400000 [00:20<00:30, 7836.73it/s] 40%|███▉      | 158045/400000 [00:20<00:30, 7852.30it/s] 40%|███▉      | 158831/400000 [00:21<00:31, 7680.57it/s] 40%|███▉      | 159601/400000 [00:21<00:31, 7595.73it/s] 40%|████      | 160418/400000 [00:21<00:30, 7757.67it/s] 40%|████      | 161218/400000 [00:21<00:30, 7828.04it/s] 41%|████      | 162005/400000 [00:21<00:30, 7838.93it/s] 41%|████      | 162790/400000 [00:21<00:30, 7836.66it/s] 41%|████      | 163579/400000 [00:21<00:30, 7850.70it/s] 41%|████      | 164365/400000 [00:21<00:30, 7823.94it/s] 41%|████▏     | 165148/400000 [00:21<00:30, 7797.63it/s] 41%|████▏     | 165941/400000 [00:22<00:29, 7835.40it/s] 42%|████▏     | 166725/400000 [00:22<00:29, 7800.68it/s] 42%|████▏     | 167549/400000 [00:22<00:29, 7926.50it/s] 42%|████▏     | 168343/400000 [00:22<00:29, 7863.72it/s] 42%|████▏     | 169130/400000 [00:22<00:29, 7731.19it/s] 42%|████▏     | 169905/400000 [00:22<00:29, 7701.67it/s] 43%|████▎     | 170676/400000 [00:22<00:30, 7616.50it/s] 43%|████▎     | 171451/400000 [00:22<00:29, 7653.46it/s] 43%|████▎     | 172217/400000 [00:22<00:30, 7587.47it/s] 43%|████▎     | 173037/400000 [00:22<00:29, 7760.17it/s] 43%|████▎     | 173816/400000 [00:23<00:29, 7768.37it/s] 44%|████▎     | 174594/400000 [00:23<00:29, 7668.61it/s] 44%|████▍     | 175362/400000 [00:23<00:30, 7480.42it/s] 44%|████▍     | 176124/400000 [00:23<00:29, 7521.65it/s] 44%|████▍     | 176906/400000 [00:23<00:29, 7605.88it/s] 44%|████▍     | 177720/400000 [00:23<00:28, 7757.58it/s] 45%|████▍     | 178498/400000 [00:23<00:29, 7529.20it/s] 45%|████▍     | 179254/400000 [00:23<00:29, 7498.39it/s] 45%|████▌     | 180068/400000 [00:23<00:28, 7677.93it/s] 45%|████▌     | 180898/400000 [00:23<00:27, 7852.29it/s] 45%|████▌     | 181730/400000 [00:24<00:27, 7986.98it/s] 46%|████▌     | 182532/400000 [00:24<00:27, 7859.24it/s] 46%|████▌     | 183331/400000 [00:24<00:27, 7896.04it/s] 46%|████▌     | 184123/400000 [00:24<00:27, 7885.76it/s] 46%|████▌     | 184913/400000 [00:24<00:27, 7869.50it/s] 46%|████▋     | 185701/400000 [00:24<00:27, 7718.58it/s] 47%|████▋     | 186475/400000 [00:24<00:28, 7576.89it/s] 47%|████▋     | 187235/400000 [00:24<00:28, 7574.78it/s] 47%|████▋     | 187994/400000 [00:24<00:28, 7555.69it/s] 47%|████▋     | 188757/400000 [00:24<00:27, 7576.57it/s] 47%|████▋     | 189546/400000 [00:25<00:27, 7667.28it/s] 48%|████▊     | 190314/400000 [00:25<00:27, 7620.09it/s] 48%|████▊     | 191097/400000 [00:25<00:27, 7681.07it/s] 48%|████▊     | 191879/400000 [00:25<00:26, 7722.14it/s] 48%|████▊     | 192682/400000 [00:25<00:26, 7809.43it/s] 48%|████▊     | 193464/400000 [00:25<00:26, 7704.84it/s] 49%|████▊     | 194236/400000 [00:25<00:26, 7636.95it/s] 49%|████▉     | 195024/400000 [00:25<00:26, 7708.07it/s] 49%|████▉     | 195839/400000 [00:25<00:26, 7835.27it/s] 49%|████▉     | 196652/400000 [00:25<00:25, 7918.48it/s] 49%|████▉     | 197445/400000 [00:26<00:25, 7908.40it/s] 50%|████▉     | 198237/400000 [00:26<00:26, 7742.66it/s] 50%|████▉     | 199022/400000 [00:26<00:25, 7774.00it/s] 50%|████▉     | 199801/400000 [00:26<00:25, 7734.76it/s] 50%|█████     | 200576/400000 [00:26<00:25, 7732.54it/s] 50%|█████     | 201350/400000 [00:26<00:25, 7727.38it/s] 51%|█████     | 202124/400000 [00:26<00:26, 7601.47it/s] 51%|█████     | 202885/400000 [00:26<00:26, 7564.16it/s] 51%|█████     | 203665/400000 [00:26<00:25, 7630.80it/s] 51%|█████     | 204429/400000 [00:27<00:25, 7613.39it/s] 51%|█████▏    | 205191/400000 [00:27<00:25, 7553.86it/s] 51%|█████▏    | 205947/400000 [00:27<00:25, 7498.02it/s] 52%|█████▏    | 206732/400000 [00:27<00:25, 7598.43it/s] 52%|█████▏    | 207508/400000 [00:27<00:25, 7645.26it/s] 52%|█████▏    | 208273/400000 [00:27<00:25, 7569.25it/s] 52%|█████▏    | 209031/400000 [00:27<00:25, 7548.48it/s] 52%|█████▏    | 209787/400000 [00:27<00:25, 7433.91it/s] 53%|█████▎    | 210532/400000 [00:27<00:25, 7387.25it/s] 53%|█████▎    | 211296/400000 [00:27<00:25, 7459.58it/s] 53%|█████▎    | 212043/400000 [00:28<00:25, 7380.54it/s] 53%|█████▎    | 212799/400000 [00:28<00:25, 7431.40it/s] 53%|█████▎    | 213581/400000 [00:28<00:24, 7541.92it/s] 54%|█████▎    | 214357/400000 [00:28<00:24, 7605.28it/s] 54%|█████▍    | 215119/400000 [00:28<00:24, 7423.61it/s] 54%|█████▍    | 215863/400000 [00:28<00:25, 7365.01it/s] 54%|█████▍    | 216601/400000 [00:28<00:25, 7213.78it/s] 54%|█████▍    | 217324/400000 [00:28<00:25, 7205.44it/s] 55%|█████▍    | 218066/400000 [00:28<00:25, 7267.43it/s] 55%|█████▍    | 218818/400000 [00:28<00:24, 7340.44it/s] 55%|█████▍    | 219553/400000 [00:29<00:25, 7217.32it/s] 55%|█████▌    | 220310/400000 [00:29<00:24, 7317.26it/s] 55%|█████▌    | 221049/400000 [00:29<00:24, 7338.89it/s] 55%|█████▌    | 221793/400000 [00:29<00:24, 7367.57it/s] 56%|█████▌    | 222543/400000 [00:29<00:23, 7405.75it/s] 56%|█████▌    | 223310/400000 [00:29<00:23, 7480.68it/s] 56%|█████▌    | 224059/400000 [00:29<00:23, 7467.32it/s] 56%|█████▌    | 224807/400000 [00:29<00:23, 7384.67it/s] 56%|█████▋    | 225555/400000 [00:29<00:23, 7411.00it/s] 57%|█████▋    | 226300/400000 [00:29<00:23, 7421.65it/s] 57%|█████▋    | 227043/400000 [00:30<00:23, 7356.77it/s] 57%|█████▋    | 227841/400000 [00:30<00:22, 7532.22it/s] 57%|█████▋    | 228596/400000 [00:30<00:22, 7519.70it/s] 57%|█████▋    | 229349/400000 [00:30<00:23, 7375.23it/s] 58%|█████▊    | 230088/400000 [00:30<00:23, 7165.45it/s] 58%|█████▊    | 230855/400000 [00:30<00:23, 7307.88it/s] 58%|█████▊    | 231632/400000 [00:30<00:22, 7439.78it/s] 58%|█████▊    | 232379/400000 [00:30<00:22, 7324.06it/s] 58%|█████▊    | 233114/400000 [00:30<00:23, 7228.40it/s] 58%|█████▊    | 233839/400000 [00:30<00:23, 7185.80it/s] 59%|█████▊    | 234570/400000 [00:31<00:22, 7217.87it/s] 59%|█████▉    | 235305/400000 [00:31<00:22, 7255.02it/s] 59%|█████▉    | 236101/400000 [00:31<00:21, 7451.28it/s] 59%|█████▉    | 236898/400000 [00:31<00:21, 7599.16it/s] 59%|█████▉    | 237664/400000 [00:31<00:21, 7596.10it/s] 60%|█████▉    | 238425/400000 [00:31<00:21, 7529.28it/s] 60%|█████▉    | 239207/400000 [00:31<00:21, 7611.92it/s] 60%|█████▉    | 239970/400000 [00:31<00:21, 7586.07it/s] 60%|██████    | 240730/400000 [00:31<00:21, 7570.08it/s] 60%|██████    | 241528/400000 [00:31<00:20, 7688.36it/s] 61%|██████    | 242298/400000 [00:32<00:20, 7578.65it/s] 61%|██████    | 243108/400000 [00:32<00:20, 7724.00it/s] 61%|██████    | 243882/400000 [00:32<00:20, 7652.27it/s] 61%|██████    | 244649/400000 [00:32<00:20, 7620.61it/s] 61%|██████▏   | 245412/400000 [00:32<00:20, 7618.15it/s] 62%|██████▏   | 246175/400000 [00:32<00:20, 7525.12it/s] 62%|██████▏   | 246929/400000 [00:32<00:20, 7514.10it/s] 62%|██████▏   | 247684/400000 [00:32<00:20, 7524.77it/s] 62%|██████▏   | 248437/400000 [00:32<00:20, 7494.51it/s] 62%|██████▏   | 249215/400000 [00:33<00:19, 7577.26it/s] 62%|██████▏   | 249985/400000 [00:33<00:19, 7611.72it/s] 63%|██████▎   | 250783/400000 [00:33<00:19, 7716.55it/s] 63%|██████▎   | 251574/400000 [00:33<00:19, 7771.09it/s] 63%|██████▎   | 252352/400000 [00:33<00:19, 7752.66it/s] 63%|██████▎   | 253128/400000 [00:33<00:19, 7680.61it/s] 63%|██████▎   | 253927/400000 [00:33<00:18, 7768.59it/s] 64%|██████▎   | 254724/400000 [00:33<00:18, 7824.86it/s] 64%|██████▍   | 255531/400000 [00:33<00:18, 7896.33it/s] 64%|██████▍   | 256330/400000 [00:33<00:18, 7922.24it/s] 64%|██████▍   | 257145/400000 [00:34<00:17, 7988.64it/s] 64%|██████▍   | 257948/400000 [00:34<00:17, 8000.55it/s] 65%|██████▍   | 258788/400000 [00:34<00:17, 8114.60it/s] 65%|██████▍   | 259601/400000 [00:34<00:17, 8006.40it/s] 65%|██████▌   | 260403/400000 [00:34<00:17, 7857.67it/s] 65%|██████▌   | 261218/400000 [00:34<00:17, 7942.89it/s] 66%|██████▌   | 262054/400000 [00:34<00:17, 8063.39it/s] 66%|██████▌   | 262872/400000 [00:34<00:16, 8095.68it/s] 66%|██████▌   | 263683/400000 [00:34<00:17, 8013.54it/s] 66%|██████▌   | 264486/400000 [00:34<00:17, 7803.52it/s] 66%|██████▋   | 265269/400000 [00:35<00:17, 7805.70it/s] 67%|██████▋   | 266051/400000 [00:35<00:17, 7651.45it/s] 67%|██████▋   | 266833/400000 [00:35<00:17, 7699.59it/s] 67%|██████▋   | 267666/400000 [00:35<00:16, 7878.09it/s] 67%|██████▋   | 268456/400000 [00:35<00:16, 7756.47it/s] 67%|██████▋   | 269259/400000 [00:35<00:16, 7834.15it/s] 68%|██████▊   | 270102/400000 [00:35<00:16, 8001.87it/s] 68%|██████▊   | 270905/400000 [00:35<00:16, 7991.61it/s] 68%|██████▊   | 271747/400000 [00:35<00:15, 8113.58it/s] 68%|██████▊   | 272560/400000 [00:35<00:16, 7815.41it/s] 68%|██████▊   | 273345/400000 [00:36<00:16, 7699.69it/s] 69%|██████▊   | 274165/400000 [00:36<00:16, 7841.90it/s] 69%|██████▉   | 275008/400000 [00:36<00:15, 8008.26it/s] 69%|██████▉   | 275812/400000 [00:36<00:15, 7917.75it/s] 69%|██████▉   | 276606/400000 [00:36<00:15, 7869.22it/s] 69%|██████▉   | 277420/400000 [00:36<00:15, 7948.04it/s] 70%|██████▉   | 278248/400000 [00:36<00:15, 8042.18it/s] 70%|██████▉   | 279077/400000 [00:36<00:14, 8114.44it/s] 70%|██████▉   | 279890/400000 [00:36<00:14, 8041.66it/s] 70%|███████   | 280695/400000 [00:36<00:14, 8037.44it/s] 70%|███████   | 281500/400000 [00:37<00:14, 8010.54it/s] 71%|███████   | 282316/400000 [00:37<00:14, 8054.17it/s] 71%|███████   | 283152/400000 [00:37<00:14, 8142.58it/s] 71%|███████   | 283985/400000 [00:37<00:14, 8195.94it/s] 71%|███████   | 284806/400000 [00:37<00:14, 8129.86it/s] 71%|███████▏  | 285626/400000 [00:37<00:14, 8148.91it/s] 72%|███████▏  | 286442/400000 [00:37<00:14, 7985.32it/s] 72%|███████▏  | 287242/400000 [00:37<00:14, 7924.31it/s] 72%|███████▏  | 288036/400000 [00:37<00:14, 7918.60it/s] 72%|███████▏  | 288829/400000 [00:37<00:14, 7811.83it/s] 72%|███████▏  | 289620/400000 [00:38<00:14, 7840.87it/s] 73%|███████▎  | 290462/400000 [00:38<00:13, 8003.76it/s] 73%|███████▎  | 291281/400000 [00:38<00:13, 8058.31it/s] 73%|███████▎  | 292117/400000 [00:38<00:13, 8144.97it/s] 73%|███████▎  | 292933/400000 [00:38<00:14, 7615.80it/s] 73%|███████▎  | 293703/400000 [00:38<00:14, 7478.95it/s] 74%|███████▎  | 294472/400000 [00:38<00:13, 7540.38it/s] 74%|███████▍  | 295249/400000 [00:38<00:13, 7603.96it/s] 74%|███████▍  | 296024/400000 [00:38<00:13, 7644.83it/s] 74%|███████▍  | 296791/400000 [00:39<00:13, 7577.64it/s] 74%|███████▍  | 297558/400000 [00:39<00:13, 7603.75it/s] 75%|███████▍  | 298334/400000 [00:39<00:13, 7647.51it/s] 75%|███████▍  | 299100/400000 [00:39<00:13, 7392.24it/s] 75%|███████▍  | 299842/400000 [00:39<00:13, 7235.36it/s] 75%|███████▌  | 300572/400000 [00:39<00:13, 7254.54it/s] 75%|███████▌  | 301319/400000 [00:39<00:13, 7315.69it/s] 76%|███████▌  | 302115/400000 [00:39<00:13, 7497.75it/s] 76%|███████▌  | 302900/400000 [00:39<00:12, 7597.96it/s] 76%|███████▌  | 303694/400000 [00:39<00:12, 7693.91it/s] 76%|███████▌  | 304493/400000 [00:40<00:12, 7777.34it/s] 76%|███████▋  | 305273/400000 [00:40<00:12, 7580.32it/s] 77%|███████▋  | 306057/400000 [00:40<00:12, 7654.22it/s] 77%|███████▋  | 306824/400000 [00:40<00:12, 7623.06it/s] 77%|███████▋  | 307622/400000 [00:40<00:11, 7724.63it/s] 77%|███████▋  | 308396/400000 [00:40<00:12, 7631.85it/s] 77%|███████▋  | 309182/400000 [00:40<00:11, 7697.85it/s] 77%|███████▋  | 309973/400000 [00:40<00:11, 7757.94it/s] 78%|███████▊  | 310750/400000 [00:40<00:11, 7711.24it/s] 78%|███████▊  | 311522/400000 [00:40<00:11, 7598.26it/s] 78%|███████▊  | 312283/400000 [00:41<00:11, 7367.60it/s] 78%|███████▊  | 313061/400000 [00:41<00:11, 7486.64it/s] 78%|███████▊  | 313835/400000 [00:41<00:11, 7559.89it/s] 79%|███████▊  | 314593/400000 [00:41<00:11, 7549.62it/s] 79%|███████▉  | 315350/400000 [00:41<00:11, 7524.33it/s] 79%|███████▉  | 316104/400000 [00:41<00:11, 7460.21it/s] 79%|███████▉  | 316875/400000 [00:41<00:11, 7532.18it/s] 79%|███████▉  | 317650/400000 [00:41<00:10, 7593.74it/s] 80%|███████▉  | 318411/400000 [00:41<00:10, 7596.69it/s] 80%|███████▉  | 319195/400000 [00:41<00:10, 7666.66it/s] 80%|███████▉  | 319963/400000 [00:42<00:10, 7590.81it/s] 80%|████████  | 320723/400000 [00:42<00:10, 7497.10it/s] 80%|████████  | 321481/400000 [00:42<00:10, 7519.46it/s] 81%|████████  | 322239/400000 [00:42<00:10, 7537.33it/s] 81%|████████  | 322994/400000 [00:42<00:10, 7538.48it/s] 81%|████████  | 323749/400000 [00:42<00:10, 7419.44it/s] 81%|████████  | 324492/400000 [00:42<00:10, 7403.61it/s] 81%|████████▏ | 325257/400000 [00:42<00:10, 7472.29it/s] 82%|████████▏ | 326029/400000 [00:42<00:09, 7543.54it/s] 82%|████████▏ | 326793/400000 [00:43<00:09, 7570.84it/s] 82%|████████▏ | 327552/400000 [00:43<00:09, 7576.45it/s] 82%|████████▏ | 328310/400000 [00:43<00:09, 7486.63it/s] 82%|████████▏ | 329069/400000 [00:43<00:09, 7517.32it/s] 82%|████████▏ | 329837/400000 [00:43<00:09, 7564.96it/s] 83%|████████▎ | 330597/400000 [00:43<00:09, 7575.38it/s] 83%|████████▎ | 331355/400000 [00:43<00:09, 7454.92it/s] 83%|████████▎ | 332131/400000 [00:43<00:08, 7542.70it/s] 83%|████████▎ | 332924/400000 [00:43<00:08, 7653.07it/s] 83%|████████▎ | 333712/400000 [00:43<00:08, 7717.90it/s] 84%|████████▎ | 334485/400000 [00:44<00:08, 7711.31it/s] 84%|████████▍ | 335272/400000 [00:44<00:08, 7757.20it/s] 84%|████████▍ | 336049/400000 [00:44<00:08, 7531.39it/s] 84%|████████▍ | 336804/400000 [00:44<00:08, 7417.44it/s] 84%|████████▍ | 337593/400000 [00:44<00:08, 7551.32it/s] 85%|████████▍ | 338380/400000 [00:44<00:08, 7642.20it/s] 85%|████████▍ | 339146/400000 [00:44<00:07, 7619.05it/s] 85%|████████▍ | 339913/400000 [00:44<00:07, 7633.20it/s] 85%|████████▌ | 340701/400000 [00:44<00:07, 7705.43it/s] 85%|████████▌ | 341497/400000 [00:44<00:07, 7777.64it/s] 86%|████████▌ | 342276/400000 [00:45<00:07, 7670.71it/s] 86%|████████▌ | 343044/400000 [00:45<00:07, 7427.51it/s] 86%|████████▌ | 343789/400000 [00:45<00:07, 7364.57it/s] 86%|████████▌ | 344551/400000 [00:45<00:07, 7439.32it/s] 86%|████████▋ | 345297/400000 [00:45<00:07, 7421.17it/s] 87%|████████▋ | 346041/400000 [00:45<00:07, 7210.19it/s] 87%|████████▋ | 346799/400000 [00:45<00:07, 7314.72it/s] 87%|████████▋ | 347555/400000 [00:45<00:07, 7383.47it/s] 87%|████████▋ | 348307/400000 [00:45<00:06, 7422.97it/s] 87%|████████▋ | 349051/400000 [00:45<00:06, 7415.68it/s] 87%|████████▋ | 349798/400000 [00:46<00:06, 7428.87it/s] 88%|████████▊ | 350552/400000 [00:46<00:06, 7461.41it/s] 88%|████████▊ | 351307/400000 [00:46<00:06, 7485.26it/s] 88%|████████▊ | 352075/400000 [00:46<00:06, 7541.49it/s] 88%|████████▊ | 352833/400000 [00:46<00:06, 7553.00it/s] 88%|████████▊ | 353620/400000 [00:46<00:06, 7642.26it/s] 89%|████████▊ | 354385/400000 [00:46<00:06, 7552.98it/s] 89%|████████▉ | 355146/400000 [00:46<00:05, 7568.39it/s] 89%|████████▉ | 355906/400000 [00:46<00:05, 7576.32it/s] 89%|████████▉ | 356664/400000 [00:46<00:05, 7295.61it/s] 89%|████████▉ | 357396/400000 [00:47<00:05, 7144.71it/s] 90%|████████▉ | 358113/400000 [00:47<00:05, 7108.39it/s] 90%|████████▉ | 358826/400000 [00:47<00:05, 6925.41it/s] 90%|████████▉ | 359540/400000 [00:47<00:05, 6987.82it/s] 90%|█████████ | 360281/400000 [00:47<00:05, 7106.70it/s] 90%|█████████ | 360994/400000 [00:47<00:05, 7032.92it/s] 90%|█████████ | 361722/400000 [00:47<00:05, 7104.98it/s] 91%|█████████ | 362434/400000 [00:47<00:05, 6863.40it/s] 91%|█████████ | 363202/400000 [00:47<00:05, 7087.45it/s] 91%|█████████ | 363915/400000 [00:48<00:05, 7013.74it/s] 91%|█████████ | 364620/400000 [00:48<00:05, 6998.72it/s] 91%|█████████▏| 365322/400000 [00:48<00:05, 6861.80it/s] 92%|█████████▏| 366054/400000 [00:48<00:04, 6990.68it/s] 92%|█████████▏| 366810/400000 [00:48<00:04, 7149.91it/s] 92%|█████████▏| 367571/400000 [00:48<00:04, 7279.99it/s] 92%|█████████▏| 368306/400000 [00:48<00:04, 7299.92it/s] 92%|█████████▏| 369065/400000 [00:48<00:04, 7384.40it/s] 92%|█████████▏| 369841/400000 [00:48<00:04, 7492.63it/s] 93%|█████████▎| 370615/400000 [00:48<00:03, 7561.84it/s] 93%|█████████▎| 371381/400000 [00:49<00:03, 7588.39it/s] 93%|█████████▎| 372141/400000 [00:49<00:03, 7557.05it/s] 93%|█████████▎| 372898/400000 [00:49<00:03, 7339.76it/s] 93%|█████████▎| 373656/400000 [00:49<00:03, 7406.96it/s] 94%|█████████▎| 374448/400000 [00:49<00:03, 7551.32it/s] 94%|█████████▍| 375232/400000 [00:49<00:03, 7634.44it/s] 94%|█████████▍| 375997/400000 [00:49<00:03, 7571.57it/s] 94%|█████████▍| 376756/400000 [00:49<00:03, 7376.28it/s] 94%|█████████▍| 377510/400000 [00:49<00:03, 7423.31it/s] 95%|█████████▍| 378287/400000 [00:49<00:02, 7522.98it/s] 95%|█████████▍| 379045/400000 [00:50<00:02, 7538.89it/s] 95%|█████████▍| 379800/400000 [00:50<00:02, 7447.70it/s] 95%|█████████▌| 380546/400000 [00:50<00:02, 7342.95it/s] 95%|█████████▌| 381326/400000 [00:50<00:02, 7472.68it/s] 96%|█████████▌| 382075/400000 [00:50<00:02, 7390.01it/s] 96%|█████████▌| 382837/400000 [00:50<00:02, 7452.22it/s] 96%|█████████▌| 383595/400000 [00:50<00:02, 7488.10it/s] 96%|█████████▌| 384359/400000 [00:50<00:02, 7532.52it/s] 96%|█████████▋| 385120/400000 [00:50<00:01, 7554.89it/s] 96%|█████████▋| 385913/400000 [00:50<00:01, 7661.98it/s] 97%|█████████▋| 386696/400000 [00:51<00:01, 7710.88it/s] 97%|█████████▋| 387481/400000 [00:51<00:01, 7750.70it/s] 97%|█████████▋| 388257/400000 [00:51<00:01, 7701.87it/s] 97%|█████████▋| 389041/400000 [00:51<00:01, 7741.11it/s] 97%|█████████▋| 389816/400000 [00:51<00:01, 7648.87it/s] 98%|█████████▊| 390582/400000 [00:51<00:01, 7494.43it/s] 98%|█████████▊| 391333/400000 [00:51<00:01, 7497.80it/s] 98%|█████████▊| 392084/400000 [00:51<00:01, 7417.81it/s] 98%|█████████▊| 392844/400000 [00:51<00:00, 7467.79it/s] 98%|█████████▊| 393616/400000 [00:51<00:00, 7539.26it/s] 99%|█████████▊| 394374/400000 [00:52<00:00, 7550.81it/s] 99%|█████████▉| 395150/400000 [00:52<00:00, 7612.21it/s] 99%|█████████▉| 395912/400000 [00:52<00:00, 7581.60it/s] 99%|█████████▉| 396671/400000 [00:52<00:00, 7551.11it/s] 99%|█████████▉| 397427/400000 [00:52<00:00, 7407.87it/s]100%|█████████▉| 398169/400000 [00:52<00:00, 7274.45it/s]100%|█████████▉| 398898/400000 [00:52<00:00, 7175.69it/s]100%|█████████▉| 399617/400000 [00:52<00:00, 7098.94it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7566.94it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc163696b00> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011502212038708689 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011346738673372811 	 Accuracy: 52

  model saves at 52% accuracy 

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

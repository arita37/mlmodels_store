
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4544ee84a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 19:13:08.863494
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 19:13:08.867223
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 19:13:08.870516
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 19:13:08.873841
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f45451e5320> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356453.2500
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 257531.6719
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 172661.6875
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 101414.1250
Epoch 5/10

1/1 [==============================] - 0s 110ms/step - loss: 60614.7344
Epoch 6/10

1/1 [==============================] - 0s 109ms/step - loss: 37759.5742
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 24722.1055
Epoch 8/10

1/1 [==============================] - 0s 109ms/step - loss: 17074.9785
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 12433.0967
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 9478.4062

  #### Inference Need return ypred, ytrue ######################### 
[[-4.10823047e-01  2.73655474e-01 -1.16981649e+00  5.52267373e-01
   5.31515896e-01 -7.19512582e-01  6.04664326e-01 -5.77621698e-01
  -8.08703661e-01  4.43838388e-01  9.88521993e-01 -5.46011150e-01
  -5.57669163e-01 -2.09150046e-01 -4.04111564e-01  3.59056830e-01
   6.48646235e-01 -3.47132623e-01  7.82729745e-01  1.04815125e+00
   7.07289934e-01 -9.96470392e-01  1.16496038e+00  5.40299892e-01
  -2.15719670e-01 -4.39682037e-01 -1.06016445e+00 -1.35856462e+00
  -1.80612206e-01  1.16001356e+00  6.81881189e-01  1.49888706e+00
   5.82684517e-01 -1.00086451e+00 -6.05362594e-01 -3.73414755e-01
  -5.72651386e-01 -4.03824747e-01  3.73430073e-01 -1.33634424e+00
   7.32724667e-01  6.24473095e-02  1.73348093e+00  1.17250526e+00
   7.13865757e-01 -5.99885732e-02  4.21188980e-01  2.36436442e-01
   4.15916026e-01  3.12578678e-01 -7.05046654e-01 -2.67432988e-01
   3.42389882e-01 -2.16747180e-01  1.13174713e+00 -6.12213254e-01
  -6.28108084e-01  1.15939581e+00 -6.24976277e-01 -5.98678946e-01
  -5.87169707e-01 -1.50524437e+00 -1.75690806e+00 -2.14249030e-01
   1.11106253e+00  5.30378461e-01 -3.46129954e-01  1.14775515e+00
  -7.68041611e-03  5.68996668e-01  9.68675554e-01 -3.70154738e-01
  -2.45769963e-01  4.24061716e-02  1.42500699e-01  2.46202916e-01
   1.07946086e+00  7.94051588e-01  7.20024049e-01  1.22595978e+00
   8.94677639e-03 -1.02990413e+00  1.43178225e+00  2.58053541e-02
   9.18939710e-01  2.94736922e-02  4.04810727e-01  5.76581240e-01
   1.74848825e-01  1.05342716e-01  7.89654970e-01 -1.43309379e+00
   5.56063771e-01 -4.95074630e-01  1.22748792e+00 -5.16730130e-01
  -2.69008905e-01  1.07652032e+00 -1.18922305e+00 -6.07865930e-01
   9.66949463e-02  4.12308574e-02  1.16931200e-02  5.08705378e-01
   5.59912324e-01  6.34114385e-01  4.05648291e-01  6.48802400e-01
  -1.66933015e-01  7.13865817e-01 -6.25628650e-01  1.46074724e+00
   4.50745285e-01 -7.70485997e-02  4.39861417e-03 -1.02489185e+00
  -3.84380609e-01 -9.70534086e-01  2.77996063e-01  1.05740070e-01
   2.61411220e-01  7.38686275e+00  6.97695017e+00  6.36068249e+00
   8.06428146e+00  5.76519012e+00  7.26519442e+00  5.53604507e+00
   7.94687557e+00  6.12114143e+00  7.22343540e+00  6.15852451e+00
   6.78891993e+00  5.47384930e+00  7.79095459e+00  6.70125532e+00
   6.25759077e+00  6.76387358e+00  6.79372501e+00  7.48836184e+00
   7.50205898e+00  7.53077221e+00  7.19402695e+00  7.78078461e+00
   5.96561193e+00  6.13811827e+00  6.06358576e+00  7.92162085e+00
   6.17351246e+00  5.55480766e+00  6.77884626e+00  6.92949295e+00
   5.43144798e+00  6.65009832e+00  6.38299894e+00  5.40923882e+00
   6.54806089e+00  6.36945391e+00  7.18733978e+00  6.03272295e+00
   6.38773060e+00  8.12831020e+00  6.43355656e+00  6.00649309e+00
   5.00202131e+00  7.11558771e+00  7.99417448e+00  6.79774809e+00
   7.99074841e+00  6.94285917e+00  7.58439493e+00  6.56438017e+00
   6.32342863e+00  6.08464384e+00  6.00433350e+00  7.67086411e+00
   6.09076309e+00  5.40464020e+00  7.78673267e+00  5.01165009e+00
   4.41792011e-01  2.93199241e-01  1.74040031e+00  1.02631330e+00
   2.13140774e+00  4.30256963e-01  1.00000107e+00  2.02372217e+00
   1.21389687e+00  1.32726455e+00  5.88216662e-01  5.49330354e-01
   1.61939549e+00  6.49758339e-01  6.39018953e-01  8.40112686e-01
   8.08135629e-01  5.10205626e-01  7.53682792e-01  9.40850437e-01
   2.45836675e-01  6.19960725e-01  2.15021420e+00  2.46522570e+00
   8.42289686e-01  9.57101822e-01  1.14621496e+00  9.86434519e-01
   2.18746328e+00  1.21906817e+00  1.91171455e+00  1.05325544e+00
   1.89853954e+00  1.87107158e+00  4.81458306e-01  4.97630358e-01
   7.36128926e-01  4.35673952e-01  8.02145481e-01  4.76844788e-01
   2.17939258e-01  2.58162212e+00  3.52012098e-01  1.77426994e+00
   2.04990292e+00  8.47178638e-01  6.15779936e-01  1.73388219e+00
   7.21959233e-01  5.69889247e-01  2.39653826e+00  5.20239353e-01
   3.76506448e-01  4.06513393e-01  1.45865202e+00  3.71441185e-01
   3.18066895e-01  1.05402231e+00  8.30474675e-01  3.16585898e-01
   7.15725303e-01  3.61926198e-01  5.05320132e-01  1.94192541e+00
   8.61383677e-01  8.50555062e-01  6.28097713e-01  6.67132437e-01
   1.80263734e+00  5.41946232e-01  9.09068346e-01  7.71748126e-01
   1.35880792e+00  2.32759833e+00  1.63805962e+00  5.75867891e-01
   5.21381497e-01  2.08482981e-01  2.12680006e+00  1.86922073e+00
   1.19536579e+00  1.03898966e+00  5.87155700e-01  1.72154760e+00
   1.92356110e+00  2.62706590e+00  5.20297706e-01  2.39468908e+00
   1.18711114e+00  1.07409310e+00  1.91975641e+00  1.02736497e+00
   6.07972383e-01  1.40159762e+00  1.14522839e+00  4.48919773e-01
   1.40181613e+00  8.46333265e-01  4.38171506e-01  1.24207520e+00
   5.97348869e-01  2.66095996e-01  4.77146387e-01  1.85844266e+00
   1.63089287e+00  1.25543213e+00  1.59608126e+00  1.21177649e+00
   1.20615768e+00  2.37797499e-01  4.55255210e-01  3.08717489e-01
   3.43615055e-01  4.12436128e-01  1.23102343e+00  1.92973232e+00
   1.27643752e+00  1.34015977e+00  4.15761054e-01  1.36452460e+00
   6.96539879e-02  7.89070559e+00  6.95072079e+00  6.65914726e+00
   7.09768248e+00  8.18766308e+00  6.72416115e+00  6.70932007e+00
   5.98016500e+00  7.49853373e+00  6.88647270e+00  7.67747021e+00
   7.67698479e+00  6.96293879e+00  5.96024370e+00  6.64648485e+00
   6.34812069e+00  5.68155718e+00  7.57973957e+00  7.48747349e+00
   6.75713158e+00  6.14533901e+00  6.70423794e+00  7.15899420e+00
   6.51273155e+00  5.36909723e+00  7.92479992e+00  7.43864965e+00
   6.91354609e+00  8.27549744e+00  7.94160509e+00  6.85225916e+00
   7.26956320e+00  7.90715408e+00  6.39044237e+00  6.93734884e+00
   7.23678446e+00  6.12803078e+00  6.97992373e+00  6.85836744e+00
   6.40404415e+00  6.88826704e+00  6.86615372e+00  5.66545582e+00
   6.72730541e+00  7.68573809e+00  7.52471876e+00  8.62136078e+00
   6.81880569e+00  7.32036591e+00  7.09153938e+00  6.09098244e+00
   7.83581734e+00  6.36279106e+00  7.85923433e+00  6.82853460e+00
   7.33936930e+00  6.92817545e+00  6.08519983e+00  6.41921425e+00
  -4.25276756e+00 -1.55135572e+00  1.03445864e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 19:13:20.832425
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.8956
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 19:13:20.836162
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9214.17
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 19:13:20.839695
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4351
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 19:13:20.842825
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -824.194
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139934797025688
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139933994521432
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139933994521936
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139933994522440
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139933994522944
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139933994523448

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f451d666eb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.750140
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.695182
grad_step = 000002, loss = 0.651798
grad_step = 000003, loss = 0.604209
grad_step = 000004, loss = 0.553623
grad_step = 000005, loss = 0.517048
grad_step = 000006, loss = 0.515661
grad_step = 000007, loss = 0.498695
grad_step = 000008, loss = 0.468967
grad_step = 000009, loss = 0.446948
grad_step = 000010, loss = 0.434883
grad_step = 000011, loss = 0.426524
grad_step = 000012, loss = 0.416883
grad_step = 000013, loss = 0.404320
grad_step = 000014, loss = 0.389867
grad_step = 000015, loss = 0.375839
grad_step = 000016, loss = 0.363585
grad_step = 000017, loss = 0.351606
grad_step = 000018, loss = 0.338468
grad_step = 000019, loss = 0.325413
grad_step = 000020, loss = 0.313913
grad_step = 000021, loss = 0.303762
grad_step = 000022, loss = 0.293836
grad_step = 000023, loss = 0.283389
grad_step = 000024, loss = 0.272472
grad_step = 000025, loss = 0.261633
grad_step = 000026, loss = 0.251349
grad_step = 000027, loss = 0.240460
grad_step = 000028, loss = 0.229318
grad_step = 000029, loss = 0.218500
grad_step = 000030, loss = 0.208371
grad_step = 000031, loss = 0.198907
grad_step = 000032, loss = 0.189941
grad_step = 000033, loss = 0.181426
grad_step = 000034, loss = 0.173429
grad_step = 000035, loss = 0.165703
grad_step = 000036, loss = 0.158177
grad_step = 000037, loss = 0.150964
grad_step = 000038, loss = 0.143877
grad_step = 000039, loss = 0.136806
grad_step = 000040, loss = 0.129789
grad_step = 000041, loss = 0.123000
grad_step = 000042, loss = 0.116492
grad_step = 000043, loss = 0.110292
grad_step = 000044, loss = 0.104471
grad_step = 000045, loss = 0.098860
grad_step = 000046, loss = 0.093419
grad_step = 000047, loss = 0.088069
grad_step = 000048, loss = 0.082885
grad_step = 000049, loss = 0.077675
grad_step = 000050, loss = 0.072581
grad_step = 000051, loss = 0.067728
grad_step = 000052, loss = 0.063115
grad_step = 000053, loss = 0.058759
grad_step = 000054, loss = 0.054677
grad_step = 000055, loss = 0.050781
grad_step = 000056, loss = 0.046993
grad_step = 000057, loss = 0.043306
grad_step = 000058, loss = 0.039753
grad_step = 000059, loss = 0.036383
grad_step = 000060, loss = 0.033221
grad_step = 000061, loss = 0.030238
grad_step = 000062, loss = 0.027439
grad_step = 000063, loss = 0.024836
grad_step = 000064, loss = 0.022404
grad_step = 000065, loss = 0.020152
grad_step = 000066, loss = 0.018084
grad_step = 000067, loss = 0.016214
grad_step = 000068, loss = 0.014530
grad_step = 000069, loss = 0.013034
grad_step = 000070, loss = 0.011618
grad_step = 000071, loss = 0.010324
grad_step = 000072, loss = 0.009154
grad_step = 000073, loss = 0.008175
grad_step = 000074, loss = 0.007359
grad_step = 000075, loss = 0.006647
grad_step = 000076, loss = 0.005996
grad_step = 000077, loss = 0.005364
grad_step = 000078, loss = 0.004820
grad_step = 000079, loss = 0.004413
grad_step = 000080, loss = 0.004109
grad_step = 000081, loss = 0.003839
grad_step = 000082, loss = 0.003556
grad_step = 000083, loss = 0.003300
grad_step = 000084, loss = 0.003116
grad_step = 000085, loss = 0.003003
grad_step = 000086, loss = 0.002923
grad_step = 000087, loss = 0.002836
grad_step = 000088, loss = 0.002745
grad_step = 000089, loss = 0.002657
grad_step = 000090, loss = 0.002586
grad_step = 000091, loss = 0.002540
grad_step = 000092, loss = 0.002515
grad_step = 000093, loss = 0.002496
grad_step = 000094, loss = 0.002468
grad_step = 000095, loss = 0.002431
grad_step = 000096, loss = 0.002392
grad_step = 000097, loss = 0.002355
grad_step = 000098, loss = 0.002323
grad_step = 000099, loss = 0.002298
grad_step = 000100, loss = 0.002274
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002251
grad_step = 000102, loss = 0.002229
grad_step = 000103, loss = 0.002205
grad_step = 000104, loss = 0.002180
grad_step = 000105, loss = 0.002152
grad_step = 000106, loss = 0.002124
grad_step = 000107, loss = 0.002096
grad_step = 000108, loss = 0.002069
grad_step = 000109, loss = 0.002045
grad_step = 000110, loss = 0.002026
grad_step = 000111, loss = 0.002009
grad_step = 000112, loss = 0.001994
grad_step = 000113, loss = 0.001982
grad_step = 000114, loss = 0.001971
grad_step = 000115, loss = 0.001961
grad_step = 000116, loss = 0.001952
grad_step = 000117, loss = 0.001947
grad_step = 000118, loss = 0.001947
grad_step = 000119, loss = 0.001960
grad_step = 000120, loss = 0.002002
grad_step = 000121, loss = 0.002098
grad_step = 000122, loss = 0.002245
grad_step = 000123, loss = 0.002367
grad_step = 000124, loss = 0.002223
grad_step = 000125, loss = 0.001959
grad_step = 000126, loss = 0.001901
grad_step = 000127, loss = 0.002070
grad_step = 000128, loss = 0.002134
grad_step = 000129, loss = 0.001950
grad_step = 000130, loss = 0.001856
grad_step = 000131, loss = 0.001972
grad_step = 000132, loss = 0.002024
grad_step = 000133, loss = 0.001914
grad_step = 000134, loss = 0.001834
grad_step = 000135, loss = 0.001896
grad_step = 000136, loss = 0.001942
grad_step = 000137, loss = 0.001878
grad_step = 000138, loss = 0.001816
grad_step = 000139, loss = 0.001840
grad_step = 000140, loss = 0.001874
grad_step = 000141, loss = 0.001844
grad_step = 000142, loss = 0.001796
grad_step = 000143, loss = 0.001796
grad_step = 000144, loss = 0.001827
grad_step = 000145, loss = 0.001820
grad_step = 000146, loss = 0.001776
grad_step = 000147, loss = 0.001767
grad_step = 000148, loss = 0.001788
grad_step = 000149, loss = 0.001791
grad_step = 000150, loss = 0.001766
grad_step = 000151, loss = 0.001742
grad_step = 000152, loss = 0.001750
grad_step = 000153, loss = 0.001762
grad_step = 000154, loss = 0.001753
grad_step = 000155, loss = 0.001732
grad_step = 000156, loss = 0.001722
grad_step = 000157, loss = 0.001724
grad_step = 000158, loss = 0.001729
grad_step = 000159, loss = 0.001725
grad_step = 000160, loss = 0.001713
grad_step = 000161, loss = 0.001700
grad_step = 000162, loss = 0.001697
grad_step = 000163, loss = 0.001697
grad_step = 000164, loss = 0.001695
grad_step = 000165, loss = 0.001692
grad_step = 000166, loss = 0.001684
grad_step = 000167, loss = 0.001676
grad_step = 000168, loss = 0.001670
grad_step = 000169, loss = 0.001668
grad_step = 000170, loss = 0.001667
grad_step = 000171, loss = 0.001668
grad_step = 000172, loss = 0.001669
grad_step = 000173, loss = 0.001672
grad_step = 000174, loss = 0.001677
grad_step = 000175, loss = 0.001685
grad_step = 000176, loss = 0.001699
grad_step = 000177, loss = 0.001722
grad_step = 000178, loss = 0.001747
grad_step = 000179, loss = 0.001778
grad_step = 000180, loss = 0.001806
grad_step = 000181, loss = 0.001846
grad_step = 000182, loss = 0.001889
grad_step = 000183, loss = 0.001930
grad_step = 000184, loss = 0.001877
grad_step = 000185, loss = 0.001762
grad_step = 000186, loss = 0.001641
grad_step = 000187, loss = 0.001622
grad_step = 000188, loss = 0.001706
grad_step = 000189, loss = 0.001778
grad_step = 000190, loss = 0.001753
grad_step = 000191, loss = 0.001663
grad_step = 000192, loss = 0.001614
grad_step = 000193, loss = 0.001634
grad_step = 000194, loss = 0.001669
grad_step = 000195, loss = 0.001662
grad_step = 000196, loss = 0.001604
grad_step = 000197, loss = 0.001578
grad_step = 000198, loss = 0.001603
grad_step = 000199, loss = 0.001640
grad_step = 000200, loss = 0.001652
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001629
grad_step = 000202, loss = 0.001610
grad_step = 000203, loss = 0.001624
grad_step = 000204, loss = 0.001661
grad_step = 000205, loss = 0.001695
grad_step = 000206, loss = 0.001704
grad_step = 000207, loss = 0.001699
grad_step = 000208, loss = 0.001681
grad_step = 000209, loss = 0.001667
grad_step = 000210, loss = 0.001643
grad_step = 000211, loss = 0.001601
grad_step = 000212, loss = 0.001563
grad_step = 000213, loss = 0.001544
grad_step = 000214, loss = 0.001555
grad_step = 000215, loss = 0.001584
grad_step = 000216, loss = 0.001610
grad_step = 000217, loss = 0.001619
grad_step = 000218, loss = 0.001602
grad_step = 000219, loss = 0.001572
grad_step = 000220, loss = 0.001542
grad_step = 000221, loss = 0.001528
grad_step = 000222, loss = 0.001531
grad_step = 000223, loss = 0.001543
grad_step = 000224, loss = 0.001556
grad_step = 000225, loss = 0.001560
grad_step = 000226, loss = 0.001556
grad_step = 000227, loss = 0.001543
grad_step = 000228, loss = 0.001529
grad_step = 000229, loss = 0.001516
grad_step = 000230, loss = 0.001508
grad_step = 000231, loss = 0.001506
grad_step = 000232, loss = 0.001512
grad_step = 000233, loss = 0.001526
grad_step = 000234, loss = 0.001551
grad_step = 000235, loss = 0.001598
grad_step = 000236, loss = 0.001675
grad_step = 000237, loss = 0.001804
grad_step = 000238, loss = 0.001957
grad_step = 000239, loss = 0.002178
grad_step = 000240, loss = 0.002300
grad_step = 000241, loss = 0.002300
grad_step = 000242, loss = 0.002020
grad_step = 000243, loss = 0.001756
grad_step = 000244, loss = 0.001756
grad_step = 000245, loss = 0.001860
grad_step = 000246, loss = 0.001848
grad_step = 000247, loss = 0.001681
grad_step = 000248, loss = 0.001627
grad_step = 000249, loss = 0.001718
grad_step = 000250, loss = 0.001731
grad_step = 000251, loss = 0.001612
grad_step = 000252, loss = 0.001561
grad_step = 000253, loss = 0.001648
grad_step = 000254, loss = 0.001674
grad_step = 000255, loss = 0.001558
grad_step = 000256, loss = 0.001509
grad_step = 000257, loss = 0.001600
grad_step = 000258, loss = 0.001620
grad_step = 000259, loss = 0.001527
grad_step = 000260, loss = 0.001479
grad_step = 000261, loss = 0.001552
grad_step = 000262, loss = 0.001582
grad_step = 000263, loss = 0.001510
grad_step = 000264, loss = 0.001468
grad_step = 000265, loss = 0.001512
grad_step = 000266, loss = 0.001542
grad_step = 000267, loss = 0.001499
grad_step = 000268, loss = 0.001464
grad_step = 000269, loss = 0.001482
grad_step = 000270, loss = 0.001508
grad_step = 000271, loss = 0.001488
grad_step = 000272, loss = 0.001463
grad_step = 000273, loss = 0.001465
grad_step = 000274, loss = 0.001480
grad_step = 000275, loss = 0.001475
grad_step = 000276, loss = 0.001458
grad_step = 000277, loss = 0.001456
grad_step = 000278, loss = 0.001464
grad_step = 000279, loss = 0.001464
grad_step = 000280, loss = 0.001451
grad_step = 000281, loss = 0.001445
grad_step = 000282, loss = 0.001449
grad_step = 000283, loss = 0.001455
grad_step = 000284, loss = 0.001451
grad_step = 000285, loss = 0.001440
grad_step = 000286, loss = 0.001435
grad_step = 000287, loss = 0.001438
grad_step = 000288, loss = 0.001441
grad_step = 000289, loss = 0.001439
grad_step = 000290, loss = 0.001434
grad_step = 000291, loss = 0.001430
grad_step = 000292, loss = 0.001431
grad_step = 000293, loss = 0.001431
grad_step = 000294, loss = 0.001429
grad_step = 000295, loss = 0.001424
grad_step = 000296, loss = 0.001421
grad_step = 000297, loss = 0.001420
grad_step = 000298, loss = 0.001420
grad_step = 000299, loss = 0.001420
grad_step = 000300, loss = 0.001417
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001415
grad_step = 000302, loss = 0.001413
grad_step = 000303, loss = 0.001413
grad_step = 000304, loss = 0.001413
grad_step = 000305, loss = 0.001414
grad_step = 000306, loss = 0.001417
grad_step = 000307, loss = 0.001421
grad_step = 000308, loss = 0.001433
grad_step = 000309, loss = 0.001457
grad_step = 000310, loss = 0.001508
grad_step = 000311, loss = 0.001590
grad_step = 000312, loss = 0.001748
grad_step = 000313, loss = 0.001883
grad_step = 000314, loss = 0.002020
grad_step = 000315, loss = 0.001836
grad_step = 000316, loss = 0.001568
grad_step = 000317, loss = 0.001406
grad_step = 000318, loss = 0.001503
grad_step = 000319, loss = 0.001654
grad_step = 000320, loss = 0.001583
grad_step = 000321, loss = 0.001432
grad_step = 000322, loss = 0.001422
grad_step = 000323, loss = 0.001525
grad_step = 000324, loss = 0.001539
grad_step = 000325, loss = 0.001427
grad_step = 000326, loss = 0.001384
grad_step = 000327, loss = 0.001453
grad_step = 000328, loss = 0.001489
grad_step = 000329, loss = 0.001437
grad_step = 000330, loss = 0.001377
grad_step = 000331, loss = 0.001397
grad_step = 000332, loss = 0.001442
grad_step = 000333, loss = 0.001428
grad_step = 000334, loss = 0.001381
grad_step = 000335, loss = 0.001369
grad_step = 000336, loss = 0.001398
grad_step = 000337, loss = 0.001412
grad_step = 000338, loss = 0.001387
grad_step = 000339, loss = 0.001361
grad_step = 000340, loss = 0.001366
grad_step = 000341, loss = 0.001383
grad_step = 000342, loss = 0.001383
grad_step = 000343, loss = 0.001362
grad_step = 000344, loss = 0.001350
grad_step = 000345, loss = 0.001356
grad_step = 000346, loss = 0.001366
grad_step = 000347, loss = 0.001364
grad_step = 000348, loss = 0.001351
grad_step = 000349, loss = 0.001342
grad_step = 000350, loss = 0.001344
grad_step = 000351, loss = 0.001349
grad_step = 000352, loss = 0.001350
grad_step = 000353, loss = 0.001344
grad_step = 000354, loss = 0.001337
grad_step = 000355, loss = 0.001334
grad_step = 000356, loss = 0.001337
grad_step = 000357, loss = 0.001340
grad_step = 000358, loss = 0.001343
grad_step = 000359, loss = 0.001343
grad_step = 000360, loss = 0.001347
grad_step = 000361, loss = 0.001356
grad_step = 000362, loss = 0.001380
grad_step = 000363, loss = 0.001421
grad_step = 000364, loss = 0.001495
grad_step = 000365, loss = 0.001598
grad_step = 000366, loss = 0.001768
grad_step = 000367, loss = 0.001908
grad_step = 000368, loss = 0.002028
grad_step = 000369, loss = 0.001865
grad_step = 000370, loss = 0.001578
grad_step = 000371, loss = 0.001338
grad_step = 000372, loss = 0.001346
grad_step = 000373, loss = 0.001519
grad_step = 000374, loss = 0.001609
grad_step = 000375, loss = 0.001519
grad_step = 000376, loss = 0.001352
grad_step = 000377, loss = 0.001317
grad_step = 000378, loss = 0.001410
grad_step = 000379, loss = 0.001466
grad_step = 000380, loss = 0.001408
grad_step = 000381, loss = 0.001315
grad_step = 000382, loss = 0.001312
grad_step = 000383, loss = 0.001374
grad_step = 000384, loss = 0.001392
grad_step = 000385, loss = 0.001340
grad_step = 000386, loss = 0.001288
grad_step = 000387, loss = 0.001298
grad_step = 000388, loss = 0.001339
grad_step = 000389, loss = 0.001345
grad_step = 000390, loss = 0.001310
grad_step = 000391, loss = 0.001280
grad_step = 000392, loss = 0.001287
grad_step = 000393, loss = 0.001310
grad_step = 000394, loss = 0.001312
grad_step = 000395, loss = 0.001290
grad_step = 000396, loss = 0.001271
grad_step = 000397, loss = 0.001273
grad_step = 000398, loss = 0.001286
grad_step = 000399, loss = 0.001290
grad_step = 000400, loss = 0.001279
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001265
grad_step = 000402, loss = 0.001262
grad_step = 000403, loss = 0.001268
grad_step = 000404, loss = 0.001275
grad_step = 000405, loss = 0.001276
grad_step = 000406, loss = 0.001273
grad_step = 000407, loss = 0.001279
grad_step = 000408, loss = 0.001302
grad_step = 000409, loss = 0.001358
grad_step = 000410, loss = 0.001452
grad_step = 000411, loss = 0.001641
grad_step = 000412, loss = 0.001832
grad_step = 000413, loss = 0.002113
grad_step = 000414, loss = 0.001953
grad_step = 000415, loss = 0.001644
grad_step = 000416, loss = 0.001332
grad_step = 000417, loss = 0.001404
grad_step = 000418, loss = 0.001595
grad_step = 000419, loss = 0.001478
grad_step = 000420, loss = 0.001310
grad_step = 000421, loss = 0.001377
grad_step = 000422, loss = 0.001483
grad_step = 000423, loss = 0.001370
grad_step = 000424, loss = 0.001246
grad_step = 000425, loss = 0.001337
grad_step = 000426, loss = 0.001414
grad_step = 000427, loss = 0.001311
grad_step = 000428, loss = 0.001249
grad_step = 000429, loss = 0.001312
grad_step = 000430, loss = 0.001324
grad_step = 000431, loss = 0.001260
grad_step = 000432, loss = 0.001252
grad_step = 000433, loss = 0.001297
grad_step = 000434, loss = 0.001285
grad_step = 000435, loss = 0.001231
grad_step = 000436, loss = 0.001236
grad_step = 000437, loss = 0.001274
grad_step = 000438, loss = 0.001261
grad_step = 000439, loss = 0.001228
grad_step = 000440, loss = 0.001229
grad_step = 000441, loss = 0.001246
grad_step = 000442, loss = 0.001237
grad_step = 000443, loss = 0.001219
grad_step = 000444, loss = 0.001224
grad_step = 000445, loss = 0.001235
grad_step = 000446, loss = 0.001227
grad_step = 000447, loss = 0.001210
grad_step = 000448, loss = 0.001210
grad_step = 000449, loss = 0.001219
grad_step = 000450, loss = 0.001219
grad_step = 000451, loss = 0.001209
grad_step = 000452, loss = 0.001204
grad_step = 000453, loss = 0.001207
grad_step = 000454, loss = 0.001209
grad_step = 000455, loss = 0.001204
grad_step = 000456, loss = 0.001198
grad_step = 000457, loss = 0.001197
grad_step = 000458, loss = 0.001201
grad_step = 000459, loss = 0.001201
grad_step = 000460, loss = 0.001196
grad_step = 000461, loss = 0.001192
grad_step = 000462, loss = 0.001191
grad_step = 000463, loss = 0.001192
grad_step = 000464, loss = 0.001191
grad_step = 000465, loss = 0.001188
grad_step = 000466, loss = 0.001185
grad_step = 000467, loss = 0.001184
grad_step = 000468, loss = 0.001185
grad_step = 000469, loss = 0.001184
grad_step = 000470, loss = 0.001183
grad_step = 000471, loss = 0.001180
grad_step = 000472, loss = 0.001178
grad_step = 000473, loss = 0.001177
grad_step = 000474, loss = 0.001177
grad_step = 000475, loss = 0.001176
grad_step = 000476, loss = 0.001174
grad_step = 000477, loss = 0.001172
grad_step = 000478, loss = 0.001171
grad_step = 000479, loss = 0.001170
grad_step = 000480, loss = 0.001169
grad_step = 000481, loss = 0.001168
grad_step = 000482, loss = 0.001167
grad_step = 000483, loss = 0.001165
grad_step = 000484, loss = 0.001164
grad_step = 000485, loss = 0.001163
grad_step = 000486, loss = 0.001162
grad_step = 000487, loss = 0.001161
grad_step = 000488, loss = 0.001160
grad_step = 000489, loss = 0.001159
grad_step = 000490, loss = 0.001158
grad_step = 000491, loss = 0.001157
grad_step = 000492, loss = 0.001157
grad_step = 000493, loss = 0.001158
grad_step = 000494, loss = 0.001161
grad_step = 000495, loss = 0.001167
grad_step = 000496, loss = 0.001180
grad_step = 000497, loss = 0.001205
grad_step = 000498, loss = 0.001257
grad_step = 000499, loss = 0.001350
grad_step = 000500, loss = 0.001541
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001823
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

  date_run                              2020-05-09 19:13:40.611992
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.268883
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 19:13:40.618708
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.193878
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 19:13:40.626861
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149736
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 19:13:40.632691
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.94604
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
0   2020-05-09 19:13:08.863494  ...    mean_absolute_error
1   2020-05-09 19:13:08.867223  ...     mean_squared_error
2   2020-05-09 19:13:08.870516  ...  median_absolute_error
3   2020-05-09 19:13:08.873841  ...               r2_score
4   2020-05-09 19:13:20.832425  ...    mean_absolute_error
5   2020-05-09 19:13:20.836162  ...     mean_squared_error
6   2020-05-09 19:13:20.839695  ...  median_absolute_error
7   2020-05-09 19:13:20.842825  ...               r2_score
8   2020-05-09 19:13:40.611992  ...    mean_absolute_error
9   2020-05-09 19:13:40.618708  ...     mean_squared_error
10  2020-05-09 19:13:40.626861  ...  median_absolute_error
11  2020-05-09 19:13:40.632691  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 315067.02it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 407264.33it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 563757.76it/s] 30%|███       | 3022848/9912422 [00:00<00:08, 794185.64it/s] 58%|█████▊    | 5734400/9912422 [00:00<00:03, 1117807.28it/s] 88%|████████▊ | 8699904/9912422 [00:00<00:00, 1565525.68it/s]9920512it [00:00, 9972111.68it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 147524.78it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:04, 325574.14it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 419905.31it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 580727.00it/s]1654784it [00:00, 2862371.58it/s]                           
0it [00:00, ?it/s]8192it [00:00, 86564.38it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6066d22780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6004466a90> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6066cd9e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6004466e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6066cd9e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6066d22e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6066d22780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f60196d6c88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6004466e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f60196d6c88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6066cd9e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f96ac489208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8b58452059628ad339b4d380037c83dba6a6b0e631aa0b6125f2f62dee6aac43
  Stored in directory: /tmp/pip-ephem-wheel-cache-vnqwrcfj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f96a430afd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 44s
   57344/17464789 [..............................] - ETA: 38s
   90112/17464789 [..............................] - ETA: 36s
  163840/17464789 [..............................] - ETA: 26s
  278528/17464789 [..............................] - ETA: 19s
  540672/17464789 [..............................] - ETA: 11s
 1064960/17464789 [>.............................] - ETA: 6s 
 2080768/17464789 [==>...........................] - ETA: 3s
 4096000/17464789 [======>.......................] - ETA: 1s
 6586368/17464789 [==========>...................] - ETA: 1s
 9125888/17464789 [==============>...............] - ETA: 0s
11288576/17464789 [==================>...........] - ETA: 0s
13811712/17464789 [======================>.......] - ETA: 0s
16334848/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 19:15:10.346950: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 19:15:10.351575: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-09 19:15:10.351704: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5577dc6c3820 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 19:15:10.351717: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7356 - accuracy: 0.4955 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6264 - accuracy: 0.5026
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6274 - accuracy: 0.5026
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
11000/25000 [============>.................] - ETA: 3s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 3s - loss: 7.6347 - accuracy: 0.5021
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6633 - accuracy: 0.5002
15000/25000 [=================>............] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6158 - accuracy: 0.5033
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6323 - accuracy: 0.5022
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6291 - accuracy: 0.5024
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6279 - accuracy: 0.5025
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6492 - accuracy: 0.5011
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 7s 286us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 19:15:24.254776
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 19:15:24.254776  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 19:15:30.546437: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 19:15:30.551831: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-09 19:15:30.551994: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fc97488bc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 19:15:30.552004: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f65ae396cf8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7069 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.6933 - val_crf_viterbi_accuracy: 0.3467

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f65cff57128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6935 - accuracy: 0.4983
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6881 - accuracy: 0.4986
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7765 - accuracy: 0.4928
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6992 - accuracy: 0.4979
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7177 - accuracy: 0.4967
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7034 - accuracy: 0.4976
11000/25000 [============>.................] - ETA: 3s - loss: 7.7266 - accuracy: 0.4961
12000/25000 [=============>................] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7032 - accuracy: 0.4976
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7115 - accuracy: 0.4971
15000/25000 [=================>............] - ETA: 2s - loss: 7.7341 - accuracy: 0.4956
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7251 - accuracy: 0.4962
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7054 - accuracy: 0.4975
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6905 - accuracy: 0.4984
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6789 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 7s 280us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f655f4c9208> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<13:08:54, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 360k/862M [00:00<9:13:14, 26.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.92M/862M [00:00<6:25:47, 37.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:00<4:27:35, 53.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.4M/862M [00:00<3:05:29, 75.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:00<2:08:57, 108kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.5M/862M [00:01<1:29:21, 154kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:01<1:02:12, 220kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:01<43:10, 314kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:01<30:38, 441kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<21:42, 619kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<11:04:10, 20.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:03<7:44:12, 28.9kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<5:26:59, 40.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<3:49:52, 58.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:42:10, 82.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<1:54:04, 117kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:21:36, 162kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:09<58:21, 227kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<42:35, 309kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:11<30:36, 430kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<23:20, 561kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<17:07, 764kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<13:56, 935kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<10:20, 1.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:17<09:18, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:17<07:12, 1.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<05:38, 2.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<6:21:01, 33.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:20<4:27:08, 48.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<3:08:07, 68.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:20<2:11:13, 97.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<1:49:47, 116kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<1:17:24, 165kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<55:58, 227kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:24<39:53, 319kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<29:45, 425kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<21:55, 577kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<17:07, 734kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<12:39, 993kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:48, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<07:54, 1.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:11, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:54, 2.53MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<6:08:57, 33.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<4:18:36, 47.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<3:01:55, 67.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<2:08:33, 95.4kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<1:30:40, 135kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<1:05:02, 187kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<46:11, 264kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<34:07, 355kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<24:40, 491kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<19:01, 634kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<13:54, 866kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<11:35, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<08:59, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:03, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<06:21, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:15, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<05:06, 2.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<04:05, 2.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<6:03:04, 32.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<4:14:21, 46.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<2:58:54, 65.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<2:06:20, 92.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<1:29:00, 131kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<1:03:50, 182kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<45:17, 256kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<33:20, 346kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<24:00, 480kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<18:29, 620kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<13:33, 844kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<11:15, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<08:31, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<07:41, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<06:20, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<06:05, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:19, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<04:10, 2.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<5:43:35, 32.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<4:00:38, 46.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<2:49:22, 65.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<1:59:32, 92.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<1:24:08, 131kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<1:00:20, 182kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<42:47, 257kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<31:32, 347kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<22:41, 481kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<17:31, 620kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<12:46, 850kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<10:37, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<08:01, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<07:17, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<05:58, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:45, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:02, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<04:00, 2.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<5:08:30, 34.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<3:36:03, 48.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<2:31:54, 69.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:26<1:45:48, 99.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<3:07:04, 56.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<2:11:28, 79.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<1:33:06, 112kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<1:06:07, 157kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<47:31, 218kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<34:13, 302kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<25:19, 406kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<18:16, 562kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<14:20, 711kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<10:30, 970kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<08:56, 1.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:38<07:14, 1.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:31, 1.54MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<05:24, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<04:13, 2.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<5:01:23, 33.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<3:30:57, 47.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<2:28:24, 67.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<1:44:43, 94.4kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<1:13:57, 134kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<52:55, 185kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<37:51, 259kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<27:47, 351kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<19:59, 487kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<15:24, 628kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<11:16, 858kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<09:21, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<07:04, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<06:24, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:23, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:08, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<03:17, 2.87MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<4:49:43, 32.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:58<3:21:49, 46.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<2:25:45, 64.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<1:42:43, 91.0kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<1:12:49, 128kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<51:32, 180kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<37:14, 248kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<26:56, 342kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<20:04, 456kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<14:31, 629kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<11:31, 788kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<08:28, 1.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<07:21, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<06:00, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:29, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:44, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:35, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:49, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<04:00, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:16, 2.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<02:40, 3.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<4:34:56, 31.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<3:12:11, 45.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<2:15:09, 64.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<1:35:14, 90.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<1:07:04, 128kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<47:59, 178kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<34:01, 251kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<25:00, 339kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<17:49, 475kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<13:47, 611kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<10:05, 833kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<08:21, 999kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<06:14, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:39, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:24, 1.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:53, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<03:03, 2.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<4:06:09, 33.2kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:34<2:51:01, 47.4kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<2:15:54, 59.6kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<1:35:46, 84.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<1:07:43, 119kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<47:50, 168kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<34:28, 231kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<24:54, 319kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<18:27, 428kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<13:20, 591kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<10:31, 744kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<07:46, 1.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<06:38, 1.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<05:14, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<04:47, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:49, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:05, 2.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:21, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:44, 2.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<02:14, 3.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<3:58:49, 31.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<2:46:42, 44.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<1:57:11, 63.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:55<1:21:26, 90.5kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<1:14:38, 98.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<52:30, 140kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<37:40, 194kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<26:38, 273kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<19:43, 367kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<14:05, 512kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<10:57, 653kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<07:56, 899kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<06:41, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<05:10, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:39, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:57, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:07<02:49, 2.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<05:46, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<04:26, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:10, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<03:18, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<02:36, 2.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<3:32:30, 32.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<2:28:13, 45.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<1:44:11, 64.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<1:13:17, 91.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<51:35, 130kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<36:50, 180kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<26:05, 254kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<19:08, 343kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<13:54, 472kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<10:36, 612kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<07:47, 832kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<06:24, 1.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:48, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:20, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:23, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:21, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:52, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:54, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:20, 2.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<01:57, 3.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<3:01:38, 34.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<2:06:36, 48.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<1:29:01, 68.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<1:02:35, 96.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<44:05, 137kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<31:29, 190kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<22:25, 266kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<16:25, 359kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<12:03, 489kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<09:12, 634kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<07:00, 832kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<04:56, 1.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<05:43, 1.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<04:30, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:57, 1.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:04, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:00, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:39, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<02:06, 2.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<2:39:21, 35.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<1:50:58, 49.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<1:18:07, 70.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:50<54:08, 101kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<50:55, 107kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<35:52, 151kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<25:42, 209kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<18:15, 294kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<13:28, 394kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<09:42, 546kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<07:33, 693kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<05:32, 942kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:39, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<03:32, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<03:15, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:40, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:36, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:14, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<01:46, 2.81MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<2:29:47, 33.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<1:44:07, 47.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<1:13:11, 67.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<51:19, 94.5kB/s]  .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<36:11, 134kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:09<25:03, 191kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<5:27:19, 14.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<3:48:51, 20.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<2:39:00, 29.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<1:51:24, 42.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<1:17:41, 59.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<54:30, 85.0kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<38:23, 119kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<26:59, 169kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<19:24, 232kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<13:54, 323kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<10:15, 432kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<07:27, 594kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<05:48, 751kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<04:17, 1.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<03:12, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<2:08:27, 33.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:29:06, 47.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:02:37, 67.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<43:49, 95.5kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<30:55, 135kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<21:56, 188kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<15:31, 265kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<11:21, 356kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<08:13, 491kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<06:16, 634kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<04:46, 832kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:51, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:04, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:40, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:12, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:04, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:38, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:43, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:33, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<01:13, 2.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<1:55:07, 31.8kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<1:19:35, 45.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<55:57, 64.1kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:45<38:30, 91.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<1:42:57, 34.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<1:12:10, 48.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<50:09, 68.9kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<35:23, 97.4kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<24:49, 136kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<17:29, 193kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<12:33, 264kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<08:58, 368kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<06:40, 487kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<04:53, 663kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<03:50, 828kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<02:50, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<02:26, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:51, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:22, 2.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:05, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:34:02, 31.9kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<1:04:44, 45.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<45:35, 64.1kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:04<31:18, 91.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:04<22:16, 128kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<3:00:30, 15.8kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<2:05:50, 22.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:26:51, 32.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:00:53, 45.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:08<41:44, 65.2kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<35:57, 75.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<25:11, 107kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<17:40, 150kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<12:26, 212kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<08:56, 289kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<06:24, 401kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<04:45, 527kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<03:25, 729kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:44, 891kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:02, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:46, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:27, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:19, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:06, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:05, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:52, 2.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:56, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<00:47, 2.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<00:38, 3.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<1:04:38, 32.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<44:04, 46.6kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<30:54, 66.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<21:17, 93.2kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<14:55, 132kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<10:26, 183kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<07:27, 256kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<05:19, 347kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<03:50, 479kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<02:52, 620kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<02:04, 854kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:40, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:15, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:06, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:53, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:50, 1.87MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:45, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:46, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:36, 2.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<00:29, 2.98MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<44:42, 32.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:46<29:50, 46.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<39:48, 34.9kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<27:50, 49.6kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<18:49, 70.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<13:09, 99.7kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<08:59, 139kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<06:17, 197kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<04:23, 269kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<03:06, 377kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<02:14, 497kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<01:36, 684kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<01:14, 846kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:57, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:58<00:38, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<04:23, 222kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<03:09, 308kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:00<02:04, 438kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<03:59, 227kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<02:49, 318kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:02<01:51, 452kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<04:41, 179kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<03:17, 251kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<02:14, 353kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<24:56, 31.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<16:03, 45.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<11:10, 64.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<07:15, 90.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:09<05:01, 128kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<03:17, 178kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<02:17, 251kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<01:31, 340kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<01:04, 473kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:44, 610kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:32, 826kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:22, 1.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:17, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:13, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:09, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:07, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:02, 3.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<04:03, 31.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<01:20, 45.3kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:49, 64.4kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 868/400000 [00:00<00:46, 8673.66it/s]  0%|          | 1734/400000 [00:00<00:45, 8667.23it/s]  1%|          | 2682/400000 [00:00<00:44, 8894.36it/s]  1%|          | 3565/400000 [00:00<00:44, 8871.63it/s]  1%|          | 4478/400000 [00:00<00:44, 8944.98it/s]  1%|▏         | 5421/400000 [00:00<00:43, 9084.53it/s]  2%|▏         | 6319/400000 [00:00<00:43, 9050.17it/s]  2%|▏         | 7264/400000 [00:00<00:42, 9165.49it/s]  2%|▏         | 8195/400000 [00:00<00:42, 9206.41it/s]  2%|▏         | 9082/400000 [00:01<00:44, 8811.13it/s]  2%|▏         | 9943/400000 [00:01<00:44, 8698.02it/s]  3%|▎         | 10828/400000 [00:01<00:44, 8741.26it/s]  3%|▎         | 11693/400000 [00:01<00:44, 8697.59it/s]  3%|▎         | 12562/400000 [00:01<00:44, 8691.34it/s]  3%|▎         | 13435/400000 [00:01<00:44, 8702.13it/s]  4%|▎         | 14306/400000 [00:01<00:44, 8702.47it/s]  4%|▍         | 15174/400000 [00:01<00:45, 8428.56it/s]  4%|▍         | 16029/400000 [00:01<00:45, 8463.42it/s]  4%|▍         | 16878/400000 [00:01<00:45, 8470.61it/s]  4%|▍         | 17758/400000 [00:02<00:44, 8566.40it/s]  5%|▍         | 18647/400000 [00:02<00:44, 8658.29it/s]  5%|▍         | 19531/400000 [00:02<00:43, 8709.89it/s]  5%|▌         | 20403/400000 [00:02<00:44, 8484.06it/s]  5%|▌         | 21276/400000 [00:02<00:44, 8554.24it/s]  6%|▌         | 22133/400000 [00:02<00:44, 8535.54it/s]  6%|▌         | 23007/400000 [00:02<00:43, 8592.08it/s]  6%|▌         | 23887/400000 [00:02<00:43, 8653.19it/s]  6%|▌         | 24811/400000 [00:02<00:42, 8819.55it/s]  6%|▋         | 25695/400000 [00:02<00:42, 8804.52it/s]  7%|▋         | 26587/400000 [00:03<00:42, 8838.82it/s]  7%|▋         | 27499/400000 [00:03<00:41, 8918.36it/s]  7%|▋         | 28394/400000 [00:03<00:41, 8925.42it/s]  7%|▋         | 29289/400000 [00:03<00:41, 8932.68it/s]  8%|▊         | 30200/400000 [00:03<00:41, 8984.82it/s]  8%|▊         | 31099/400000 [00:03<00:41, 8875.46it/s]  8%|▊         | 31988/400000 [00:03<00:42, 8666.22it/s]  8%|▊         | 32857/400000 [00:03<00:43, 8487.93it/s]  8%|▊         | 33740/400000 [00:03<00:42, 8586.56it/s]  9%|▊         | 34617/400000 [00:03<00:42, 8640.21it/s]  9%|▉         | 35483/400000 [00:04<00:42, 8621.07it/s]  9%|▉         | 36346/400000 [00:04<00:42, 8605.51it/s]  9%|▉         | 37222/400000 [00:04<00:41, 8648.97it/s] 10%|▉         | 38144/400000 [00:04<00:41, 8811.89it/s] 10%|▉         | 39027/400000 [00:04<00:41, 8764.24it/s] 10%|▉         | 39923/400000 [00:04<00:40, 8819.58it/s] 10%|█         | 40806/400000 [00:04<00:41, 8754.79it/s] 10%|█         | 41683/400000 [00:04<00:41, 8563.98it/s] 11%|█         | 42541/400000 [00:04<00:43, 8151.02it/s] 11%|█         | 43401/400000 [00:04<00:43, 8279.26it/s] 11%|█         | 44254/400000 [00:05<00:42, 8352.63it/s] 11%|█▏        | 45093/400000 [00:05<00:42, 8269.80it/s] 11%|█▏        | 45923/400000 [00:05<00:43, 8168.57it/s] 12%|█▏        | 46742/400000 [00:05<00:43, 8145.57it/s] 12%|█▏        | 47607/400000 [00:05<00:42, 8289.19it/s] 12%|█▏        | 48447/400000 [00:05<00:42, 8319.55it/s] 12%|█▏        | 49281/400000 [00:05<00:42, 8254.10it/s] 13%|█▎        | 50138/400000 [00:05<00:41, 8345.89it/s] 13%|█▎        | 51011/400000 [00:05<00:41, 8454.36it/s] 13%|█▎        | 51861/400000 [00:06<00:41, 8461.32it/s] 13%|█▎        | 52708/400000 [00:06<00:41, 8356.26it/s] 13%|█▎        | 53570/400000 [00:06<00:41, 8431.15it/s] 14%|█▎        | 54446/400000 [00:06<00:40, 8525.78it/s] 14%|█▍        | 55320/400000 [00:06<00:40, 8587.09it/s] 14%|█▍        | 56181/400000 [00:06<00:40, 8590.97it/s] 14%|█▍        | 57052/400000 [00:06<00:39, 8623.60it/s] 14%|█▍        | 57937/400000 [00:06<00:39, 8688.31it/s] 15%|█▍        | 58807/400000 [00:06<00:39, 8672.81it/s] 15%|█▍        | 59675/400000 [00:06<00:39, 8612.41it/s] 15%|█▌        | 60590/400000 [00:07<00:38, 8764.26it/s] 15%|█▌        | 61471/400000 [00:07<00:38, 8776.94it/s] 16%|█▌        | 62350/400000 [00:07<00:39, 8607.49it/s] 16%|█▌        | 63212/400000 [00:07<00:39, 8584.36it/s] 16%|█▌        | 64083/400000 [00:07<00:38, 8621.35it/s] 16%|█▌        | 64946/400000 [00:07<00:39, 8562.57it/s] 16%|█▋        | 65803/400000 [00:07<00:39, 8506.71it/s] 17%|█▋        | 66655/400000 [00:07<00:41, 8074.31it/s] 17%|█▋        | 67468/400000 [00:07<00:41, 8081.59it/s] 17%|█▋        | 68280/400000 [00:07<00:41, 8049.85it/s] 17%|█▋        | 69133/400000 [00:08<00:40, 8185.70it/s] 17%|█▋        | 69992/400000 [00:08<00:39, 8300.43it/s] 18%|█▊        | 70894/400000 [00:08<00:38, 8502.40it/s] 18%|█▊        | 71778/400000 [00:08<00:38, 8600.16it/s] 18%|█▊        | 72689/400000 [00:08<00:37, 8746.39it/s] 18%|█▊        | 73605/400000 [00:08<00:36, 8865.02it/s] 19%|█▊        | 74499/400000 [00:08<00:36, 8884.70it/s] 19%|█▉        | 75389/400000 [00:08<00:37, 8611.37it/s] 19%|█▉        | 76256/400000 [00:08<00:37, 8628.84it/s] 19%|█▉        | 77147/400000 [00:08<00:37, 8710.51it/s] 20%|█▉        | 78020/400000 [00:09<00:37, 8680.81it/s] 20%|█▉        | 78890/400000 [00:09<00:37, 8657.68it/s] 20%|█▉        | 79764/400000 [00:09<00:36, 8681.29it/s] 20%|██        | 80664/400000 [00:09<00:36, 8771.63it/s] 20%|██        | 81542/400000 [00:09<00:37, 8499.59it/s] 21%|██        | 82395/400000 [00:09<00:37, 8369.63it/s] 21%|██        | 83251/400000 [00:09<00:37, 8423.32it/s] 21%|██        | 84148/400000 [00:09<00:36, 8579.85it/s] 21%|██▏       | 85024/400000 [00:09<00:36, 8632.01it/s] 21%|██▏       | 85934/400000 [00:09<00:35, 8765.46it/s] 22%|██▏       | 86820/400000 [00:10<00:35, 8793.05it/s] 22%|██▏       | 87701/400000 [00:10<00:36, 8554.90it/s] 22%|██▏       | 88559/400000 [00:10<00:37, 8235.52it/s] 22%|██▏       | 89412/400000 [00:10<00:37, 8320.10it/s] 23%|██▎       | 90264/400000 [00:10<00:36, 8378.32it/s] 23%|██▎       | 91105/400000 [00:10<00:37, 8299.15it/s] 23%|██▎       | 91961/400000 [00:10<00:36, 8371.76it/s] 23%|██▎       | 92812/400000 [00:10<00:36, 8412.55it/s] 23%|██▎       | 93670/400000 [00:10<00:36, 8461.14it/s] 24%|██▎       | 94517/400000 [00:10<00:36, 8445.28it/s] 24%|██▍       | 95387/400000 [00:11<00:35, 8518.62it/s] 24%|██▍       | 96240/400000 [00:11<00:36, 8436.49it/s] 24%|██▍       | 97085/400000 [00:11<00:36, 8218.44it/s] 24%|██▍       | 97909/400000 [00:11<00:37, 8146.79it/s] 25%|██▍       | 98725/400000 [00:11<00:37, 8001.30it/s] 25%|██▍       | 99542/400000 [00:11<00:37, 8049.23it/s] 25%|██▌       | 100349/400000 [00:11<00:38, 7764.29it/s] 25%|██▌       | 101194/400000 [00:11<00:37, 7957.85it/s] 26%|██▌       | 102038/400000 [00:11<00:36, 8096.42it/s] 26%|██▌       | 102908/400000 [00:12<00:35, 8267.44it/s] 26%|██▌       | 103753/400000 [00:12<00:35, 8320.90it/s] 26%|██▌       | 104593/400000 [00:12<00:35, 8344.42it/s] 26%|██▋       | 105429/400000 [00:12<00:35, 8238.80it/s] 27%|██▋       | 106255/400000 [00:12<00:36, 8055.81it/s] 27%|██▋       | 107090/400000 [00:12<00:35, 8139.78it/s] 27%|██▋       | 107906/400000 [00:12<00:36, 8086.25it/s] 27%|██▋       | 108745/400000 [00:12<00:35, 8172.67it/s] 27%|██▋       | 109564/400000 [00:12<00:36, 7933.32it/s] 28%|██▊       | 110384/400000 [00:12<00:36, 8009.74it/s] 28%|██▊       | 111211/400000 [00:13<00:35, 8083.68it/s] 28%|██▊       | 112042/400000 [00:13<00:35, 8147.55it/s] 28%|██▊       | 112927/400000 [00:13<00:34, 8344.68it/s] 28%|██▊       | 113823/400000 [00:13<00:33, 8518.85it/s] 29%|██▊       | 114758/400000 [00:13<00:32, 8752.24it/s] 29%|██▉       | 115656/400000 [00:13<00:32, 8817.20it/s] 29%|██▉       | 116569/400000 [00:13<00:31, 8907.61it/s] 29%|██▉       | 117462/400000 [00:13<00:32, 8817.46it/s] 30%|██▉       | 118351/400000 [00:13<00:31, 8838.15it/s] 30%|██▉       | 119263/400000 [00:13<00:31, 8920.46it/s] 30%|███       | 120156/400000 [00:14<00:31, 8863.63it/s] 30%|███       | 121044/400000 [00:14<00:31, 8755.42it/s] 30%|███       | 121921/400000 [00:14<00:31, 8733.02it/s] 31%|███       | 122795/400000 [00:14<00:31, 8707.88it/s] 31%|███       | 123692/400000 [00:14<00:31, 8782.43it/s] 31%|███       | 124575/400000 [00:14<00:31, 8794.05it/s] 31%|███▏      | 125455/400000 [00:14<00:31, 8656.61it/s] 32%|███▏      | 126322/400000 [00:14<00:31, 8641.37it/s] 32%|███▏      | 127210/400000 [00:14<00:31, 8711.40it/s] 32%|███▏      | 128108/400000 [00:14<00:30, 8789.57it/s] 32%|███▏      | 128989/400000 [00:15<00:30, 8793.76it/s] 32%|███▏      | 129869/400000 [00:15<00:30, 8743.82it/s] 33%|███▎      | 130744/400000 [00:15<00:30, 8717.69it/s] 33%|███▎      | 131617/400000 [00:15<00:30, 8678.15it/s] 33%|███▎      | 132486/400000 [00:15<00:31, 8561.20it/s] 33%|███▎      | 133361/400000 [00:15<00:30, 8615.83it/s] 34%|███▎      | 134228/400000 [00:15<00:30, 8631.15it/s] 34%|███▍      | 135110/400000 [00:15<00:30, 8686.17it/s] 34%|███▍      | 135988/400000 [00:15<00:30, 8711.56it/s] 34%|███▍      | 136876/400000 [00:15<00:30, 8760.57it/s] 34%|███▍      | 137753/400000 [00:16<00:30, 8548.61it/s] 35%|███▍      | 138666/400000 [00:16<00:29, 8714.08it/s] 35%|███▍      | 139540/400000 [00:16<00:29, 8691.51it/s] 35%|███▌      | 140411/400000 [00:16<00:29, 8691.57it/s] 35%|███▌      | 141300/400000 [00:16<00:29, 8749.72it/s] 36%|███▌      | 142192/400000 [00:16<00:29, 8798.43it/s] 36%|███▌      | 143073/400000 [00:16<00:29, 8610.16it/s] 36%|███▌      | 143967/400000 [00:16<00:29, 8705.23it/s] 36%|███▌      | 144839/400000 [00:16<00:29, 8696.86it/s] 36%|███▋      | 145711/400000 [00:17<00:29, 8701.99it/s] 37%|███▋      | 146595/400000 [00:17<00:28, 8742.26it/s] 37%|███▋      | 147470/400000 [00:17<00:29, 8625.12it/s] 37%|███▋      | 148334/400000 [00:17<00:29, 8601.00it/s] 37%|███▋      | 149204/400000 [00:17<00:29, 8628.18it/s] 38%|███▊      | 150072/400000 [00:17<00:28, 8641.91it/s] 38%|███▊      | 150942/400000 [00:17<00:28, 8658.25it/s] 38%|███▊      | 151810/400000 [00:17<00:28, 8664.39it/s] 38%|███▊      | 152680/400000 [00:17<00:28, 8674.04it/s] 38%|███▊      | 153557/400000 [00:17<00:28, 8699.80it/s] 39%|███▊      | 154428/400000 [00:18<00:28, 8501.39it/s] 39%|███▉      | 155280/400000 [00:18<00:28, 8449.34it/s] 39%|███▉      | 156126/400000 [00:18<00:29, 8396.29it/s] 39%|███▉      | 156993/400000 [00:18<00:28, 8475.47it/s] 39%|███▉      | 157862/400000 [00:18<00:28, 8536.04it/s] 40%|███▉      | 158725/400000 [00:18<00:28, 8562.16it/s] 40%|███▉      | 159582/400000 [00:18<00:28, 8561.79it/s] 40%|████      | 160445/400000 [00:18<00:27, 8581.01it/s] 40%|████      | 161304/400000 [00:18<00:28, 8412.82it/s] 41%|████      | 162180/400000 [00:18<00:27, 8511.72it/s] 41%|████      | 163075/400000 [00:19<00:27, 8637.30it/s] 41%|████      | 163983/400000 [00:19<00:26, 8765.14it/s] 41%|████      | 164869/400000 [00:19<00:26, 8792.00it/s] 41%|████▏     | 165756/400000 [00:19<00:26, 8812.80it/s] 42%|████▏     | 166643/400000 [00:19<00:26, 8802.77it/s] 42%|████▏     | 167524/400000 [00:19<00:26, 8743.56it/s] 42%|████▏     | 168399/400000 [00:19<00:26, 8743.30it/s] 42%|████▏     | 169274/400000 [00:19<00:26, 8702.76it/s] 43%|████▎     | 170145/400000 [00:19<00:26, 8671.74it/s] 43%|████▎     | 171013/400000 [00:19<00:26, 8626.63it/s] 43%|████▎     | 171876/400000 [00:20<00:26, 8590.68it/s] 43%|████▎     | 172736/400000 [00:20<00:26, 8545.34it/s] 43%|████▎     | 173591/400000 [00:20<00:26, 8530.15it/s] 44%|████▎     | 174458/400000 [00:20<00:26, 8570.21it/s] 44%|████▍     | 175347/400000 [00:20<00:25, 8661.59it/s] 44%|████▍     | 176218/400000 [00:20<00:25, 8675.37it/s] 44%|████▍     | 177115/400000 [00:20<00:25, 8761.70it/s] 45%|████▍     | 178022/400000 [00:20<00:25, 8850.93it/s] 45%|████▍     | 178908/400000 [00:20<00:24, 8847.31it/s] 45%|████▍     | 179794/400000 [00:20<00:24, 8848.13it/s] 45%|████▌     | 180699/400000 [00:21<00:24, 8905.68it/s] 45%|████▌     | 181607/400000 [00:21<00:24, 8834.41it/s] 46%|████▌     | 182509/400000 [00:21<00:24, 8889.07it/s] 46%|████▌     | 183431/400000 [00:21<00:24, 8985.14it/s] 46%|████▌     | 184330/400000 [00:21<00:24, 8797.33it/s] 46%|████▋     | 185211/400000 [00:21<00:24, 8622.17it/s] 47%|████▋     | 186075/400000 [00:21<00:25, 8486.05it/s] 47%|████▋     | 186926/400000 [00:21<00:25, 8404.33it/s] 47%|████▋     | 187796/400000 [00:21<00:24, 8489.12it/s] 47%|████▋     | 188676/400000 [00:21<00:24, 8577.67it/s] 47%|████▋     | 189535/400000 [00:22<00:25, 8354.38it/s] 48%|████▊     | 190373/400000 [00:22<00:25, 8284.17it/s] 48%|████▊     | 191203/400000 [00:22<00:25, 8223.88it/s] 48%|████▊     | 192039/400000 [00:22<00:25, 8263.32it/s] 48%|████▊     | 192956/400000 [00:22<00:24, 8515.18it/s] 48%|████▊     | 193861/400000 [00:22<00:23, 8668.47it/s] 49%|████▊     | 194731/400000 [00:22<00:23, 8579.65it/s] 49%|████▉     | 195645/400000 [00:22<00:23, 8738.66it/s] 49%|████▉     | 196526/400000 [00:22<00:23, 8759.02it/s] 49%|████▉     | 197473/400000 [00:22<00:22, 8960.14it/s] 50%|████▉     | 198392/400000 [00:23<00:22, 9027.78it/s] 50%|████▉     | 199297/400000 [00:23<00:22, 8951.38it/s] 50%|█████     | 200194/400000 [00:23<00:22, 8817.94it/s] 50%|█████     | 201087/400000 [00:23<00:22, 8850.32it/s] 50%|█████     | 201974/400000 [00:23<00:22, 8754.71it/s] 51%|█████     | 202851/400000 [00:23<00:22, 8716.76it/s] 51%|█████     | 203731/400000 [00:23<00:22, 8739.00it/s] 51%|█████     | 204606/400000 [00:23<00:22, 8723.79it/s] 51%|█████▏    | 205482/400000 [00:23<00:22, 8732.73it/s] 52%|█████▏    | 206356/400000 [00:24<00:22, 8638.96it/s] 52%|█████▏    | 207221/400000 [00:24<00:22, 8520.25it/s] 52%|█████▏    | 208093/400000 [00:24<00:22, 8576.68it/s] 52%|█████▏    | 208966/400000 [00:24<00:22, 8620.03it/s] 52%|█████▏    | 209839/400000 [00:24<00:21, 8650.60it/s] 53%|█████▎    | 210716/400000 [00:24<00:21, 8685.24it/s] 53%|█████▎    | 211585/400000 [00:24<00:21, 8641.61it/s] 53%|█████▎    | 212489/400000 [00:24<00:21, 8756.43it/s] 53%|█████▎    | 213366/400000 [00:24<00:21, 8691.09it/s] 54%|█████▎    | 214236/400000 [00:24<00:21, 8663.70it/s] 54%|█████▍    | 215126/400000 [00:25<00:21, 8732.51it/s] 54%|█████▍    | 216027/400000 [00:25<00:20, 8813.56it/s] 54%|█████▍    | 216929/400000 [00:25<00:20, 8872.04it/s] 54%|█████▍    | 217817/400000 [00:25<00:21, 8636.94it/s] 55%|█████▍    | 218698/400000 [00:25<00:20, 8687.20it/s] 55%|█████▍    | 219574/400000 [00:25<00:20, 8706.48it/s] 55%|█████▌    | 220460/400000 [00:25<00:20, 8750.93it/s] 55%|█████▌    | 221336/400000 [00:25<00:20, 8743.81it/s] 56%|█████▌    | 222211/400000 [00:25<00:20, 8617.71it/s] 56%|█████▌    | 223074/400000 [00:25<00:21, 8208.66it/s] 56%|█████▌    | 223947/400000 [00:26<00:21, 8356.54it/s] 56%|█████▌    | 224797/400000 [00:26<00:20, 8376.53it/s] 56%|█████▋    | 225638/400000 [00:26<00:20, 8339.96it/s] 57%|█████▋    | 226508/400000 [00:26<00:20, 8443.72it/s] 57%|█████▋    | 227376/400000 [00:26<00:20, 8511.62it/s] 57%|█████▋    | 228229/400000 [00:26<00:20, 8431.56it/s] 57%|█████▋    | 229074/400000 [00:26<00:20, 8360.89it/s] 57%|█████▋    | 229927/400000 [00:26<00:20, 8408.36it/s] 58%|█████▊    | 230813/400000 [00:26<00:19, 8538.13it/s] 58%|█████▊    | 231687/400000 [00:26<00:19, 8596.71it/s] 58%|█████▊    | 232556/400000 [00:27<00:19, 8621.72it/s] 58%|█████▊    | 233433/400000 [00:27<00:19, 8664.01it/s] 59%|█████▊    | 234300/400000 [00:27<00:19, 8658.91it/s] 59%|█████▉    | 235167/400000 [00:27<00:19, 8649.28it/s] 59%|█████▉    | 236037/400000 [00:27<00:18, 8663.46it/s] 59%|█████▉    | 236952/400000 [00:27<00:18, 8801.94it/s] 59%|█████▉    | 237833/400000 [00:27<00:19, 8424.04it/s] 60%|█████▉    | 238680/400000 [00:27<00:19, 8389.41it/s] 60%|█████▉    | 239522/400000 [00:27<00:19, 8367.61it/s] 60%|██████    | 240361/400000 [00:27<00:19, 8318.92it/s] 60%|██████    | 241195/400000 [00:28<00:19, 8082.37it/s] 61%|██████    | 242060/400000 [00:28<00:19, 8244.23it/s] 61%|██████    | 242950/400000 [00:28<00:18, 8429.38it/s] 61%|██████    | 243811/400000 [00:28<00:18, 8481.31it/s] 61%|██████    | 244699/400000 [00:28<00:18, 8594.89it/s] 61%|██████▏   | 245586/400000 [00:28<00:17, 8674.30it/s] 62%|██████▏   | 246455/400000 [00:28<00:17, 8656.75it/s] 62%|██████▏   | 247343/400000 [00:28<00:17, 8721.01it/s] 62%|██████▏   | 248228/400000 [00:28<00:17, 8756.61it/s] 62%|██████▏   | 249105/400000 [00:28<00:17, 8718.46it/s] 63%|██████▎   | 250016/400000 [00:29<00:16, 8829.62it/s] 63%|██████▎   | 250900/400000 [00:29<00:17, 8735.42it/s] 63%|██████▎   | 251793/400000 [00:29<00:16, 8792.43it/s] 63%|██████▎   | 252703/400000 [00:29<00:16, 8879.86it/s] 63%|██████▎   | 253631/400000 [00:29<00:16, 8993.42it/s] 64%|██████▎   | 254538/400000 [00:29<00:16, 9013.69it/s] 64%|██████▍   | 255440/400000 [00:29<00:16, 8876.09it/s] 64%|██████▍   | 256329/400000 [00:29<00:16, 8780.29it/s] 64%|██████▍   | 257208/400000 [00:29<00:16, 8584.23it/s] 65%|██████▍   | 258068/400000 [00:30<00:16, 8504.82it/s] 65%|██████▍   | 258932/400000 [00:30<00:16, 8542.34it/s] 65%|██████▍   | 259801/400000 [00:30<00:16, 8583.21it/s] 65%|██████▌   | 260681/400000 [00:30<00:16, 8646.78it/s] 65%|██████▌   | 261547/400000 [00:30<00:16, 8586.50it/s] 66%|██████▌   | 262407/400000 [00:30<00:16, 8348.84it/s] 66%|██████▌   | 263268/400000 [00:30<00:16, 8425.20it/s] 66%|██████▌   | 264112/400000 [00:30<00:16, 8380.08it/s] 66%|██████▌   | 264965/400000 [00:30<00:16, 8423.36it/s] 66%|██████▋   | 265816/400000 [00:30<00:15, 8447.87it/s] 67%|██████▋   | 266717/400000 [00:31<00:15, 8606.57it/s] 67%|██████▋   | 267579/400000 [00:31<00:15, 8342.49it/s] 67%|██████▋   | 268446/400000 [00:31<00:15, 8437.21it/s] 67%|██████▋   | 269292/400000 [00:31<00:15, 8351.23it/s] 68%|██████▊   | 270146/400000 [00:31<00:15, 8406.48it/s] 68%|██████▊   | 271038/400000 [00:31<00:15, 8551.97it/s] 68%|██████▊   | 271897/400000 [00:31<00:14, 8560.58it/s] 68%|██████▊   | 272755/400000 [00:31<00:14, 8550.56it/s] 68%|██████▊   | 273611/400000 [00:31<00:14, 8551.84it/s] 69%|██████▊   | 274486/400000 [00:31<00:14, 8607.77it/s] 69%|██████▉   | 275348/400000 [00:32<00:14, 8568.72it/s] 69%|██████▉   | 276206/400000 [00:32<00:14, 8495.59it/s] 69%|██████▉   | 277062/400000 [00:32<00:14, 8513.29it/s] 69%|██████▉   | 277914/400000 [00:32<00:14, 8358.58it/s] 70%|██████▉   | 278779/400000 [00:32<00:14, 8443.08it/s] 70%|██████▉   | 279625/400000 [00:32<00:14, 8290.79it/s] 70%|███████   | 280456/400000 [00:32<00:14, 8134.01it/s] 70%|███████   | 281310/400000 [00:32<00:14, 8250.79it/s] 71%|███████   | 282165/400000 [00:32<00:14, 8337.92it/s] 71%|███████   | 283037/400000 [00:32<00:13, 8447.02it/s] 71%|███████   | 283883/400000 [00:33<00:13, 8332.06it/s] 71%|███████   | 284718/400000 [00:33<00:13, 8282.23it/s] 71%|███████▏  | 285563/400000 [00:33<00:13, 8330.30it/s] 72%|███████▏  | 286427/400000 [00:33<00:13, 8419.19it/s] 72%|███████▏  | 287288/400000 [00:33<00:13, 8472.81it/s] 72%|███████▏  | 288166/400000 [00:33<00:13, 8560.02it/s] 72%|███████▏  | 289032/400000 [00:33<00:12, 8589.70it/s] 72%|███████▏  | 289892/400000 [00:33<00:12, 8562.00it/s] 73%|███████▎  | 290764/400000 [00:33<00:12, 8608.59it/s] 73%|███████▎  | 291660/400000 [00:33<00:12, 8710.27it/s] 73%|███████▎  | 292554/400000 [00:34<00:12, 8775.79it/s] 73%|███████▎  | 293440/400000 [00:34<00:12, 8799.98it/s] 74%|███████▎  | 294321/400000 [00:34<00:12, 8724.92it/s] 74%|███████▍  | 295194/400000 [00:34<00:12, 8712.48it/s] 74%|███████▍  | 296083/400000 [00:34<00:11, 8763.60it/s] 74%|███████▍  | 296960/400000 [00:34<00:11, 8761.39it/s] 74%|███████▍  | 297837/400000 [00:34<00:11, 8740.43it/s] 75%|███████▍  | 298719/400000 [00:34<00:11, 8761.85it/s] 75%|███████▍  | 299605/400000 [00:34<00:11, 8788.57it/s] 75%|███████▌  | 300484/400000 [00:34<00:11, 8641.66it/s] 75%|███████▌  | 301349/400000 [00:35<00:11, 8363.99it/s] 76%|███████▌  | 302188/400000 [00:35<00:11, 8260.33it/s] 76%|███████▌  | 303034/400000 [00:35<00:11, 8317.06it/s] 76%|███████▌  | 303868/400000 [00:35<00:11, 8289.19it/s] 76%|███████▌  | 304698/400000 [00:35<00:11, 8232.43it/s] 76%|███████▋  | 305523/400000 [00:35<00:11, 8209.74it/s] 77%|███████▋  | 306345/400000 [00:35<00:11, 8192.55it/s] 77%|███████▋  | 307165/400000 [00:35<00:11, 7970.44it/s] 77%|███████▋  | 308037/400000 [00:35<00:11, 8181.23it/s] 77%|███████▋  | 308909/400000 [00:36<00:10, 8333.64it/s] 77%|███████▋  | 309786/400000 [00:36<00:10, 8458.33it/s] 78%|███████▊  | 310654/400000 [00:36<00:10, 8522.04it/s] 78%|███████▊  | 311508/400000 [00:36<00:10, 8473.34it/s] 78%|███████▊  | 312376/400000 [00:36<00:10, 8533.54it/s] 78%|███████▊  | 313231/400000 [00:36<00:10, 8508.38it/s] 79%|███████▊  | 314083/400000 [00:36<00:10, 8504.08it/s] 79%|███████▊  | 314941/400000 [00:36<00:09, 8523.49it/s] 79%|███████▉  | 315794/400000 [00:36<00:09, 8505.21it/s] 79%|███████▉  | 316662/400000 [00:36<00:09, 8556.31it/s] 79%|███████▉  | 317541/400000 [00:37<00:09, 8623.68it/s] 80%|███████▉  | 318446/400000 [00:37<00:09, 8744.84it/s] 80%|███████▉  | 319338/400000 [00:37<00:09, 8793.70it/s] 80%|████████  | 320241/400000 [00:37<00:09, 8860.72it/s] 80%|████████  | 321143/400000 [00:37<00:08, 8907.16it/s] 81%|████████  | 322035/400000 [00:37<00:08, 8890.99it/s] 81%|████████  | 322925/400000 [00:37<00:08, 8835.75it/s] 81%|████████  | 323809/400000 [00:37<00:08, 8715.31it/s] 81%|████████  | 324682/400000 [00:37<00:08, 8643.03it/s] 81%|████████▏ | 325547/400000 [00:37<00:08, 8462.92it/s] 82%|████████▏ | 326395/400000 [00:38<00:08, 8345.44it/s] 82%|████████▏ | 327231/400000 [00:38<00:08, 8166.77it/s] 82%|████████▏ | 328079/400000 [00:38<00:08, 8255.99it/s] 82%|████████▏ | 328934/400000 [00:38<00:08, 8340.89it/s] 82%|████████▏ | 329784/400000 [00:38<00:08, 8385.96it/s] 83%|████████▎ | 330624/400000 [00:38<00:08, 8306.39it/s] 83%|████████▎ | 331456/400000 [00:38<00:08, 8148.90it/s] 83%|████████▎ | 332298/400000 [00:38<00:08, 8227.86it/s] 83%|████████▎ | 333157/400000 [00:38<00:08, 8332.03it/s] 84%|████████▎ | 334022/400000 [00:38<00:07, 8423.18it/s] 84%|████████▎ | 334891/400000 [00:39<00:07, 8499.13it/s] 84%|████████▍ | 335775/400000 [00:39<00:07, 8596.73it/s] 84%|████████▍ | 336648/400000 [00:39<00:07, 8634.39it/s] 84%|████████▍ | 337519/400000 [00:39<00:07, 8654.92it/s] 85%|████████▍ | 338413/400000 [00:39<00:07, 8735.98it/s] 85%|████████▍ | 339288/400000 [00:39<00:07, 8576.85it/s] 85%|████████▌ | 340147/400000 [00:39<00:07, 8461.97it/s] 85%|████████▌ | 340995/400000 [00:39<00:07, 8199.82it/s] 85%|████████▌ | 341844/400000 [00:39<00:07, 8284.18it/s] 86%|████████▌ | 342728/400000 [00:39<00:06, 8441.28it/s] 86%|████████▌ | 343627/400000 [00:40<00:06, 8596.74it/s] 86%|████████▌ | 344520/400000 [00:40<00:06, 8692.34it/s] 86%|████████▋ | 345417/400000 [00:40<00:06, 8770.88it/s] 87%|████████▋ | 346296/400000 [00:40<00:06, 8678.47it/s] 87%|████████▋ | 347173/400000 [00:40<00:06, 8703.33it/s] 87%|████████▋ | 348045/400000 [00:40<00:05, 8695.82it/s] 87%|████████▋ | 348935/400000 [00:40<00:05, 8753.86it/s] 87%|████████▋ | 349811/400000 [00:40<00:05, 8744.38it/s] 88%|████████▊ | 350688/400000 [00:40<00:05, 8750.86it/s] 88%|████████▊ | 351589/400000 [00:40<00:05, 8826.95it/s] 88%|████████▊ | 352508/400000 [00:41<00:05, 8929.90it/s] 88%|████████▊ | 353402/400000 [00:41<00:05, 8824.02it/s] 89%|████████▊ | 354286/400000 [00:41<00:05, 8764.98it/s] 89%|████████▉ | 355164/400000 [00:41<00:05, 8761.46it/s] 89%|████████▉ | 356041/400000 [00:41<00:05, 8661.75it/s] 89%|████████▉ | 356911/400000 [00:41<00:04, 8671.06it/s] 89%|████████▉ | 357779/400000 [00:41<00:05, 8381.65it/s] 90%|████████▉ | 358620/400000 [00:41<00:05, 8093.70it/s] 90%|████████▉ | 359459/400000 [00:41<00:04, 8177.69it/s] 90%|█████████ | 360341/400000 [00:42<00:04, 8359.59it/s] 90%|█████████ | 361237/400000 [00:42<00:04, 8528.61it/s] 91%|█████████ | 362124/400000 [00:42<00:04, 8627.81it/s] 91%|█████████ | 363003/400000 [00:42<00:04, 8673.75it/s] 91%|█████████ | 363875/400000 [00:42<00:04, 8686.30it/s] 91%|█████████ | 364745/400000 [00:42<00:04, 8582.11it/s] 91%|█████████▏| 365605/400000 [00:42<00:04, 8587.44it/s] 92%|█████████▏| 366503/400000 [00:42<00:03, 8701.21it/s] 92%|█████████▏| 367375/400000 [00:42<00:03, 8544.34it/s] 92%|█████████▏| 368236/400000 [00:42<00:03, 8562.37it/s] 92%|█████████▏| 369094/400000 [00:43<00:03, 8485.85it/s] 92%|█████████▏| 369944/400000 [00:43<00:03, 8432.17it/s] 93%|█████████▎| 370788/400000 [00:43<00:03, 8368.49it/s] 93%|█████████▎| 371626/400000 [00:43<00:03, 8302.37it/s] 93%|█████████▎| 372457/400000 [00:43<00:03, 8258.55it/s] 93%|█████████▎| 373318/400000 [00:43<00:03, 8358.87it/s] 94%|█████████▎| 374155/400000 [00:43<00:03, 8230.26it/s] 94%|█████████▎| 374979/400000 [00:43<00:03, 8183.48it/s] 94%|█████████▍| 375843/400000 [00:43<00:02, 8314.99it/s] 94%|█████████▍| 376706/400000 [00:43<00:02, 8406.77it/s] 94%|█████████▍| 377572/400000 [00:44<00:02, 8479.92it/s] 95%|█████████▍| 378421/400000 [00:44<00:02, 8399.61it/s] 95%|█████████▍| 379269/400000 [00:44<00:02, 8421.63it/s] 95%|█████████▌| 380139/400000 [00:44<00:02, 8500.67it/s] 95%|█████████▌| 381001/400000 [00:44<00:02, 8534.68it/s] 95%|█████████▌| 381857/400000 [00:44<00:02, 8541.05it/s] 96%|█████████▌| 382712/400000 [00:44<00:02, 8324.97it/s] 96%|█████████▌| 383579/400000 [00:44<00:01, 8422.82it/s] 96%|█████████▌| 384442/400000 [00:44<00:01, 8481.67it/s] 96%|█████████▋| 385321/400000 [00:44<00:01, 8571.52it/s] 97%|█████████▋| 386180/400000 [00:45<00:01, 8513.68it/s] 97%|█████████▋| 387047/400000 [00:45<00:01, 8557.66it/s] 97%|█████████▋| 387930/400000 [00:45<00:01, 8635.93it/s] 97%|█████████▋| 388808/400000 [00:45<00:01, 8677.12it/s] 97%|█████████▋| 389694/400000 [00:45<00:01, 8729.16it/s] 98%|█████████▊| 390576/400000 [00:45<00:01, 8754.00it/s] 98%|█████████▊| 391452/400000 [00:45<00:00, 8749.95it/s] 98%|█████████▊| 392328/400000 [00:45<00:00, 8648.75it/s] 98%|█████████▊| 393194/400000 [00:45<00:00, 8595.74it/s] 99%|█████████▊| 394080/400000 [00:45<00:00, 8670.58it/s] 99%|█████████▊| 394967/400000 [00:46<00:00, 8727.57it/s] 99%|█████████▉| 395841/400000 [00:46<00:00, 8612.15it/s] 99%|█████████▉| 396714/400000 [00:46<00:00, 8645.37it/s] 99%|█████████▉| 397579/400000 [00:46<00:00, 8620.07it/s]100%|█████████▉| 398454/400000 [00:46<00:00, 8657.91it/s]100%|█████████▉| 399321/400000 [00:46<00:00, 8406.96it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8568.08it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6568c96ac8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011079404657661317 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011120367010301571 	 Accuracy: 62

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

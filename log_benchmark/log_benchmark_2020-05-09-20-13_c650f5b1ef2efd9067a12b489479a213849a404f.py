
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6a6fe05470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 20:13:55.218841
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 20:13:55.224057
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 20:13:55.228710
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 20:13:55.233511
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6a55bd9ac8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 3s 3s/step - loss: 352077.1875
Epoch 2/10

1/1 [==============================] - 0s 128ms/step - loss: 233787.6250
Epoch 3/10

1/1 [==============================] - 0s 114ms/step - loss: 135091.4844
Epoch 4/10

1/1 [==============================] - 0s 128ms/step - loss: 72334.1953
Epoch 5/10

1/1 [==============================] - 0s 117ms/step - loss: 39934.4609
Epoch 6/10

1/1 [==============================] - 0s 110ms/step - loss: 23830.4688
Epoch 7/10

1/1 [==============================] - 0s 115ms/step - loss: 15417.7314
Epoch 8/10

1/1 [==============================] - 0s 116ms/step - loss: 10766.1641
Epoch 9/10

1/1 [==============================] - 0s 128ms/step - loss: 7990.3452
Epoch 10/10

1/1 [==============================] - 0s 118ms/step - loss: 6265.3325

  #### Inference Need return ypred, ytrue ######################### 
[[  0.268782    -0.20317726   0.5369562    0.6876571    0.5518988
    0.77999175  -1.3619041   -0.9977937    0.28814062  -0.8323499
    0.47048092   0.72662336  -1.1931238   -1.8111699    1.3426745
   -0.17606318   0.2837718   -0.45570236   2.1113331   -0.6529968
    0.4922639    0.04157132  -0.56294304   0.72421277   0.84875846
    0.73789287  -0.7113291   -0.10390564   0.12176335  -0.66109735
    0.8909285   -1.1653234    0.38174906   0.88010705   0.41248348
    0.66667277  -0.423269     0.04782097   0.41453594   0.9589609
    0.34489614   1.3237451    1.532882    -0.461042     0.425088
    1.1672883    0.30001292   1.1832764    0.62471235   1.1606815
    2.2898955    0.4699186   -1.1897334   -0.41836685   0.83187723
   -0.60001457   1.7092543   -0.7888279   -1.7455878    0.6596976
    1.8161957   -1.4775187   -0.6155206   -1.1606867   -1.2525041
   -0.7152498    0.43891138  -1.376399     0.8990501    1.1878684
    1.0150926   -1.1352774   -0.5515921   -0.7613675    0.9904827
   -1.19556      0.5892854    0.3318721   -0.5131335   -1.2448174
   -0.2843126    0.07374854  -1.3894603    0.6528316   -1.3322432
   -0.41609365   0.9105218    0.15087879  -0.56440634   0.74560606
    0.36516982   0.34332716  -0.21405661   0.5363894    1.6998334
    1.0938257   -1.1922685    0.89467174  -1.2826192    0.83593565
    0.24782118   1.3523936   -0.28250208  -0.9650866   -0.93043673
    0.08861203   1.8456135   -1.333663    -1.1677852   -0.35192287
    1.0648468    1.3447183   -0.4608909   -0.94461405  -0.28899637
    0.5498822    0.71887344  -1.9514713   -0.36154336  -0.41048607
   -0.24793956   7.5842834    7.411539     5.9135046    7.9510145
    7.993304     7.603212     7.229607     9.761704     9.072993
    8.189789     9.853784     8.375046     8.33522      7.4983664
    8.588687     8.80344      8.255391     9.853908     9.21448
    8.909861     8.925532     7.5759673    8.561394     8.579752
    8.211007     7.2840853    7.840917     8.216674     9.570841
    8.321127     8.018669     8.122559     8.563789     7.8097467
    8.299134     9.168731     8.240694     6.8716516    9.507161
   10.181137     6.572        8.906073     8.622307     6.8839426
    9.128805     8.118557     8.846078     8.321554     7.468693
    9.151546     9.031875     9.100019     8.258395    10.061477
    9.052147     8.347865     8.971306     8.558424     8.5663595
    0.9654804    0.54521376   0.7332368    0.9346337    0.10124493
    0.16494578   2.2624414    0.76354957   2.3554382    0.2517478
    0.35532403   1.3764658    3.0683355    0.18408346   1.5063586
    1.1719831    1.1769766    1.030718     0.42964667   1.361958
    1.3322834    0.6671052    0.28301692   3.2612853    1.575325
    0.36763573   0.5731877    2.24673      1.223358     0.67595637
    0.5291276    0.5146196    1.3877378    1.5501739    1.6237919
    2.8851976    0.6486355    0.98077005   0.12209129   2.0190568
    0.1696508    0.58787084   2.1421838    0.97462976   1.0934465
    0.9695354    0.6809778    1.0609901    1.4328052    1.5101681
    0.2842713    0.79775786   0.3225597    3.207695     0.5622311
    1.2271662    1.5237387    1.8480735    0.7650517    0.66909593
    0.6909273    1.4023416    0.5580554    2.2180738    0.4540704
    1.459858     1.5050427    0.25236297   0.26640874   0.81788224
    3.1346993    1.3320398    0.5937928    0.71782804   1.7996022
    1.8858335    1.1017879    2.6818256    0.89790976   0.85984397
    0.61301357   2.921761     0.53447044   1.4818244    0.2789458
    0.33973932   0.72255325   1.7090247    0.22817194   1.7483056
    0.2837804    0.7686788    2.6403837    0.56241316   2.1465435
    1.7629087    0.36839843   1.0802037    2.1163042    0.59416634
    0.6818943    2.3885121    0.43346822   0.31746852   1.4390743
    2.8011909    0.98246074   2.2323744    2.7438827    0.38067853
    1.7290623    0.5440553    1.2212062    0.14216036   2.5581598
    0.3756134    0.6840577    1.5888166    0.595391     0.5590669
    0.0918889   10.120905     8.750217     7.9821506    9.568877
    7.395413     8.604207     9.830068     7.5166945    8.550109
    8.325491     8.271628     8.594028     9.140174     8.485858
    7.0179734    9.970082     9.630991     8.384152     9.12687
    7.2264495    8.263049     7.1879416    9.432482     8.649435
   10.100874     9.315335     8.748646     8.002849     8.37211
    8.234488     8.4972925    9.154994     9.309404     9.231102
    7.875725     8.827021     8.991178    10.629727     7.7948947
    8.1329365    7.544034     9.204643     8.679305     8.9426775
    9.09941     10.049977     7.268307     6.793446     7.9870663
   10.104189     8.292337     8.860499     8.813456     8.87139
    7.503936     9.606696     7.7959685    6.9466243    9.7844715
   -3.8739316  -10.576726     9.662362  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 20:14:06.749263
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.0114
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 20:14:06.754448
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8858.65
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 20:14:06.759761
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.5501
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 20:14:06.764509
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -792.354
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140094431437264
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140091918853008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140091918853512
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140091918854016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140091918854520
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140091918855024

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6a63fd7ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.668724
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.626330
grad_step = 000002, loss = 0.596119
grad_step = 000003, loss = 0.561939
grad_step = 000004, loss = 0.522394
grad_step = 000005, loss = 0.477611
grad_step = 000006, loss = 0.432418
grad_step = 000007, loss = 0.404474
grad_step = 000008, loss = 0.398238
grad_step = 000009, loss = 0.389108
grad_step = 000010, loss = 0.359733
grad_step = 000011, loss = 0.330280
grad_step = 000012, loss = 0.313905
grad_step = 000013, loss = 0.306280
grad_step = 000014, loss = 0.297300
grad_step = 000015, loss = 0.285424
grad_step = 000016, loss = 0.272926
grad_step = 000017, loss = 0.260991
grad_step = 000018, loss = 0.246831
grad_step = 000019, loss = 0.233829
grad_step = 000020, loss = 0.225605
grad_step = 000021, loss = 0.218131
grad_step = 000022, loss = 0.207167
grad_step = 000023, loss = 0.195059
grad_step = 000024, loss = 0.184976
grad_step = 000025, loss = 0.177172
grad_step = 000026, loss = 0.170505
grad_step = 000027, loss = 0.163790
grad_step = 000028, loss = 0.156214
grad_step = 000029, loss = 0.147748
grad_step = 000030, loss = 0.139040
grad_step = 000031, loss = 0.130881
grad_step = 000032, loss = 0.123767
grad_step = 000033, loss = 0.117480
grad_step = 000034, loss = 0.111332
grad_step = 000035, loss = 0.104949
grad_step = 000036, loss = 0.098543
grad_step = 000037, loss = 0.092447
grad_step = 000038, loss = 0.087040
grad_step = 000039, loss = 0.082183
grad_step = 000040, loss = 0.077479
grad_step = 000041, loss = 0.072662
grad_step = 000042, loss = 0.067778
grad_step = 000043, loss = 0.063107
grad_step = 000044, loss = 0.058925
grad_step = 000045, loss = 0.055253
grad_step = 000046, loss = 0.051696
grad_step = 000047, loss = 0.048063
grad_step = 000048, loss = 0.044670
grad_step = 000049, loss = 0.041663
grad_step = 000050, loss = 0.038909
grad_step = 000051, loss = 0.036210
grad_step = 000052, loss = 0.033499
grad_step = 000053, loss = 0.030947
grad_step = 000054, loss = 0.028714
grad_step = 000055, loss = 0.026688
grad_step = 000056, loss = 0.024708
grad_step = 000057, loss = 0.022801
grad_step = 000058, loss = 0.021061
grad_step = 000059, loss = 0.019500
grad_step = 000060, loss = 0.018065
grad_step = 000061, loss = 0.016687
grad_step = 000062, loss = 0.015359
grad_step = 000063, loss = 0.014133
grad_step = 000064, loss = 0.013037
grad_step = 000065, loss = 0.012071
grad_step = 000066, loss = 0.011168
grad_step = 000067, loss = 0.010283
grad_step = 000068, loss = 0.009452
grad_step = 000069, loss = 0.008741
grad_step = 000070, loss = 0.008115
grad_step = 000071, loss = 0.007505
grad_step = 000072, loss = 0.006917
grad_step = 000073, loss = 0.006405
grad_step = 000074, loss = 0.005964
grad_step = 000075, loss = 0.005553
grad_step = 000076, loss = 0.005167
grad_step = 000077, loss = 0.004826
grad_step = 000078, loss = 0.004527
grad_step = 000079, loss = 0.004251
grad_step = 000080, loss = 0.004001
grad_step = 000081, loss = 0.003783
grad_step = 000082, loss = 0.003592
grad_step = 000083, loss = 0.003421
grad_step = 000084, loss = 0.003266
grad_step = 000085, loss = 0.003129
grad_step = 000086, loss = 0.003013
grad_step = 000087, loss = 0.002915
grad_step = 000088, loss = 0.002822
grad_step = 000089, loss = 0.002736
grad_step = 000090, loss = 0.002667
grad_step = 000091, loss = 0.002614
grad_step = 000092, loss = 0.002562
grad_step = 000093, loss = 0.002513
grad_step = 000094, loss = 0.002472
grad_step = 000095, loss = 0.002438
grad_step = 000096, loss = 0.002407
grad_step = 000097, loss = 0.002379
grad_step = 000098, loss = 0.002356
grad_step = 000099, loss = 0.002335
grad_step = 000100, loss = 0.002316
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002299
grad_step = 000102, loss = 0.002284
grad_step = 000103, loss = 0.002271
grad_step = 000104, loss = 0.002258
grad_step = 000105, loss = 0.002246
grad_step = 000106, loss = 0.002237
grad_step = 000107, loss = 0.002229
grad_step = 000108, loss = 0.002220
grad_step = 000109, loss = 0.002212
grad_step = 000110, loss = 0.002205
grad_step = 000111, loss = 0.002199
grad_step = 000112, loss = 0.002193
grad_step = 000113, loss = 0.002187
grad_step = 000114, loss = 0.002182
grad_step = 000115, loss = 0.002178
grad_step = 000116, loss = 0.002173
grad_step = 000117, loss = 0.002169
grad_step = 000118, loss = 0.002165
grad_step = 000119, loss = 0.002161
grad_step = 000120, loss = 0.002158
grad_step = 000121, loss = 0.002155
grad_step = 000122, loss = 0.002152
grad_step = 000123, loss = 0.002149
grad_step = 000124, loss = 0.002146
grad_step = 000125, loss = 0.002143
grad_step = 000126, loss = 0.002141
grad_step = 000127, loss = 0.002138
grad_step = 000128, loss = 0.002136
grad_step = 000129, loss = 0.002134
grad_step = 000130, loss = 0.002131
grad_step = 000131, loss = 0.002129
grad_step = 000132, loss = 0.002127
grad_step = 000133, loss = 0.002125
grad_step = 000134, loss = 0.002123
grad_step = 000135, loss = 0.002121
grad_step = 000136, loss = 0.002119
grad_step = 000137, loss = 0.002117
grad_step = 000138, loss = 0.002115
grad_step = 000139, loss = 0.002113
grad_step = 000140, loss = 0.002111
grad_step = 000141, loss = 0.002109
grad_step = 000142, loss = 0.002107
grad_step = 000143, loss = 0.002105
grad_step = 000144, loss = 0.002103
grad_step = 000145, loss = 0.002101
grad_step = 000146, loss = 0.002099
grad_step = 000147, loss = 0.002097
grad_step = 000148, loss = 0.002095
grad_step = 000149, loss = 0.002093
grad_step = 000150, loss = 0.002091
grad_step = 000151, loss = 0.002089
grad_step = 000152, loss = 0.002087
grad_step = 000153, loss = 0.002085
grad_step = 000154, loss = 0.002083
grad_step = 000155, loss = 0.002082
grad_step = 000156, loss = 0.002080
grad_step = 000157, loss = 0.002078
grad_step = 000158, loss = 0.002076
grad_step = 000159, loss = 0.002074
grad_step = 000160, loss = 0.002072
grad_step = 000161, loss = 0.002070
grad_step = 000162, loss = 0.002068
grad_step = 000163, loss = 0.002066
grad_step = 000164, loss = 0.002065
grad_step = 000165, loss = 0.002063
grad_step = 000166, loss = 0.002061
grad_step = 000167, loss = 0.002059
grad_step = 000168, loss = 0.002057
grad_step = 000169, loss = 0.002055
grad_step = 000170, loss = 0.002054
grad_step = 000171, loss = 0.002052
grad_step = 000172, loss = 0.002050
grad_step = 000173, loss = 0.002047
grad_step = 000174, loss = 0.002045
grad_step = 000175, loss = 0.002043
grad_step = 000176, loss = 0.002041
grad_step = 000177, loss = 0.002039
grad_step = 000178, loss = 0.002036
grad_step = 000179, loss = 0.002033
grad_step = 000180, loss = 0.002032
grad_step = 000181, loss = 0.002029
grad_step = 000182, loss = 0.002026
grad_step = 000183, loss = 0.002022
grad_step = 000184, loss = 0.002021
grad_step = 000185, loss = 0.002017
grad_step = 000186, loss = 0.002014
grad_step = 000187, loss = 0.002009
grad_step = 000188, loss = 0.002007
grad_step = 000189, loss = 0.002003
grad_step = 000190, loss = 0.002000
grad_step = 000191, loss = 0.001996
grad_step = 000192, loss = 0.001991
grad_step = 000193, loss = 0.001988
grad_step = 000194, loss = 0.001983
grad_step = 000195, loss = 0.001979
grad_step = 000196, loss = 0.001975
grad_step = 000197, loss = 0.001971
grad_step = 000198, loss = 0.001970
grad_step = 000199, loss = 0.001971
grad_step = 000200, loss = 0.001972
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001963
grad_step = 000202, loss = 0.001951
grad_step = 000203, loss = 0.001954
grad_step = 000204, loss = 0.001951
grad_step = 000205, loss = 0.001942
grad_step = 000206, loss = 0.001938
grad_step = 000207, loss = 0.001938
grad_step = 000208, loss = 0.001932
grad_step = 000209, loss = 0.001926
grad_step = 000210, loss = 0.001923
grad_step = 000211, loss = 0.001922
grad_step = 000212, loss = 0.001916
grad_step = 000213, loss = 0.001911
grad_step = 000214, loss = 0.001908
grad_step = 000215, loss = 0.001905
grad_step = 000216, loss = 0.001902
grad_step = 000217, loss = 0.001897
grad_step = 000218, loss = 0.001893
grad_step = 000219, loss = 0.001889
grad_step = 000220, loss = 0.001888
grad_step = 000221, loss = 0.001890
grad_step = 000222, loss = 0.001892
grad_step = 000223, loss = 0.001890
grad_step = 000224, loss = 0.001879
grad_step = 000225, loss = 0.001870
grad_step = 000226, loss = 0.001870
grad_step = 000227, loss = 0.001872
grad_step = 000228, loss = 0.001868
grad_step = 000229, loss = 0.001860
grad_step = 000230, loss = 0.001855
grad_step = 000231, loss = 0.001856
grad_step = 000232, loss = 0.001856
grad_step = 000233, loss = 0.001851
grad_step = 000234, loss = 0.001844
grad_step = 000235, loss = 0.001840
grad_step = 000236, loss = 0.001840
grad_step = 000237, loss = 0.001840
grad_step = 000238, loss = 0.001840
grad_step = 000239, loss = 0.001838
grad_step = 000240, loss = 0.001833
grad_step = 000241, loss = 0.001827
grad_step = 000242, loss = 0.001821
grad_step = 000243, loss = 0.001818
grad_step = 000244, loss = 0.001817
grad_step = 000245, loss = 0.001817
grad_step = 000246, loss = 0.001817
grad_step = 000247, loss = 0.001819
grad_step = 000248, loss = 0.001818
grad_step = 000249, loss = 0.001813
grad_step = 000250, loss = 0.001804
grad_step = 000251, loss = 0.001797
grad_step = 000252, loss = 0.001793
grad_step = 000253, loss = 0.001793
grad_step = 000254, loss = 0.001796
grad_step = 000255, loss = 0.001800
grad_step = 000256, loss = 0.001809
grad_step = 000257, loss = 0.001811
grad_step = 000258, loss = 0.001807
grad_step = 000259, loss = 0.001790
grad_step = 000260, loss = 0.001777
grad_step = 000261, loss = 0.001772
grad_step = 000262, loss = 0.001774
grad_step = 000263, loss = 0.001778
grad_step = 000264, loss = 0.001778
grad_step = 000265, loss = 0.001782
grad_step = 000266, loss = 0.001787
grad_step = 000267, loss = 0.001800
grad_step = 000268, loss = 0.001787
grad_step = 000269, loss = 0.001770
grad_step = 000270, loss = 0.001756
grad_step = 000271, loss = 0.001759
grad_step = 000272, loss = 0.001763
grad_step = 000273, loss = 0.001754
grad_step = 000274, loss = 0.001747
grad_step = 000275, loss = 0.001753
grad_step = 000276, loss = 0.001762
grad_step = 000277, loss = 0.001764
grad_step = 000278, loss = 0.001769
grad_step = 000279, loss = 0.001783
grad_step = 000280, loss = 0.001818
grad_step = 000281, loss = 0.001801
grad_step = 000282, loss = 0.001778
grad_step = 000283, loss = 0.001749
grad_step = 000284, loss = 0.001736
grad_step = 000285, loss = 0.001733
grad_step = 000286, loss = 0.001737
grad_step = 000287, loss = 0.001756
grad_step = 000288, loss = 0.001773
grad_step = 000289, loss = 0.001779
grad_step = 000290, loss = 0.001766
grad_step = 000291, loss = 0.001757
grad_step = 000292, loss = 0.001740
grad_step = 000293, loss = 0.001726
grad_step = 000294, loss = 0.001714
grad_step = 000295, loss = 0.001722
grad_step = 000296, loss = 0.001745
grad_step = 000297, loss = 0.001752
grad_step = 000298, loss = 0.001755
grad_step = 000299, loss = 0.001749
grad_step = 000300, loss = 0.001745
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001728
grad_step = 000302, loss = 0.001708
grad_step = 000303, loss = 0.001701
grad_step = 000304, loss = 0.001705
grad_step = 000305, loss = 0.001708
grad_step = 000306, loss = 0.001709
grad_step = 000307, loss = 0.001715
grad_step = 000308, loss = 0.001719
grad_step = 000309, loss = 0.001725
grad_step = 000310, loss = 0.001719
grad_step = 000311, loss = 0.001718
grad_step = 000312, loss = 0.001716
grad_step = 000313, loss = 0.001714
grad_step = 000314, loss = 0.001709
grad_step = 000315, loss = 0.001706
grad_step = 000316, loss = 0.001702
grad_step = 000317, loss = 0.001703
grad_step = 000318, loss = 0.001698
grad_step = 000319, loss = 0.001697
grad_step = 000320, loss = 0.001696
grad_step = 000321, loss = 0.001700
grad_step = 000322, loss = 0.001703
grad_step = 000323, loss = 0.001707
grad_step = 000324, loss = 0.001709
grad_step = 000325, loss = 0.001716
grad_step = 000326, loss = 0.001715
grad_step = 000327, loss = 0.001720
grad_step = 000328, loss = 0.001711
grad_step = 000329, loss = 0.001705
grad_step = 000330, loss = 0.001692
grad_step = 000331, loss = 0.001681
grad_step = 000332, loss = 0.001670
grad_step = 000333, loss = 0.001662
grad_step = 000334, loss = 0.001658
grad_step = 000335, loss = 0.001658
grad_step = 000336, loss = 0.001659
grad_step = 000337, loss = 0.001660
grad_step = 000338, loss = 0.001663
grad_step = 000339, loss = 0.001670
grad_step = 000340, loss = 0.001688
grad_step = 000341, loss = 0.001710
grad_step = 000342, loss = 0.001756
grad_step = 000343, loss = 0.001770
grad_step = 000344, loss = 0.001796
grad_step = 000345, loss = 0.001749
grad_step = 000346, loss = 0.001704
grad_step = 000347, loss = 0.001659
grad_step = 000348, loss = 0.001650
grad_step = 000349, loss = 0.001678
grad_step = 000350, loss = 0.001700
grad_step = 000351, loss = 0.001704
grad_step = 000352, loss = 0.001670
grad_step = 000353, loss = 0.001637
grad_step = 000354, loss = 0.001626
grad_step = 000355, loss = 0.001639
grad_step = 000356, loss = 0.001661
grad_step = 000357, loss = 0.001669
grad_step = 000358, loss = 0.001665
grad_step = 000359, loss = 0.001648
grad_step = 000360, loss = 0.001638
grad_step = 000361, loss = 0.001635
grad_step = 000362, loss = 0.001644
grad_step = 000363, loss = 0.001647
grad_step = 000364, loss = 0.001639
grad_step = 000365, loss = 0.001621
grad_step = 000366, loss = 0.001604
grad_step = 000367, loss = 0.001596
grad_step = 000368, loss = 0.001600
grad_step = 000369, loss = 0.001607
grad_step = 000370, loss = 0.001611
grad_step = 000371, loss = 0.001608
grad_step = 000372, loss = 0.001599
grad_step = 000373, loss = 0.001590
grad_step = 000374, loss = 0.001584
grad_step = 000375, loss = 0.001584
grad_step = 000376, loss = 0.001589
grad_step = 000377, loss = 0.001594
grad_step = 000378, loss = 0.001600
grad_step = 000379, loss = 0.001607
grad_step = 000380, loss = 0.001616
grad_step = 000381, loss = 0.001641
grad_step = 000382, loss = 0.001676
grad_step = 000383, loss = 0.001752
grad_step = 000384, loss = 0.001744
grad_step = 000385, loss = 0.001735
grad_step = 000386, loss = 0.001612
grad_step = 000387, loss = 0.001562
grad_step = 000388, loss = 0.001595
grad_step = 000389, loss = 0.001620
grad_step = 000390, loss = 0.001600
grad_step = 000391, loss = 0.001557
grad_step = 000392, loss = 0.001573
grad_step = 000393, loss = 0.001611
grad_step = 000394, loss = 0.001585
grad_step = 000395, loss = 0.001551
grad_step = 000396, loss = 0.001555
grad_step = 000397, loss = 0.001578
grad_step = 000398, loss = 0.001580
grad_step = 000399, loss = 0.001551
grad_step = 000400, loss = 0.001539
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001553
grad_step = 000402, loss = 0.001567
grad_step = 000403, loss = 0.001561
grad_step = 000404, loss = 0.001542
grad_step = 000405, loss = 0.001532
grad_step = 000406, loss = 0.001539
grad_step = 000407, loss = 0.001549
grad_step = 000408, loss = 0.001552
grad_step = 000409, loss = 0.001545
grad_step = 000410, loss = 0.001539
grad_step = 000411, loss = 0.001544
grad_step = 000412, loss = 0.001565
grad_step = 000413, loss = 0.001606
grad_step = 000414, loss = 0.001667
grad_step = 000415, loss = 0.001779
grad_step = 000416, loss = 0.001861
grad_step = 000417, loss = 0.001930
grad_step = 000418, loss = 0.001740
grad_step = 000419, loss = 0.001560
grad_step = 000420, loss = 0.001529
grad_step = 000421, loss = 0.001633
grad_step = 000422, loss = 0.001672
grad_step = 000423, loss = 0.001552
grad_step = 000424, loss = 0.001532
grad_step = 000425, loss = 0.001611
grad_step = 000426, loss = 0.001598
grad_step = 000427, loss = 0.001534
grad_step = 000428, loss = 0.001534
grad_step = 000429, loss = 0.001582
grad_step = 000430, loss = 0.001582
grad_step = 000431, loss = 0.001523
grad_step = 000432, loss = 0.001513
grad_step = 000433, loss = 0.001548
grad_step = 000434, loss = 0.001549
grad_step = 000435, loss = 0.001511
grad_step = 000436, loss = 0.001500
grad_step = 000437, loss = 0.001524
grad_step = 000438, loss = 0.001530
grad_step = 000439, loss = 0.001507
grad_step = 000440, loss = 0.001496
grad_step = 000441, loss = 0.001511
grad_step = 000442, loss = 0.001518
grad_step = 000443, loss = 0.001505
grad_step = 000444, loss = 0.001495
grad_step = 000445, loss = 0.001505
grad_step = 000446, loss = 0.001516
grad_step = 000447, loss = 0.001511
grad_step = 000448, loss = 0.001510
grad_step = 000449, loss = 0.001524
grad_step = 000450, loss = 0.001546
grad_step = 000451, loss = 0.001553
grad_step = 000452, loss = 0.001565
grad_step = 000453, loss = 0.001568
grad_step = 000454, loss = 0.001582
grad_step = 000455, loss = 0.001558
grad_step = 000456, loss = 0.001528
grad_step = 000457, loss = 0.001494
grad_step = 000458, loss = 0.001479
grad_step = 000459, loss = 0.001478
grad_step = 000460, loss = 0.001486
grad_step = 000461, loss = 0.001502
grad_step = 000462, loss = 0.001511
grad_step = 000463, loss = 0.001509
grad_step = 000464, loss = 0.001493
grad_step = 000465, loss = 0.001479
grad_step = 000466, loss = 0.001470
grad_step = 000467, loss = 0.001465
grad_step = 000468, loss = 0.001465
grad_step = 000469, loss = 0.001470
grad_step = 000470, loss = 0.001478
grad_step = 000471, loss = 0.001483
grad_step = 000472, loss = 0.001486
grad_step = 000473, loss = 0.001481
grad_step = 000474, loss = 0.001477
grad_step = 000475, loss = 0.001469
grad_step = 000476, loss = 0.001462
grad_step = 000477, loss = 0.001455
grad_step = 000478, loss = 0.001451
grad_step = 000479, loss = 0.001449
grad_step = 000480, loss = 0.001449
grad_step = 000481, loss = 0.001449
grad_step = 000482, loss = 0.001450
grad_step = 000483, loss = 0.001453
grad_step = 000484, loss = 0.001456
grad_step = 000485, loss = 0.001462
grad_step = 000486, loss = 0.001468
grad_step = 000487, loss = 0.001480
grad_step = 000488, loss = 0.001488
grad_step = 000489, loss = 0.001506
grad_step = 000490, loss = 0.001502
grad_step = 000491, loss = 0.001506
grad_step = 000492, loss = 0.001479
grad_step = 000493, loss = 0.001458
grad_step = 000494, loss = 0.001438
grad_step = 000495, loss = 0.001433
grad_step = 000496, loss = 0.001440
grad_step = 000497, loss = 0.001451
grad_step = 000498, loss = 0.001471
grad_step = 000499, loss = 0.001482
grad_step = 000500, loss = 0.001504
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001502
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

  date_run                              2020-05-09 20:14:32.518259
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.241185
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 20:14:32.525712
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.151889
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 20:14:32.534749
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.136303
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 20:14:32.542153
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    -1.308
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
0   2020-05-09 20:13:55.218841  ...    mean_absolute_error
1   2020-05-09 20:13:55.224057  ...     mean_squared_error
2   2020-05-09 20:13:55.228710  ...  median_absolute_error
3   2020-05-09 20:13:55.233511  ...               r2_score
4   2020-05-09 20:14:06.749263  ...    mean_absolute_error
5   2020-05-09 20:14:06.754448  ...     mean_squared_error
6   2020-05-09 20:14:06.759761  ...  median_absolute_error
7   2020-05-09 20:14:06.764509  ...               r2_score
8   2020-05-09 20:14:32.518259  ...    mean_absolute_error
9   2020-05-09 20:14:32.525712  ...     mean_squared_error
10  2020-05-09 20:14:32.534749  ...  median_absolute_error
11  2020-05-09 20:14:32.542153  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 52%|█████▏    | 5136384/9912422 [00:00<00:00, 43389536.99it/s]9920512it [00:00, 32408990.14it/s]                             
0it [00:00, ?it/s] 57%|█████▋    | 16384/28881 [00:00<00:00, 150384.67it/s]32768it [00:00, 299186.84it/s]                           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457008.65it/s]1654784it [00:00, 11582604.32it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186892.10it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27d4b6d9e8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2787525cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27d4b29e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27722b6da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27d4b29e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2787525cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27d4b6d9e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f277b5d6400> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27d4b29e10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2787525cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f27d4b6d9e8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f130b5b2208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
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
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c325157fc1629b5c452b55b698f80da668b5ec48819b0158b3fca4dbb2f8a4d4
  Stored in directory: /tmp/pip-ephem-wheel-cache-ofjh3795/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1301720048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3563520/17464789 [=====>........................] - ETA: 0s
11821056/17464789 [===================>..........] - ETA: 0s
16826368/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 20:16:03.534123: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 20:16:03.539770: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-09 20:16:03.539941: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d8cb73b140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 20:16:03.539958: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 16s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 11s - loss: 7.8736 - accuracy: 0.4865
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7842 - accuracy: 0.4923 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7088 - accuracy: 0.4972
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6850 - accuracy: 0.4988
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6601 - accuracy: 0.5004
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6462 - accuracy: 0.5013
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
11000/25000 [============>.................] - ETA: 5s - loss: 7.6304 - accuracy: 0.5024
12000/25000 [=============>................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6277 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
15000/25000 [=================>............] - ETA: 3s - loss: 7.6043 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6216 - accuracy: 0.5029
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6396 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6636 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6534 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6417 - accuracy: 0.5016
25000/25000 [==============================] - 11s 421us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 20:16:22.270120
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 20:16:22.270120  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 20:16:29.874569: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 20:16:29.881013: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-09 20:16:29.881665: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ea28616f80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 20:16:29.881720: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9a985eabe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 2s 2s/step - loss: 1.7822 - crf_viterbi_accuracy: 0.2533 - val_loss: 1.7195 - val_crf_viterbi_accuracy: 0.2667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9a7352ff60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 17s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 12s - loss: 7.7586 - accuracy: 0.4940
 3000/25000 [==>...........................] - ETA: 10s - loss: 7.7791 - accuracy: 0.4927
 4000/25000 [===>..........................] - ETA: 9s - loss: 7.8315 - accuracy: 0.4893 
 5000/25000 [=====>........................] - ETA: 8s - loss: 7.7402 - accuracy: 0.4952
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6641 - accuracy: 0.5002
 7000/25000 [=======>......................] - ETA: 7s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6992 - accuracy: 0.4979
 9000/25000 [=========>....................] - ETA: 6s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 5s - loss: 7.6736 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 4s - loss: 7.6564 - accuracy: 0.5007
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6324 - accuracy: 0.5022
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
15000/25000 [=================>............] - ETA: 3s - loss: 7.6176 - accuracy: 0.5032
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6120 - accuracy: 0.5036
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6233 - accuracy: 0.5028
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6419 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6432 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6482 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6476 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 11s 424us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f9a2f2c4240> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:12:37, 11.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:04:44, 15.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:36:31, 22.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:25:57, 32.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.53M/862M [00:01<5:11:50, 45.9kB/s].vector_cache/glove.6B.zip:   1%|          | 6.74M/862M [00:01<3:37:20, 65.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:01<2:31:18, 93.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:45:30, 134kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:01<1:13:36, 191kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.2M/862M [00:01<51:18, 272kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.9M/862M [00:01<35:46, 387kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.5M/862M [00:02<24:58, 551kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.2M/862M [00:02<17:27, 783kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<12:13, 1.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<08:59, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<08:11, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<09:35, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<07:36, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<05:32, 2.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<09:37, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<08:30, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<06:22, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<07:04, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:39, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<05:58, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<04:20, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<11:14, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<09:12, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<06:46, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<07:48, 1.68MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:10, 1.61MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<06:18, 2.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:13<04:34, 2.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<11:51, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<09:39, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<07:04, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<08:00, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:55, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<05:10, 2.51MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:39, 1.95MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:19, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<05:46, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:06, 2.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<05:37, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<04:12, 3.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<05:56, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<06:45, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:18, 2.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:22<03:51, 3.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<10:08, 1.26MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<08:25, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<06:12, 2.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:18, 1.73MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:25, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:48, 2.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:19, 1.99MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:59, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:32, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:00, 3.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:23:04, 151kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<59:12, 212kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<41:38, 300kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<29:15, 426kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:34:53, 131kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:07:39, 184kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<47:34, 261kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<36:08, 343kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<27:48, 446kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<19:58, 620kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<14:13, 869kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:22, 922kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:38, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:17, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:45, 1.58MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:38, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:54, 2.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:09, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:43, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:18, 2.83MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:34, 2.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<03:51, 3.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:32, 2.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<04:55, 2.45MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:42, 3.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<02:44, 4.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:22:27, 145kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:00:08, 199kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<42:38, 281kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<31:39, 376kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<23:23, 509kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<16:35, 716kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<14:22, 824kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:15, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:09, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:29, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:44, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:51, 2.00MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:42, 2.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:27, 3.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:47, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:34, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:25, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:35, 1.76MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<06:22, 1.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:54, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:28, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:38, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:19, 2.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:11, 3.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:54, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:01, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:03, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:15, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:06, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:42, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:18, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:58, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:33, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:08, 2.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:27, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:35, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:20, 2.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:01, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:32, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:21, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:56, 2.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:11, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:43, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:07, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:33, 1.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:37, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:21, 2.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:59, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:04, 2.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:47, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:00, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:55, 2.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:40, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:33, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:19, 2.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:55, 2.75MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:30, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:16, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:59, 2.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:51, 2.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:51, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:37, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:06, 2.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:18, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:05, 1.74MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:36, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:22, 3.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<09:23, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:57, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<05:54, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:09, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:30, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:03, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<04:24, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:29, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:52, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:23, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:11, 3.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<36:34, 282kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<28:46, 359kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<20:47, 496kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<14:45, 698kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<12:37, 813kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<10:24, 986kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<07:40, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:10, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:37, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:01, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:18, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:18, 1.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:06, 2.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:40, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:48, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:41, 2.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<02:43, 3.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:48, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:44, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:47, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:48, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:19, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:46, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:14, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:17, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:15, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:00, 2.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<02:55, 3.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:58, 891kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:13, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<06:47, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<04:50, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<20:51, 465kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<16:05, 603kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<11:33, 837kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<08:11, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<13:54, 693kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<11:15, 856kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:14, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:27, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:43, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:04, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:14, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:06, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:56, 2.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:26, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:09, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:02, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:41, 2.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:37, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:24, 1.73MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:05, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:59, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:02, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:38, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:39, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:53, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:18, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:11, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:59, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<02:54, 3.15MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<13:33, 675kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<13:12, 693kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<10:01, 913kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<07:13, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:10, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<08:27, 1.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:46, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:55, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:34, 2.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<1:40:47, 89.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<1:13:40, 122kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<52:20, 172kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<36:43, 244kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<27:54, 321kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<22:52, 391kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<16:48, 532kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<11:54, 749kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<10:35, 838kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<10:33, 841kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:08, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:51, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:23, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:32, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<06:01, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:22, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:12, 2.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<55:34, 157kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<41:57, 208kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<29:57, 291kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<21:09, 412kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<14:55, 582kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<17:42, 490kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<15:25, 562kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<11:30, 753kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<08:13, 1.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<08:03, 1.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<08:25, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:29, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:41, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:26, 2.48MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<10:24, 820kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<10:03, 848kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:37, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:30, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<04:01, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<11:49, 715kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<11:01, 768kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:23, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<06:02, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<06:30, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:28, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:52, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:16, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:09, 2.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<09:31, 874kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<09:22, 887kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:14, 1.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:14, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:54, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:48, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:19, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:53, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:53, 2.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<10:16, 796kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<09:58, 820kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:38, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:28, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:03, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:52, 1.18MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:27, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:58, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:02, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:09, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:39, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:41, 2.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:14, 863kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<09:03, 880kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:58, 1.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:01, 1.58MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:39, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<06:33, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:08, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:46, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:45, 2.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<10:22, 756kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<09:48, 799kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<07:29, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:22, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:54, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:29, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<05:05, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:41, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:45, 2.79MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<11:01, 699kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<10:03, 766kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<07:39, 1.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:30, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:55, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:37, 1.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:10, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:44, 2.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:46, 2.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<09:23, 805kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<09:12, 821kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:57, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:59, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:40, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<09:43, 770kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<09:15, 810kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<06:57, 1.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<05:03, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:39, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<11:18, 656kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<10:19, 719kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:47, 952kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:34, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:57, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:24, 1.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:03, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:40, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:38, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:19, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:17, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<03:07, 2.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:15, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:29, 1.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:26, 1.62MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<03:14, 2.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:05, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:04, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:58, 2.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:17, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:01, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:56, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:53, 2.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:09, 3.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<25:49, 271kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<20:04, 349kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<14:31, 481kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<10:16, 678kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<09:18, 746kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:29, 817kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:21, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<04:34, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:50, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:10, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:04, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:57, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:42, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:08, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:02, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:56, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:43, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:01, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:57, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:52, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:53, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:46, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:28, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:44, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:38, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:38, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:05, 1.60MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:22, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:23, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:30, 2.59MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:25, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:50, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:59, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:14, 2.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:07, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:35, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:51, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:05, 3.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:02, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:51, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:43, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:40, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<06:23, 978kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:47, 1.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:19, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:07, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:15, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:29, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:28, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:30, 2.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<04:59, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:37, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:31, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<02:32, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<12:14, 493kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<10:02, 601kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<07:23, 816kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<05:13, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<07:17, 818kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:25, 928kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:49, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:51, 860kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:49, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:19, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:00, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:05, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:07, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:16, 2.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:27, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:20, 1.72MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:33, 2.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:47, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:50, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:10, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:34, 3.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<11:14, 500kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<09:01, 623kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<06:35, 851kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:28, 1.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:37, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:25, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:23, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:26, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:40, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:44, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:56, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:18, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:47, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:12, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:23, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:42, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:08, 2.45MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:32, 3.38MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<51:50, 101kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<37:16, 140kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<26:16, 198kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<19:03, 270kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<14:20, 358kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<10:16, 499kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<07:57, 638kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<06:33, 774kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:49, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:09, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:53, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:55, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:08, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:53, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:59, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:19, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:39, 2.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<16:58, 287kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<12:49, 380kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<09:11, 528kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:09, 671kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:54, 812kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<04:21, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:03, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<2:20:16, 33.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<1:39:03, 47.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<1:09:21, 68.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<48:47, 95.6kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<35:02, 133kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<24:41, 188kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<17:50, 258kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<13:22, 343kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<09:33, 479kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<07:21, 615kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:01, 750kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<04:26, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:48, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:31, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:40, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:34, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:35, 1.22MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:55, 1.49MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:07, 2.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:30, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:36, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:01, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:06, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:10, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:52, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:10, 1.92MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:41, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:13, 3.38MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<07:36, 541kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:59, 588kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<05:17, 776kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<03:46, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:35, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:17, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:22, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:24, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:52, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:56, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:07, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:40, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:47, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:59, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:34, 2.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:42, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:55, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:31, 2.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:39, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:53, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:29, 2.47MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:37, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:30, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:03, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:33, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:42, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:52, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:28, 2.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:35, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:48, 1.94MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:25, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:32, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:44, 1.96MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:23, 2.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:30, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:42, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:21, 2.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:28, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:40, 1.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:19, 2.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:26, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:38, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:17, 2.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:24, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:36, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:16, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:22, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:33, 1.97MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:13, 2.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<00:53, 3.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:50, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:52, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:27, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:29, 1.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:23, 2.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:06, 2.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:47, 3.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<08:34, 336kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<06:33, 439kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<04:41, 611kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:17, 861kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:22, 833kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:54, 966kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:08, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:30, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:37, 756kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:33, 769kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:45, 990kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:58, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:59, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:54, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:26, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:02, 2.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:00, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:54, 1.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:26, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:01, 2.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:55, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:49, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:21, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:26, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:07, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:09, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:16, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:00, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:04, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:12, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:55, 2.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:40, 3.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:21, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:23, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:04, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:45, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:01, 544kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:14, 675kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:21, 921kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:57, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:47, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:20, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:16, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:58, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:41, 2.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:36, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:30, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:08, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:47, 2.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:52, 667kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:23, 803kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:44, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:13, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:46, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:35, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:12, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:06, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:06, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:51, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:55, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:43, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:40, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:42, 2.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:48, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:37, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:26, 3.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:10, 1.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:07, 1.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:50, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:35, 2.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<02:49, 508kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<02:14, 638kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:37, 873kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:19, 1.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:11, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:53, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:48, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:37, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:37, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:40, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:30, 2.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:22, 3.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:44, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:44, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:33, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:23, 2.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:41, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:41, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:31, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:34, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:29, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:23, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:21, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:14, 3.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:53, 926kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:46, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:34, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:23, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:36, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:33, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:25, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:23, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:32, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:26, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:18, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:21, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:11, 2.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:25, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:24, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:17, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:11, 2.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:26, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:29, 980kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:22, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:15, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:15, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:15, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:11, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:06, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.73MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.21MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 712/400000 [00:00<00:56, 7113.58it/s]  0%|          | 1338/400000 [00:00<00:58, 6833.55it/s]  1%|          | 2018/400000 [00:00<00:58, 6820.25it/s]  1%|          | 2719/400000 [00:00<00:57, 6874.09it/s]  1%|          | 3428/400000 [00:00<00:57, 6935.29it/s]  1%|          | 4159/400000 [00:00<00:56, 7042.63it/s]  1%|          | 4829/400000 [00:00<00:56, 6933.64it/s]  1%|▏         | 5555/400000 [00:00<00:56, 7025.78it/s]  2%|▏         | 6274/400000 [00:00<00:55, 7072.98it/s]  2%|▏         | 6997/400000 [00:01<00:55, 7117.86it/s]  2%|▏         | 7718/400000 [00:01<00:54, 7143.70it/s]  2%|▏         | 8447/400000 [00:01<00:54, 7186.39it/s]  2%|▏         | 9173/400000 [00:01<00:54, 7207.27it/s]  2%|▏         | 9888/400000 [00:01<00:54, 7172.76it/s]  3%|▎         | 10601/400000 [00:01<00:54, 7139.00it/s]  3%|▎         | 11312/400000 [00:01<00:54, 7108.31it/s]  3%|▎         | 12021/400000 [00:01<00:54, 7075.65it/s]  3%|▎         | 12738/400000 [00:01<00:54, 7102.22it/s]  3%|▎         | 13448/400000 [00:01<00:54, 7073.91it/s]  4%|▎         | 14155/400000 [00:02<00:55, 7002.43it/s]  4%|▎         | 14877/400000 [00:02<00:54, 7064.26it/s]  4%|▍         | 15605/400000 [00:02<00:53, 7126.01it/s]  4%|▍         | 16330/400000 [00:02<00:53, 7161.46it/s]  4%|▍         | 17047/400000 [00:02<00:54, 7049.09it/s]  4%|▍         | 17775/400000 [00:02<00:53, 7116.01it/s]  5%|▍         | 18510/400000 [00:02<00:53, 7166.02it/s]  5%|▍         | 19228/400000 [00:02<00:53, 7094.12it/s]  5%|▍         | 19954/400000 [00:02<00:53, 7142.68it/s]  5%|▌         | 20669/400000 [00:02<00:53, 7109.17it/s]  5%|▌         | 21381/400000 [00:03<00:53, 7030.57it/s]  6%|▌         | 22113/400000 [00:03<00:53, 7113.06it/s]  6%|▌         | 22848/400000 [00:03<00:52, 7180.90it/s]  6%|▌         | 23571/400000 [00:03<00:52, 7193.69it/s]  6%|▌         | 24291/400000 [00:03<00:52, 7185.22it/s]  6%|▋         | 25010/400000 [00:03<00:52, 7143.73it/s]  6%|▋         | 25725/400000 [00:03<00:53, 7045.82it/s]  7%|▋         | 26431/400000 [00:03<00:52, 7049.30it/s]  7%|▋         | 27174/400000 [00:03<00:52, 7157.91it/s]  7%|▋         | 27924/400000 [00:03<00:51, 7256.68it/s]  7%|▋         | 28651/400000 [00:04<00:51, 7242.73it/s]  7%|▋         | 29376/400000 [00:04<00:51, 7217.12it/s]  8%|▊         | 30099/400000 [00:04<00:51, 7202.74it/s]  8%|▊         | 30820/400000 [00:04<00:51, 7201.08it/s]  8%|▊         | 31552/400000 [00:04<00:50, 7234.29it/s]  8%|▊         | 32276/400000 [00:04<00:50, 7219.20it/s]  8%|▊         | 33003/400000 [00:04<00:50, 7233.76it/s]  8%|▊         | 33727/400000 [00:04<00:50, 7211.35it/s]  9%|▊         | 34449/400000 [00:04<00:50, 7185.84it/s]  9%|▉         | 35168/400000 [00:04<00:50, 7175.20it/s]  9%|▉         | 35886/400000 [00:05<00:51, 7137.64it/s]  9%|▉         | 36600/400000 [00:05<00:51, 7072.65it/s]  9%|▉         | 37319/400000 [00:05<00:51, 7104.88it/s] 10%|▉         | 38035/400000 [00:05<00:50, 7119.67it/s] 10%|▉         | 38769/400000 [00:05<00:50, 7181.54it/s] 10%|▉         | 39488/400000 [00:05<00:50, 7117.22it/s] 10%|█         | 40201/400000 [00:05<00:50, 7055.06it/s] 10%|█         | 40950/400000 [00:05<00:50, 7178.91it/s] 10%|█         | 41694/400000 [00:05<00:49, 7253.83it/s] 11%|█         | 42441/400000 [00:05<00:48, 7316.39it/s] 11%|█         | 43174/400000 [00:06<00:48, 7302.59it/s] 11%|█         | 43917/400000 [00:06<00:48, 7339.92it/s] 11%|█         | 44663/400000 [00:06<00:48, 7375.21it/s] 11%|█▏        | 45409/400000 [00:06<00:47, 7397.60it/s] 12%|█▏        | 46149/400000 [00:06<00:47, 7397.29it/s] 12%|█▏        | 46889/400000 [00:06<00:48, 7302.30it/s] 12%|█▏        | 47631/400000 [00:06<00:48, 7335.60it/s] 12%|█▏        | 48383/400000 [00:06<00:47, 7389.39it/s] 12%|█▏        | 49123/400000 [00:06<00:47, 7374.14it/s] 12%|█▏        | 49863/400000 [00:06<00:47, 7379.51it/s] 13%|█▎        | 50602/400000 [00:07<00:47, 7382.06it/s] 13%|█▎        | 51341/400000 [00:07<00:47, 7338.40it/s] 13%|█▎        | 52075/400000 [00:07<00:47, 7305.91it/s] 13%|█▎        | 52806/400000 [00:07<00:48, 7210.49it/s] 13%|█▎        | 53528/400000 [00:07<00:48, 7150.89it/s] 14%|█▎        | 54244/400000 [00:07<00:48, 7074.95it/s] 14%|█▎        | 54952/400000 [00:07<00:50, 6864.34it/s] 14%|█▍        | 55666/400000 [00:07<00:49, 6943.74it/s] 14%|█▍        | 56394/400000 [00:07<00:48, 7040.20it/s] 14%|█▍        | 57120/400000 [00:07<00:48, 7103.65it/s] 14%|█▍        | 57858/400000 [00:08<00:47, 7183.80it/s] 15%|█▍        | 58610/400000 [00:08<00:46, 7279.97it/s] 15%|█▍        | 59355/400000 [00:08<00:46, 7328.08it/s] 15%|█▌        | 60105/400000 [00:08<00:46, 7378.70it/s] 15%|█▌        | 60850/400000 [00:08<00:45, 7399.82it/s] 15%|█▌        | 61591/400000 [00:08<00:45, 7388.52it/s] 16%|█▌        | 62338/400000 [00:08<00:45, 7411.63it/s] 16%|█▌        | 63089/400000 [00:08<00:45, 7438.58it/s] 16%|█▌        | 63834/400000 [00:08<00:45, 7410.62it/s] 16%|█▌        | 64588/400000 [00:08<00:45, 7447.37it/s] 16%|█▋        | 65333/400000 [00:09<00:45, 7345.64it/s] 17%|█▋        | 66068/400000 [00:09<00:45, 7339.37it/s] 17%|█▋        | 66804/400000 [00:09<00:45, 7345.10it/s] 17%|█▋        | 67539/400000 [00:09<00:45, 7300.72it/s] 17%|█▋        | 68273/400000 [00:09<00:45, 7309.85it/s] 17%|█▋        | 69005/400000 [00:09<00:45, 7288.07it/s] 17%|█▋        | 69752/400000 [00:09<00:44, 7340.77it/s] 18%|█▊        | 70489/400000 [00:09<00:44, 7346.62it/s] 18%|█▊        | 71224/400000 [00:09<00:44, 7347.02it/s] 18%|█▊        | 71959/400000 [00:09<00:44, 7331.28it/s] 18%|█▊        | 72693/400000 [00:10<00:44, 7276.44it/s] 18%|█▊        | 73435/400000 [00:10<00:44, 7317.77it/s] 19%|█▊        | 74181/400000 [00:10<00:44, 7358.27it/s] 19%|█▊        | 74917/400000 [00:10<00:44, 7334.96it/s] 19%|█▉        | 75651/400000 [00:10<00:44, 7322.26it/s] 19%|█▉        | 76384/400000 [00:10<00:44, 7276.15it/s] 19%|█▉        | 77124/400000 [00:10<00:44, 7310.30it/s] 19%|█▉        | 77866/400000 [00:10<00:43, 7341.38it/s] 20%|█▉        | 78601/400000 [00:10<00:43, 7343.92it/s] 20%|█▉        | 79336/400000 [00:11<00:43, 7340.79it/s] 20%|██        | 80071/400000 [00:11<00:43, 7292.56it/s] 20%|██        | 80805/400000 [00:11<00:43, 7304.02it/s] 20%|██        | 81549/400000 [00:11<00:43, 7342.52it/s] 21%|██        | 82290/400000 [00:11<00:43, 7361.02it/s] 21%|██        | 83027/400000 [00:11<00:43, 7334.53it/s] 21%|██        | 83761/400000 [00:11<00:43, 7317.78it/s] 21%|██        | 84493/400000 [00:11<00:44, 7037.28it/s] 21%|██▏       | 85200/400000 [00:11<00:45, 6986.46it/s] 21%|██▏       | 85950/400000 [00:11<00:44, 7130.38it/s] 22%|██▏       | 86695/400000 [00:12<00:43, 7222.77it/s] 22%|██▏       | 87441/400000 [00:12<00:42, 7289.63it/s] 22%|██▏       | 88183/400000 [00:12<00:42, 7325.45it/s] 22%|██▏       | 88920/400000 [00:12<00:42, 7336.39it/s] 22%|██▏       | 89656/400000 [00:12<00:42, 7341.54it/s] 23%|██▎       | 90392/400000 [00:12<00:42, 7346.29it/s] 23%|██▎       | 91127/400000 [00:12<00:42, 7306.17it/s] 23%|██▎       | 91862/400000 [00:12<00:42, 7317.61it/s] 23%|██▎       | 92614/400000 [00:12<00:41, 7376.32it/s] 23%|██▎       | 93362/400000 [00:12<00:41, 7406.59it/s] 24%|██▎       | 94104/400000 [00:13<00:41, 7408.41it/s] 24%|██▎       | 94848/400000 [00:13<00:41, 7416.68it/s] 24%|██▍       | 95590/400000 [00:13<00:41, 7357.13it/s] 24%|██▍       | 96326/400000 [00:13<00:41, 7327.22it/s] 24%|██▍       | 97062/400000 [00:13<00:41, 7335.31it/s] 24%|██▍       | 97796/400000 [00:13<00:41, 7297.18it/s] 25%|██▍       | 98530/400000 [00:13<00:41, 7307.49it/s] 25%|██▍       | 99261/400000 [00:13<00:41, 7278.22it/s] 25%|██▌       | 100000/400000 [00:13<00:41, 7310.89it/s] 25%|██▌       | 100740/400000 [00:13<00:40, 7335.85it/s] 25%|██▌       | 101475/400000 [00:14<00:40, 7339.51it/s] 26%|██▌       | 102210/400000 [00:14<00:40, 7334.77it/s] 26%|██▌       | 102944/400000 [00:14<00:41, 7218.65it/s] 26%|██▌       | 103672/400000 [00:14<00:40, 7236.00it/s] 26%|██▌       | 104396/400000 [00:14<00:41, 7177.40it/s] 26%|██▋       | 105118/400000 [00:14<00:41, 7189.74it/s] 26%|██▋       | 105838/400000 [00:14<00:41, 7153.93it/s] 27%|██▋       | 106564/400000 [00:14<00:40, 7184.72it/s] 27%|██▋       | 107287/400000 [00:14<00:40, 7196.97it/s] 27%|██▋       | 108007/400000 [00:14<00:40, 7191.79it/s] 27%|██▋       | 108727/400000 [00:15<00:41, 7084.68it/s] 27%|██▋       | 109444/400000 [00:15<00:40, 7109.99it/s] 28%|██▊       | 110156/400000 [00:15<00:41, 7006.66it/s] 28%|██▊       | 110858/400000 [00:15<00:41, 7004.41it/s] 28%|██▊       | 111591/400000 [00:15<00:40, 7096.54it/s] 28%|██▊       | 112318/400000 [00:15<00:40, 7145.83it/s] 28%|██▊       | 113034/400000 [00:15<00:40, 7145.57it/s] 28%|██▊       | 113763/400000 [00:15<00:39, 7187.93it/s] 29%|██▊       | 114509/400000 [00:15<00:39, 7265.68it/s] 29%|██▉       | 115243/400000 [00:15<00:39, 7285.92it/s] 29%|██▉       | 115972/400000 [00:16<00:39, 7277.09it/s] 29%|██▉       | 116703/400000 [00:16<00:38, 7284.39it/s] 29%|██▉       | 117432/400000 [00:16<00:38, 7268.44it/s] 30%|██▉       | 118176/400000 [00:16<00:38, 7318.64it/s] 30%|██▉       | 118910/400000 [00:16<00:38, 7324.50it/s] 30%|██▉       | 119652/400000 [00:16<00:38, 7352.44it/s] 30%|███       | 120388/400000 [00:16<00:38, 7305.59it/s] 30%|███       | 121119/400000 [00:16<00:39, 7044.52it/s] 30%|███       | 121826/400000 [00:16<00:40, 6938.23it/s] 31%|███       | 122548/400000 [00:16<00:39, 7018.11it/s] 31%|███       | 123292/400000 [00:17<00:38, 7139.35it/s] 31%|███       | 124028/400000 [00:17<00:38, 7202.19it/s] 31%|███       | 124750/400000 [00:17<00:38, 7161.80it/s] 31%|███▏      | 125497/400000 [00:17<00:37, 7248.84it/s] 32%|███▏      | 126245/400000 [00:17<00:37, 7314.37it/s] 32%|███▏      | 126992/400000 [00:17<00:37, 7360.01it/s] 32%|███▏      | 127739/400000 [00:17<00:36, 7390.34it/s] 32%|███▏      | 128480/400000 [00:17<00:36, 7393.46it/s] 32%|███▏      | 129228/400000 [00:17<00:36, 7418.53it/s] 32%|███▏      | 129971/400000 [00:17<00:36, 7416.09it/s] 33%|███▎      | 130713/400000 [00:18<00:36, 7406.28it/s] 33%|███▎      | 131461/400000 [00:18<00:36, 7427.63it/s] 33%|███▎      | 132204/400000 [00:18<00:36, 7384.94it/s] 33%|███▎      | 132947/400000 [00:18<00:36, 7397.26it/s] 33%|███▎      | 133687/400000 [00:18<00:36, 7391.39it/s] 34%|███▎      | 134427/400000 [00:18<00:36, 7298.41it/s] 34%|███▍      | 135158/400000 [00:18<00:36, 7272.21it/s] 34%|███▍      | 135895/400000 [00:18<00:36, 7299.65it/s] 34%|███▍      | 136626/400000 [00:18<00:36, 7298.60it/s] 34%|███▍      | 137356/400000 [00:18<00:36, 7288.62it/s] 35%|███▍      | 138091/400000 [00:19<00:35, 7305.85it/s] 35%|███▍      | 138830/400000 [00:19<00:35, 7328.76it/s] 35%|███▍      | 139563/400000 [00:19<00:35, 7269.73it/s] 35%|███▌      | 140291/400000 [00:19<00:35, 7258.63it/s] 35%|███▌      | 141021/400000 [00:19<00:35, 7269.95it/s] 35%|███▌      | 141761/400000 [00:19<00:35, 7307.67it/s] 36%|███▌      | 142501/400000 [00:19<00:35, 7333.00it/s] 36%|███▌      | 143246/400000 [00:19<00:34, 7366.45it/s] 36%|███▌      | 143994/400000 [00:19<00:34, 7397.19it/s] 36%|███▌      | 144736/400000 [00:19<00:34, 7402.33it/s] 36%|███▋      | 145477/400000 [00:20<00:34, 7398.26it/s] 37%|███▋      | 146217/400000 [00:20<00:34, 7374.91it/s] 37%|███▋      | 146955/400000 [00:20<00:34, 7364.37it/s] 37%|███▋      | 147702/400000 [00:20<00:34, 7389.37it/s] 37%|███▋      | 148452/400000 [00:20<00:33, 7419.98it/s] 37%|███▋      | 149195/400000 [00:20<00:33, 7408.04it/s] 37%|███▋      | 149938/400000 [00:20<00:33, 7414.52it/s] 38%|███▊      | 150680/400000 [00:20<00:33, 7368.79it/s] 38%|███▊      | 151423/400000 [00:20<00:33, 7386.84it/s] 38%|███▊      | 152168/400000 [00:20<00:33, 7404.85it/s] 38%|███▊      | 152914/400000 [00:21<00:33, 7418.50it/s] 38%|███▊      | 153656/400000 [00:21<00:33, 7418.55it/s] 39%|███▊      | 154398/400000 [00:21<00:33, 7359.90it/s] 39%|███▉      | 155135/400000 [00:21<00:33, 7329.56it/s] 39%|███▉      | 155869/400000 [00:21<00:33, 7287.93it/s] 39%|███▉      | 156614/400000 [00:21<00:33, 7335.65it/s] 39%|███▉      | 157363/400000 [00:21<00:32, 7380.90it/s] 40%|███▉      | 158102/400000 [00:21<00:32, 7357.42it/s] 40%|███▉      | 158847/400000 [00:21<00:32, 7382.29it/s] 40%|███▉      | 159586/400000 [00:22<00:32, 7379.07it/s] 40%|████      | 160324/400000 [00:22<00:32, 7320.11it/s] 40%|████      | 161057/400000 [00:22<00:32, 7314.04it/s] 40%|████      | 161789/400000 [00:22<00:32, 7304.83it/s] 41%|████      | 162541/400000 [00:22<00:32, 7368.01it/s] 41%|████      | 163292/400000 [00:22<00:31, 7407.51it/s] 41%|████      | 164036/400000 [00:22<00:31, 7415.48it/s] 41%|████      | 164788/400000 [00:22<00:31, 7444.90it/s] 41%|████▏     | 165533/400000 [00:22<00:31, 7365.52it/s] 42%|████▏     | 166278/400000 [00:22<00:31, 7390.45it/s] 42%|████▏     | 167027/400000 [00:23<00:31, 7419.36it/s] 42%|████▏     | 167775/400000 [00:23<00:31, 7434.54it/s] 42%|████▏     | 168519/400000 [00:23<00:31, 7421.61it/s] 42%|████▏     | 169262/400000 [00:23<00:31, 7379.55it/s] 43%|████▎     | 170003/400000 [00:23<00:31, 7388.58it/s] 43%|████▎     | 170742/400000 [00:23<00:31, 7383.07it/s] 43%|████▎     | 171481/400000 [00:23<00:31, 7358.19it/s] 43%|████▎     | 172217/400000 [00:23<00:31, 7273.23it/s] 43%|████▎     | 172946/400000 [00:23<00:31, 7277.55it/s] 43%|████▎     | 173692/400000 [00:23<00:30, 7329.68it/s] 44%|████▎     | 174426/400000 [00:24<00:31, 7188.66it/s] 44%|████▍     | 175154/400000 [00:24<00:31, 7214.29it/s] 44%|████▍     | 175893/400000 [00:24<00:30, 7265.67it/s] 44%|████▍     | 176624/400000 [00:24<00:30, 7276.31it/s] 44%|████▍     | 177355/400000 [00:24<00:30, 7284.97it/s] 45%|████▍     | 178103/400000 [00:24<00:30, 7341.71it/s] 45%|████▍     | 178845/400000 [00:24<00:30, 7363.30it/s] 45%|████▍     | 179588/400000 [00:24<00:29, 7382.64it/s] 45%|████▌     | 180329/400000 [00:24<00:29, 7390.28it/s] 45%|████▌     | 181069/400000 [00:24<00:29, 7350.38it/s] 45%|████▌     | 181817/400000 [00:25<00:29, 7388.75it/s] 46%|████▌     | 182561/400000 [00:25<00:29, 7402.62it/s] 46%|████▌     | 183305/400000 [00:25<00:29, 7411.88it/s] 46%|████▌     | 184047/400000 [00:25<00:29, 7373.08it/s] 46%|████▌     | 184785/400000 [00:25<00:29, 7297.04it/s] 46%|████▋     | 185526/400000 [00:25<00:29, 7328.06it/s] 47%|████▋     | 186270/400000 [00:25<00:29, 7359.27it/s] 47%|████▋     | 187012/400000 [00:25<00:28, 7372.07it/s] 47%|████▋     | 187750/400000 [00:25<00:28, 7354.41it/s] 47%|████▋     | 188486/400000 [00:25<00:28, 7334.94it/s] 47%|████▋     | 189226/400000 [00:26<00:28, 7351.91it/s] 47%|████▋     | 189962/400000 [00:26<00:28, 7352.26it/s] 48%|████▊     | 190698/400000 [00:26<00:28, 7351.33it/s] 48%|████▊     | 191440/400000 [00:26<00:28, 7368.85it/s] 48%|████▊     | 192177/400000 [00:26<00:28, 7343.63it/s] 48%|████▊     | 192912/400000 [00:26<00:28, 7339.56it/s] 48%|████▊     | 193656/400000 [00:26<00:28, 7368.18it/s] 49%|████▊     | 194394/400000 [00:26<00:27, 7370.25it/s] 49%|████▉     | 195132/400000 [00:26<00:27, 7360.32it/s] 49%|████▉     | 195869/400000 [00:26<00:27, 7303.33it/s] 49%|████▉     | 196600/400000 [00:27<00:28, 7241.49it/s] 49%|████▉     | 197325/400000 [00:27<00:28, 7164.95it/s] 50%|████▉     | 198061/400000 [00:27<00:27, 7220.56it/s] 50%|████▉     | 198793/400000 [00:27<00:27, 7248.27it/s] 50%|████▉     | 199521/400000 [00:27<00:27, 7254.98it/s] 50%|█████     | 200259/400000 [00:27<00:27, 7290.52it/s] 50%|█████     | 201008/400000 [00:27<00:27, 7346.90it/s] 50%|█████     | 201743/400000 [00:27<00:27, 7342.58it/s] 51%|█████     | 202478/400000 [00:27<00:27, 7283.38it/s] 51%|█████     | 203207/400000 [00:27<00:27, 7276.93it/s] 51%|█████     | 203953/400000 [00:28<00:26, 7329.29it/s] 51%|█████     | 204695/400000 [00:28<00:26, 7355.10it/s] 51%|█████▏    | 205431/400000 [00:28<00:26, 7294.33it/s] 52%|█████▏    | 206171/400000 [00:28<00:26, 7323.92it/s] 52%|█████▏    | 206904/400000 [00:28<00:26, 7316.66it/s] 52%|█████▏    | 207654/400000 [00:28<00:26, 7370.07it/s] 52%|█████▏    | 208392/400000 [00:28<00:26, 7351.35it/s] 52%|█████▏    | 209128/400000 [00:28<00:26, 7299.44it/s] 52%|█████▏    | 209859/400000 [00:28<00:26, 7208.13it/s] 53%|█████▎    | 210581/400000 [00:28<00:26, 7115.43it/s] 53%|█████▎    | 211318/400000 [00:29<00:26, 7189.79it/s] 53%|█████▎    | 212053/400000 [00:29<00:25, 7235.94it/s] 53%|█████▎    | 212778/400000 [00:29<00:25, 7214.94it/s] 53%|█████▎    | 213500/400000 [00:29<00:26, 7157.14it/s] 54%|█████▎    | 214222/400000 [00:29<00:25, 7175.23it/s] 54%|█████▎    | 214959/400000 [00:29<00:25, 7231.90it/s] 54%|█████▍    | 215695/400000 [00:29<00:25, 7269.76it/s] 54%|█████▍    | 216440/400000 [00:29<00:25, 7322.59it/s] 54%|█████▍    | 217178/400000 [00:29<00:24, 7339.04it/s] 54%|█████▍    | 217913/400000 [00:29<00:24, 7287.72it/s] 55%|█████▍    | 218651/400000 [00:30<00:24, 7312.89it/s] 55%|█████▍    | 219390/400000 [00:30<00:24, 7333.28it/s] 55%|█████▌    | 220136/400000 [00:30<00:24, 7370.30it/s] 55%|█████▌    | 220874/400000 [00:30<00:24, 7350.09it/s] 55%|█████▌    | 221610/400000 [00:30<00:25, 7104.32it/s] 56%|█████▌    | 222344/400000 [00:30<00:24, 7173.39it/s] 56%|█████▌    | 223084/400000 [00:30<00:24, 7238.35it/s] 56%|█████▌    | 223825/400000 [00:30<00:24, 7288.75it/s] 56%|█████▌    | 224570/400000 [00:30<00:23, 7334.32it/s] 56%|█████▋    | 225305/400000 [00:30<00:23, 7294.81it/s] 57%|█████▋    | 226042/400000 [00:31<00:23, 7315.99it/s] 57%|█████▋    | 226783/400000 [00:31<00:23, 7343.63it/s] 57%|█████▋    | 227520/400000 [00:31<00:23, 7348.79it/s] 57%|█████▋    | 228256/400000 [00:31<00:23, 7348.14it/s] 57%|█████▋    | 228991/400000 [00:31<00:23, 7306.24it/s] 57%|█████▋    | 229730/400000 [00:31<00:23, 7330.60it/s] 58%|█████▊    | 230467/400000 [00:31<00:23, 7342.01it/s] 58%|█████▊    | 231202/400000 [00:31<00:23, 7338.90it/s] 58%|█████▊    | 231936/400000 [00:31<00:22, 7332.77it/s] 58%|█████▊    | 232670/400000 [00:31<00:22, 7308.28it/s] 58%|█████▊    | 233401/400000 [00:32<00:22, 7290.23it/s] 59%|█████▊    | 234132/400000 [00:32<00:22, 7294.23it/s] 59%|█████▊    | 234862/400000 [00:32<00:22, 7197.64it/s] 59%|█████▉    | 235588/400000 [00:32<00:22, 7213.87it/s] 59%|█████▉    | 236310/400000 [00:32<00:22, 7207.66it/s] 59%|█████▉    | 237049/400000 [00:32<00:22, 7260.70it/s] 59%|█████▉    | 237790/400000 [00:32<00:22, 7302.78it/s] 60%|█████▉    | 238531/400000 [00:32<00:22, 7333.97it/s] 60%|█████▉    | 239281/400000 [00:32<00:21, 7382.57it/s] 60%|██████    | 240020/400000 [00:33<00:21, 7375.96it/s] 60%|██████    | 240758/400000 [00:33<00:21, 7361.72it/s] 60%|██████    | 241501/400000 [00:33<00:21, 7379.37it/s] 61%|██████    | 242243/400000 [00:33<00:21, 7390.80it/s] 61%|██████    | 242991/400000 [00:33<00:21, 7416.03it/s] 61%|██████    | 243733/400000 [00:33<00:21, 7324.30it/s] 61%|██████    | 244470/400000 [00:33<00:21, 7336.36it/s] 61%|██████▏   | 245204/400000 [00:33<00:21, 7333.87it/s] 61%|██████▏   | 245938/400000 [00:33<00:21, 7328.11it/s] 62%|██████▏   | 246671/400000 [00:33<00:21, 7250.23it/s] 62%|██████▏   | 247398/400000 [00:34<00:21, 7254.20it/s] 62%|██████▏   | 248129/400000 [00:34<00:20, 7270.53it/s] 62%|██████▏   | 248857/400000 [00:34<00:21, 7163.73it/s] 62%|██████▏   | 249594/400000 [00:34<00:20, 7220.20it/s] 63%|██████▎   | 250317/400000 [00:34<00:20, 7206.98it/s] 63%|██████▎   | 251045/400000 [00:34<00:20, 7226.81it/s] 63%|██████▎   | 251769/400000 [00:34<00:20, 7226.51it/s] 63%|██████▎   | 252492/400000 [00:34<00:20, 7188.19it/s] 63%|██████▎   | 253225/400000 [00:34<00:20, 7228.90it/s] 63%|██████▎   | 253949/400000 [00:34<00:20, 7174.12it/s] 64%|██████▎   | 254678/400000 [00:35<00:20, 7206.78it/s] 64%|██████▍   | 255428/400000 [00:35<00:19, 7289.96it/s] 64%|██████▍   | 256164/400000 [00:35<00:19, 7310.05it/s] 64%|██████▍   | 256911/400000 [00:35<00:19, 7356.42it/s] 64%|██████▍   | 257657/400000 [00:35<00:19, 7386.12it/s] 65%|██████▍   | 258396/400000 [00:35<00:19, 7267.57it/s] 65%|██████▍   | 259124/400000 [00:35<00:19, 7248.12it/s] 65%|██████▍   | 259866/400000 [00:35<00:19, 7297.56it/s] 65%|██████▌   | 260607/400000 [00:35<00:19, 7327.36it/s] 65%|██████▌   | 261352/400000 [00:35<00:18, 7361.81it/s] 66%|██████▌   | 262089/400000 [00:36<00:18, 7340.36it/s] 66%|██████▌   | 262824/400000 [00:36<00:18, 7319.50it/s] 66%|██████▌   | 263571/400000 [00:36<00:18, 7361.27it/s] 66%|██████▌   | 264308/400000 [00:36<00:18, 7348.91it/s] 66%|██████▋   | 265046/400000 [00:36<00:18, 7357.02it/s] 66%|██████▋   | 265782/400000 [00:36<00:18, 7338.62it/s] 67%|██████▋   | 266521/400000 [00:36<00:18, 7351.24it/s] 67%|██████▋   | 267267/400000 [00:36<00:17, 7382.31it/s] 67%|██████▋   | 268006/400000 [00:36<00:17, 7373.82it/s] 67%|██████▋   | 268750/400000 [00:36<00:17, 7392.50it/s] 67%|██████▋   | 269490/400000 [00:37<00:17, 7373.36it/s] 68%|██████▊   | 270228/400000 [00:37<00:17, 7316.90it/s] 68%|██████▊   | 270960/400000 [00:37<00:17, 7278.20it/s] 68%|██████▊   | 271688/400000 [00:37<00:18, 7025.51it/s] 68%|██████▊   | 272424/400000 [00:37<00:17, 7120.28it/s] 68%|██████▊   | 273157/400000 [00:37<00:17, 7180.57it/s] 68%|██████▊   | 273893/400000 [00:37<00:17, 7231.60it/s] 69%|██████▊   | 274621/400000 [00:37<00:17, 7244.73it/s] 69%|██████▉   | 275363/400000 [00:37<00:17, 7296.19it/s] 69%|██████▉   | 276113/400000 [00:37<00:16, 7353.80it/s] 69%|██████▉   | 276849/400000 [00:38<00:16, 7350.37it/s] 69%|██████▉   | 277593/400000 [00:38<00:16, 7376.56it/s] 70%|██████▉   | 278342/400000 [00:38<00:16, 7409.92it/s] 70%|██████▉   | 279099/400000 [00:38<00:16, 7455.40it/s] 70%|██████▉   | 279845/400000 [00:38<00:16, 7440.20it/s] 70%|███████   | 280590/400000 [00:38<00:16, 7429.96it/s] 70%|███████   | 281334/400000 [00:38<00:16, 7384.59it/s] 71%|███████   | 282079/400000 [00:38<00:15, 7402.91it/s] 71%|███████   | 282820/400000 [00:38<00:16, 7255.49it/s] 71%|███████   | 283547/400000 [00:38<00:16, 7191.88it/s] 71%|███████   | 284299/400000 [00:39<00:15, 7285.63it/s] 71%|███████▏  | 285031/400000 [00:39<00:15, 7295.80it/s] 71%|███████▏  | 285778/400000 [00:39<00:15, 7345.28it/s] 72%|███████▏  | 286523/400000 [00:39<00:15, 7371.96it/s] 72%|███████▏  | 287261/400000 [00:39<00:15, 7348.34it/s] 72%|███████▏  | 287997/400000 [00:39<00:15, 7334.59it/s] 72%|███████▏  | 288731/400000 [00:39<00:15, 7289.74it/s] 72%|███████▏  | 289464/400000 [00:39<00:15, 7297.94it/s] 73%|███████▎  | 290194/400000 [00:39<00:15, 7297.40it/s] 73%|███████▎  | 290924/400000 [00:39<00:14, 7282.65it/s] 73%|███████▎  | 291665/400000 [00:40<00:14, 7320.15it/s] 73%|███████▎  | 292398/400000 [00:40<00:14, 7320.27it/s] 73%|███████▎  | 293144/400000 [00:40<00:14, 7359.21it/s] 73%|███████▎  | 293892/400000 [00:40<00:14, 7394.45it/s] 74%|███████▎  | 294632/400000 [00:40<00:14, 7352.33it/s] 74%|███████▍  | 295368/400000 [00:40<00:14, 7328.41it/s] 74%|███████▍  | 296101/400000 [00:40<00:14, 7197.25it/s] 74%|███████▍  | 296844/400000 [00:40<00:14, 7264.68it/s] 74%|███████▍  | 297585/400000 [00:40<00:14, 7305.79it/s] 75%|███████▍  | 298317/400000 [00:40<00:13, 7273.60it/s] 75%|███████▍  | 299045/400000 [00:41<00:13, 7272.08it/s] 75%|███████▍  | 299777/400000 [00:41<00:13, 7285.75it/s] 75%|███████▌  | 300508/400000 [00:41<00:13, 7291.09it/s] 75%|███████▌  | 301241/400000 [00:41<00:13, 7300.51it/s] 75%|███████▌  | 301972/400000 [00:41<00:13, 7248.45it/s] 76%|███████▌  | 302702/400000 [00:41<00:13, 7262.74it/s] 76%|███████▌  | 303435/400000 [00:41<00:13, 7281.75it/s] 76%|███████▌  | 304184/400000 [00:41<00:13, 7341.65it/s] 76%|███████▌  | 304926/400000 [00:41<00:12, 7363.64it/s] 76%|███████▋  | 305667/400000 [00:41<00:12, 7376.73it/s] 77%|███████▋  | 306405/400000 [00:42<00:12, 7376.53it/s] 77%|███████▋  | 307143/400000 [00:42<00:12, 7332.26it/s] 77%|███████▋  | 307879/400000 [00:42<00:12, 7338.49it/s] 77%|███████▋  | 308613/400000 [00:42<00:12, 7332.85it/s] 77%|███████▋  | 309347/400000 [00:42<00:12, 7307.09it/s] 78%|███████▊  | 310081/400000 [00:42<00:12, 7316.56it/s] 78%|███████▊  | 310813/400000 [00:42<00:12, 7301.78it/s] 78%|███████▊  | 311554/400000 [00:42<00:12, 7331.69it/s] 78%|███████▊  | 312288/400000 [00:42<00:12, 7228.95it/s] 78%|███████▊  | 313012/400000 [00:43<00:12, 7124.56it/s] 78%|███████▊  | 313726/400000 [00:43<00:12, 7104.38it/s] 79%|███████▊  | 314455/400000 [00:43<00:11, 7157.33it/s] 79%|███████▉  | 315195/400000 [00:43<00:11, 7227.95it/s] 79%|███████▉  | 315927/400000 [00:43<00:11, 7255.03it/s] 79%|███████▉  | 316653/400000 [00:43<00:11, 7158.11it/s] 79%|███████▉  | 317403/400000 [00:43<00:11, 7256.01it/s] 80%|███████▉  | 318146/400000 [00:43<00:11, 7306.05it/s] 80%|███████▉  | 318899/400000 [00:43<00:11, 7370.07it/s] 80%|███████▉  | 319637/400000 [00:43<00:10, 7368.62it/s] 80%|████████  | 320375/400000 [00:44<00:10, 7326.55it/s] 80%|████████  | 321115/400000 [00:44<00:10, 7347.92it/s] 80%|████████  | 321851/400000 [00:44<00:10, 7268.70it/s] 81%|████████  | 322596/400000 [00:44<00:10, 7321.33it/s] 81%|████████  | 323343/400000 [00:44<00:10, 7363.30it/s] 81%|████████  | 324084/400000 [00:44<00:10, 7376.21it/s] 81%|████████  | 324828/400000 [00:44<00:10, 7392.89it/s] 81%|████████▏ | 325568/400000 [00:44<00:10, 7337.40it/s] 82%|████████▏ | 326310/400000 [00:44<00:10, 7359.50it/s] 82%|████████▏ | 327047/400000 [00:44<00:10, 7210.39it/s] 82%|████████▏ | 327781/400000 [00:45<00:09, 7248.77it/s] 82%|████████▏ | 328534/400000 [00:45<00:09, 7328.45it/s] 82%|████████▏ | 329273/400000 [00:45<00:09, 7344.90it/s] 83%|████████▎ | 330033/400000 [00:45<00:09, 7417.47it/s] 83%|████████▎ | 330776/400000 [00:45<00:09, 7381.70it/s] 83%|████████▎ | 331534/400000 [00:45<00:09, 7439.74it/s] 83%|████████▎ | 332279/400000 [00:45<00:09, 7437.61it/s] 83%|████████▎ | 333024/400000 [00:45<00:09, 7391.15it/s] 83%|████████▎ | 333764/400000 [00:45<00:08, 7368.82it/s] 84%|████████▎ | 334504/400000 [00:45<00:08, 7375.62it/s] 84%|████████▍ | 335255/400000 [00:46<00:08, 7414.77it/s] 84%|████████▍ | 336002/400000 [00:46<00:08, 7429.67it/s] 84%|████████▍ | 336746/400000 [00:46<00:08, 7379.61it/s] 84%|████████▍ | 337495/400000 [00:46<00:08, 7409.24it/s] 85%|████████▍ | 338240/400000 [00:46<00:08, 7420.52it/s] 85%|████████▍ | 338989/400000 [00:46<00:08, 7438.37it/s] 85%|████████▍ | 339733/400000 [00:46<00:08, 7319.24it/s] 85%|████████▌ | 340466/400000 [00:46<00:08, 7312.50it/s] 85%|████████▌ | 341211/400000 [00:46<00:07, 7352.76it/s] 85%|████████▌ | 341947/400000 [00:46<00:07, 7351.74it/s] 86%|████████▌ | 342692/400000 [00:47<00:07, 7380.01it/s] 86%|████████▌ | 343438/400000 [00:47<00:07, 7403.81it/s] 86%|████████▌ | 344179/400000 [00:47<00:07, 7370.35it/s] 86%|████████▌ | 344917/400000 [00:47<00:07, 7256.67it/s] 86%|████████▋ | 345644/400000 [00:47<00:07, 6852.75it/s] 87%|████████▋ | 346387/400000 [00:47<00:07, 7015.34it/s] 87%|████████▋ | 347126/400000 [00:47<00:07, 7123.55it/s] 87%|████████▋ | 347856/400000 [00:47<00:07, 7175.35it/s] 87%|████████▋ | 348610/400000 [00:47<00:07, 7280.08it/s] 87%|████████▋ | 349360/400000 [00:47<00:06, 7342.85it/s] 88%|████████▊ | 350116/400000 [00:48<00:06, 7405.36it/s] 88%|████████▊ | 350859/400000 [00:48<00:06, 7410.13it/s] 88%|████████▊ | 351606/400000 [00:48<00:06, 7425.56it/s] 88%|████████▊ | 352359/400000 [00:48<00:06, 7456.28it/s] 88%|████████▊ | 353111/400000 [00:48<00:06, 7474.73it/s] 88%|████████▊ | 353859/400000 [00:48<00:06, 7461.61it/s] 89%|████████▊ | 354606/400000 [00:48<00:06, 7414.90it/s] 89%|████████▉ | 355348/400000 [00:48<00:06, 7327.54it/s] 89%|████████▉ | 356091/400000 [00:48<00:05, 7357.43it/s] 89%|████████▉ | 356828/400000 [00:48<00:05, 7210.26it/s] 89%|████████▉ | 357550/400000 [00:49<00:05, 7184.07it/s] 90%|████████▉ | 358282/400000 [00:49<00:05, 7224.05it/s] 90%|████████▉ | 359012/400000 [00:49<00:05, 7245.22it/s] 90%|████████▉ | 359746/400000 [00:49<00:05, 7271.05it/s] 90%|█████████ | 360474/400000 [00:49<00:05, 7273.22it/s] 90%|█████████ | 361210/400000 [00:49<00:05, 7298.19it/s] 90%|█████████ | 361940/400000 [00:49<00:05, 7261.69it/s] 91%|█████████ | 362667/400000 [00:49<00:05, 7244.63it/s] 91%|█████████ | 363403/400000 [00:49<00:05, 7277.33it/s] 91%|█████████ | 364144/400000 [00:49<00:04, 7316.25it/s] 91%|█████████ | 364890/400000 [00:50<00:04, 7357.53it/s] 91%|█████████▏| 365626/400000 [00:50<00:04, 7339.73it/s] 92%|█████████▏| 366361/400000 [00:50<00:04, 7280.14it/s] 92%|█████████▏| 367107/400000 [00:50<00:04, 7332.78it/s] 92%|█████████▏| 367848/400000 [00:50<00:04, 7354.41it/s] 92%|█████████▏| 368588/400000 [00:50<00:04, 7367.43it/s] 92%|█████████▏| 369332/400000 [00:50<00:04, 7388.22it/s] 93%|█████████▎| 370071/400000 [00:50<00:04, 7267.05it/s] 93%|█████████▎| 370802/400000 [00:50<00:04, 7277.78it/s] 93%|█████████▎| 371546/400000 [00:50<00:03, 7323.49it/s] 93%|█████████▎| 372290/400000 [00:51<00:03, 7357.46it/s] 93%|█████████▎| 373035/400000 [00:51<00:03, 7383.21it/s] 93%|█████████▎| 373774/400000 [00:51<00:03, 7357.54it/s] 94%|█████████▎| 374516/400000 [00:51<00:03, 7375.55it/s] 94%|█████████▍| 375256/400000 [00:51<00:03, 7377.98it/s] 94%|█████████▍| 375994/400000 [00:51<00:03, 7377.05it/s] 94%|█████████▍| 376732/400000 [00:51<00:03, 7361.31it/s] 94%|█████████▍| 377469/400000 [00:51<00:03, 7346.11it/s] 95%|█████████▍| 378205/400000 [00:51<00:02, 7347.49it/s] 95%|█████████▍| 378941/400000 [00:51<00:02, 7348.52it/s] 95%|█████████▍| 379676/400000 [00:52<00:02, 7345.17it/s] 95%|█████████▌| 380420/400000 [00:52<00:02, 7372.43it/s] 95%|█████████▌| 381158/400000 [00:52<00:02, 7361.91it/s] 95%|█████████▌| 381895/400000 [00:52<00:02, 7293.71it/s] 96%|█████████▌| 382625/400000 [00:52<00:02, 7246.73it/s] 96%|█████████▌| 383362/400000 [00:52<00:02, 7281.50it/s] 96%|█████████▌| 384103/400000 [00:52<00:02, 7318.56it/s] 96%|█████████▌| 384836/400000 [00:52<00:02, 7289.03it/s] 96%|█████████▋| 385576/400000 [00:52<00:01, 7321.46it/s] 97%|█████████▋| 386317/400000 [00:53<00:01, 7347.47it/s] 97%|█████████▋| 387061/400000 [00:53<00:01, 7373.59it/s] 97%|█████████▋| 387810/400000 [00:53<00:01, 7407.14it/s] 97%|█████████▋| 388551/400000 [00:53<00:01, 7392.32it/s] 97%|█████████▋| 389291/400000 [00:53<00:01, 7343.55it/s] 98%|█████████▊| 390026/400000 [00:53<00:01, 7339.07it/s] 98%|█████████▊| 390772/400000 [00:53<00:01, 7373.72it/s] 98%|█████████▊| 391513/400000 [00:53<00:01, 7382.15it/s] 98%|█████████▊| 392260/400000 [00:53<00:01, 7405.54it/s] 98%|█████████▊| 393001/400000 [00:53<00:00, 7362.46it/s] 98%|█████████▊| 393748/400000 [00:54<00:00, 7394.14it/s] 99%|█████████▊| 394488/400000 [00:54<00:00, 7387.80it/s] 99%|█████████▉| 395227/400000 [00:54<00:00, 7282.22it/s] 99%|█████████▉| 395956/400000 [00:54<00:00, 7265.80it/s] 99%|█████████▉| 396683/400000 [00:54<00:00, 7226.90it/s] 99%|█████████▉| 397406/400000 [00:54<00:00, 7221.47it/s]100%|█████████▉| 398141/400000 [00:54<00:00, 7257.28it/s]100%|█████████▉| 398875/400000 [00:54<00:00, 7281.00it/s]100%|█████████▉| 399607/400000 [00:54<00:00, 7290.60it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7289.10it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9a382db0f0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010975534909102368 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011077714205585594 	 Accuracy: 61

  model saves at 61% accuracy 

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

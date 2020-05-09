
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f851c3c6470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 21:11:42.060755
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 21:11:42.064049
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 21:11:42.067011
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 21:11:42.069997
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f8508951b00> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 351178.4375
Epoch 2/10

1/1 [==============================] - 0s 91ms/step - loss: 229863.8906
Epoch 3/10

1/1 [==============================] - 0s 88ms/step - loss: 130224.1641
Epoch 4/10

1/1 [==============================] - 0s 86ms/step - loss: 66166.9688
Epoch 5/10

1/1 [==============================] - 0s 87ms/step - loss: 34805.4609
Epoch 6/10

1/1 [==============================] - 0s 91ms/step - loss: 19936.9102
Epoch 7/10

1/1 [==============================] - 0s 87ms/step - loss: 12489.0420
Epoch 8/10

1/1 [==============================] - 0s 86ms/step - loss: 8441.3008
Epoch 9/10

1/1 [==============================] - 0s 88ms/step - loss: 6122.5083
Epoch 10/10

1/1 [==============================] - 0s 84ms/step - loss: 4728.3267

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.0852048   1.2217312  -1.0939116   1.9791559   0.44557184  0.23101798
  -0.36541152  0.70413417 -2.3844097  -1.6103312  -0.12724829  0.3886656
  -1.5014336  -2.076607   -0.3257674   0.5196178  -0.9125887  -1.9447526
   0.08282098 -0.85407007 -0.8553444  -1.6699644  -0.0626862  -2.4873257
   1.320411    1.1303287   1.0146347  -0.46889237  0.59234935 -1.1386656
   0.7251719   2.4923067  -1.8084664   0.14576006  1.3450003   0.35116732
   0.37575126 -0.88278735  1.2843461   0.65144753 -0.39488757  0.08364531
  -0.6666838  -0.5148448  -1.3761727   1.2750211   0.57368946  1.7102878
   0.4267002  -0.1082326   1.306231   -0.47857845 -0.25399962 -0.0728581
  -0.34286788 -0.49176848 -0.79592013  1.3682406  -0.4248868  -0.10295469
   0.21484703 10.611929    9.50726    11.047702    9.736874   10.333467
  11.050239    9.932462    9.783048   10.116286   11.747546    9.83668
   7.8002014  10.344143    7.6175876  10.051568   11.233775    9.63258
  10.0928335  10.309017    9.982924   10.923723   10.777976   11.38039
  10.522758   10.340234   11.488885   10.116329   11.664969   12.16652
  12.4229355   8.968433    9.0064     11.322425    9.592088    8.867127
  10.100017    8.642238   10.745475   10.842088   11.117133    9.74652
  11.075491    9.390221   10.326031    8.680824   10.07665    10.019722
   9.859142   10.196831   10.773706   10.886471   10.16363     9.962723
  11.450673    8.808465    9.546476    8.17788    11.9688425  10.280416
   1.0927005   1.8041215  -1.7966117   1.2026906   0.6787011   0.05245444
   0.6761714  -0.96158695  0.48702472  0.8305361   2.232545   -0.63244474
   1.8425814   1.0810149  -0.6435383   0.22733572 -0.41011703  0.6994765
   0.7841351   1.3907056   0.08262104 -0.70322335  0.63572943  0.8870599
  -0.6492331   0.09570777 -0.08591316 -1.0270404   0.79205924  0.4310624
   0.7609724   0.7344522   0.2197574   0.9547395   0.9688251   1.342118
   0.7610351   0.20299509  0.34702766 -1.3225121   1.9346668   1.5054088
   0.05845988 -0.9926671   1.4292299   1.1392403  -1.0414349  -2.336654
   0.6695007   0.50056005 -0.20535833  0.75028455 -0.62339526 -1.3190739
  -1.4898726   1.4764676   0.95799    -0.03930593  1.0969048  -1.45912
   0.6862772   1.9132473   0.50142497  0.19328117  0.80242956  2.1365094
   2.7676659   0.7555191   0.26925308  0.8192401   0.12698102  0.06702548
   1.842022    0.9205141   0.27294356  1.0552524   0.933587    0.22629845
   0.6211666   0.6796139   0.41177845  2.388183    0.18654668  1.8949299
   0.7306179   0.7423264   0.7712746   1.5515878   0.29552686  0.17874926
   1.7177207   1.5325028   2.9292173   1.4314798   0.15239906  0.9537872
   0.2023462   0.3252712   0.20110577  1.3741713   1.046775    0.5128866
   0.49364555  0.21651495  0.23252273  0.17564166  1.3657501   2.9991617
   2.8223724   0.35779822  0.20610476  3.4730496   0.08321416  2.645482
   1.0385587   0.6910159   1.1632212   0.92875296  0.91929317  0.6233056
   0.23679274  8.407233   10.761862    9.129193   10.551047   11.042204
  11.428059   10.019387   11.343283   11.091307   10.211229   10.504239
   9.438311   11.105152    9.622977    9.726931    8.253326    8.035756
   9.422014    9.810212   10.928859    9.85447     9.429258   11.236073
   9.455123   10.243936   10.193377   11.026007    9.373465   10.813801
   8.197228    9.348793    9.632781    8.573028   10.372115    9.583319
   9.844136    8.50855    11.851731    9.37987    11.514644   10.916535
  12.351333    9.705858   10.785077   11.313583    8.234806   10.183442
  10.70826    11.160805   11.100228   11.210343    9.378462    9.649963
  11.405221   10.286182   10.509505   10.827638    9.479021    8.013961
   1.3028175   0.7855867   0.338728    0.6953033   0.52937555  0.34157366
   1.4786906   1.9016187   1.904527    0.49807596  0.9999711   0.2992047
   1.245019    0.12086052  1.5171835   2.310848    1.0491261   2.5590744
   0.20264184  2.018706    0.5764141   0.5958533   1.3472042   2.6733208
   0.3193059   0.61044073  3.3349829   0.30878735  0.5523877   0.29687518
   0.7270486   1.4875747   1.9562457   0.38330555  1.3588374   2.100922
   2.4111547   0.20208973  0.4546618   0.15082002  1.3914938   0.9682106
   0.28312933  3.8315578   0.18531084  0.8167322   1.9034549   0.9963716
   0.33921236  2.322082    3.1833982   1.4147694   0.52023405  0.4646837
   1.867048    0.35272074  0.66455805  0.3033319   1.8014827   0.22479439
  -3.148239    7.2059245  -6.3270535 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 21:11:49.626383
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.0204
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 21:11:49.629745
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8493.22
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 21:11:49.632322
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.8632
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 21:11:49.635168
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -759.628
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140208992147384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140207782282128
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140207782282632
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140207782283136
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140207782283640
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140207782284144

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f84f4b44ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.533117
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.497086
grad_step = 000002, loss = 0.471808
grad_step = 000003, loss = 0.445046
grad_step = 000004, loss = 0.417290
grad_step = 000005, loss = 0.392080
grad_step = 000006, loss = 0.374512
grad_step = 000007, loss = 0.365493
grad_step = 000008, loss = 0.356509
grad_step = 000009, loss = 0.344276
grad_step = 000010, loss = 0.331366
grad_step = 000011, loss = 0.321521
grad_step = 000012, loss = 0.313862
grad_step = 000013, loss = 0.306206
grad_step = 000014, loss = 0.297407
grad_step = 000015, loss = 0.287599
grad_step = 000016, loss = 0.277657
grad_step = 000017, loss = 0.268435
grad_step = 000018, loss = 0.260053
grad_step = 000019, loss = 0.252007
grad_step = 000020, loss = 0.244159
grad_step = 000021, loss = 0.236619
grad_step = 000022, loss = 0.229203
grad_step = 000023, loss = 0.221701
grad_step = 000024, loss = 0.214221
grad_step = 000025, loss = 0.206943
grad_step = 000026, loss = 0.199951
grad_step = 000027, loss = 0.193297
grad_step = 000028, loss = 0.186971
grad_step = 000029, loss = 0.180750
grad_step = 000030, loss = 0.174360
grad_step = 000031, loss = 0.167895
grad_step = 000032, loss = 0.161796
grad_step = 000033, loss = 0.156273
grad_step = 000034, loss = 0.151001
grad_step = 000035, loss = 0.145568
grad_step = 000036, loss = 0.140008
grad_step = 000037, loss = 0.134667
grad_step = 000038, loss = 0.129735
grad_step = 000039, loss = 0.125069
grad_step = 000040, loss = 0.120405
grad_step = 000041, loss = 0.115703
grad_step = 000042, loss = 0.111145
grad_step = 000043, loss = 0.106881
grad_step = 000044, loss = 0.102837
grad_step = 000045, loss = 0.098837
grad_step = 000046, loss = 0.094857
grad_step = 000047, loss = 0.091029
grad_step = 000048, loss = 0.087408
grad_step = 000049, loss = 0.083916
grad_step = 000050, loss = 0.080507
grad_step = 000051, loss = 0.077185
grad_step = 000052, loss = 0.073981
grad_step = 000053, loss = 0.070905
grad_step = 000054, loss = 0.067932
grad_step = 000055, loss = 0.065075
grad_step = 000056, loss = 0.062337
grad_step = 000057, loss = 0.059663
grad_step = 000058, loss = 0.057059
grad_step = 000059, loss = 0.054566
grad_step = 000060, loss = 0.052202
grad_step = 000061, loss = 0.049930
grad_step = 000062, loss = 0.047704
grad_step = 000063, loss = 0.045556
grad_step = 000064, loss = 0.043510
grad_step = 000065, loss = 0.041556
grad_step = 000066, loss = 0.039666
grad_step = 000067, loss = 0.037843
grad_step = 000068, loss = 0.036104
grad_step = 000069, loss = 0.034436
grad_step = 000070, loss = 0.032830
grad_step = 000071, loss = 0.031290
grad_step = 000072, loss = 0.029822
grad_step = 000073, loss = 0.028412
grad_step = 000074, loss = 0.027063
grad_step = 000075, loss = 0.025774
grad_step = 000076, loss = 0.024540
grad_step = 000077, loss = 0.023354
grad_step = 000078, loss = 0.022228
grad_step = 000079, loss = 0.021158
grad_step = 000080, loss = 0.020131
grad_step = 000081, loss = 0.019147
grad_step = 000082, loss = 0.018212
grad_step = 000083, loss = 0.017325
grad_step = 000084, loss = 0.016475
grad_step = 000085, loss = 0.015666
grad_step = 000086, loss = 0.014899
grad_step = 000087, loss = 0.014167
grad_step = 000088, loss = 0.013472
grad_step = 000089, loss = 0.012810
grad_step = 000090, loss = 0.012181
grad_step = 000091, loss = 0.011585
grad_step = 000092, loss = 0.011019
grad_step = 000093, loss = 0.010481
grad_step = 000094, loss = 0.009971
grad_step = 000095, loss = 0.009488
grad_step = 000096, loss = 0.009031
grad_step = 000097, loss = 0.008597
grad_step = 000098, loss = 0.008186
grad_step = 000099, loss = 0.007798
grad_step = 000100, loss = 0.007431
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.007085
grad_step = 000102, loss = 0.006757
grad_step = 000103, loss = 0.006448
grad_step = 000104, loss = 0.006156
grad_step = 000105, loss = 0.005881
grad_step = 000106, loss = 0.005622
grad_step = 000107, loss = 0.005378
grad_step = 000108, loss = 0.005148
grad_step = 000109, loss = 0.004932
grad_step = 000110, loss = 0.004730
grad_step = 000111, loss = 0.004544
grad_step = 000112, loss = 0.004378
grad_step = 000113, loss = 0.004239
grad_step = 000114, loss = 0.004096
grad_step = 000115, loss = 0.003932
grad_step = 000116, loss = 0.003752
grad_step = 000117, loss = 0.003631
grad_step = 000118, loss = 0.003542
grad_step = 000119, loss = 0.003416
grad_step = 000120, loss = 0.003284
grad_step = 000121, loss = 0.003192
grad_step = 000122, loss = 0.003119
grad_step = 000123, loss = 0.003065
grad_step = 000124, loss = 0.002934
grad_step = 000125, loss = 0.002867
grad_step = 000126, loss = 0.002806
grad_step = 000127, loss = 0.002737
grad_step = 000128, loss = 0.002671
grad_step = 000129, loss = 0.002612
grad_step = 000130, loss = 0.002572
grad_step = 000131, loss = 0.002526
grad_step = 000132, loss = 0.002472
grad_step = 000133, loss = 0.002430
grad_step = 000134, loss = 0.002392
grad_step = 000135, loss = 0.002362
grad_step = 000136, loss = 0.002332
grad_step = 000137, loss = 0.002297
grad_step = 000138, loss = 0.002268
grad_step = 000139, loss = 0.002240
grad_step = 000140, loss = 0.002219
grad_step = 000141, loss = 0.002200
grad_step = 000142, loss = 0.002180
grad_step = 000143, loss = 0.002163
grad_step = 000144, loss = 0.002144
grad_step = 000145, loss = 0.002127
grad_step = 000146, loss = 0.002111
grad_step = 000147, loss = 0.002095
grad_step = 000148, loss = 0.002083
grad_step = 000149, loss = 0.002070
grad_step = 000150, loss = 0.002059
grad_step = 000151, loss = 0.002050
grad_step = 000152, loss = 0.002042
grad_step = 000153, loss = 0.002041
grad_step = 000154, loss = 0.002050
grad_step = 000155, loss = 0.002092
grad_step = 000156, loss = 0.002164
grad_step = 000157, loss = 0.002249
grad_step = 000158, loss = 0.002104
grad_step = 000159, loss = 0.001988
grad_step = 000160, loss = 0.002051
grad_step = 000161, loss = 0.002082
grad_step = 000162, loss = 0.002002
grad_step = 000163, loss = 0.001980
grad_step = 000164, loss = 0.002036
grad_step = 000165, loss = 0.001999
grad_step = 000166, loss = 0.001956
grad_step = 000167, loss = 0.002004
grad_step = 000168, loss = 0.001988
grad_step = 000169, loss = 0.001941
grad_step = 000170, loss = 0.001978
grad_step = 000171, loss = 0.001972
grad_step = 000172, loss = 0.001930
grad_step = 000173, loss = 0.001957
grad_step = 000174, loss = 0.001955
grad_step = 000175, loss = 0.001918
grad_step = 000176, loss = 0.001938
grad_step = 000177, loss = 0.001939
grad_step = 000178, loss = 0.001907
grad_step = 000179, loss = 0.001920
grad_step = 000180, loss = 0.001923
grad_step = 000181, loss = 0.001896
grad_step = 000182, loss = 0.001903
grad_step = 000183, loss = 0.001908
grad_step = 000184, loss = 0.001886
grad_step = 000185, loss = 0.001886
grad_step = 000186, loss = 0.001892
grad_step = 000187, loss = 0.001875
grad_step = 000188, loss = 0.001870
grad_step = 000189, loss = 0.001875
grad_step = 000190, loss = 0.001865
grad_step = 000191, loss = 0.001855
grad_step = 000192, loss = 0.001857
grad_step = 000193, loss = 0.001854
grad_step = 000194, loss = 0.001843
grad_step = 000195, loss = 0.001839
grad_step = 000196, loss = 0.001839
grad_step = 000197, loss = 0.001833
grad_step = 000198, loss = 0.001824
grad_step = 000199, loss = 0.001821
grad_step = 000200, loss = 0.001819
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001814
grad_step = 000202, loss = 0.001806
grad_step = 000203, loss = 0.001801
grad_step = 000204, loss = 0.001798
grad_step = 000205, loss = 0.001794
grad_step = 000206, loss = 0.001788
grad_step = 000207, loss = 0.001782
grad_step = 000208, loss = 0.001776
grad_step = 000209, loss = 0.001771
grad_step = 000210, loss = 0.001767
grad_step = 000211, loss = 0.001763
grad_step = 000212, loss = 0.001759
grad_step = 000213, loss = 0.001753
grad_step = 000214, loss = 0.001748
grad_step = 000215, loss = 0.001743
grad_step = 000216, loss = 0.001738
grad_step = 000217, loss = 0.001734
grad_step = 000218, loss = 0.001731
grad_step = 000219, loss = 0.001731
grad_step = 000220, loss = 0.001737
grad_step = 000221, loss = 0.001760
grad_step = 000222, loss = 0.001815
grad_step = 000223, loss = 0.001854
grad_step = 000224, loss = 0.001882
grad_step = 000225, loss = 0.001764
grad_step = 000226, loss = 0.001688
grad_step = 000227, loss = 0.001721
grad_step = 000228, loss = 0.001757
grad_step = 000229, loss = 0.001721
grad_step = 000230, loss = 0.001668
grad_step = 000231, loss = 0.001697
grad_step = 000232, loss = 0.001729
grad_step = 000233, loss = 0.001678
grad_step = 000234, loss = 0.001650
grad_step = 000235, loss = 0.001677
grad_step = 000236, loss = 0.001681
grad_step = 000237, loss = 0.001648
grad_step = 000238, loss = 0.001633
grad_step = 000239, loss = 0.001650
grad_step = 000240, loss = 0.001650
grad_step = 000241, loss = 0.001624
grad_step = 000242, loss = 0.001617
grad_step = 000243, loss = 0.001629
grad_step = 000244, loss = 0.001622
grad_step = 000245, loss = 0.001604
grad_step = 000246, loss = 0.001600
grad_step = 000247, loss = 0.001605
grad_step = 000248, loss = 0.001602
grad_step = 000249, loss = 0.001589
grad_step = 000250, loss = 0.001581
grad_step = 000251, loss = 0.001583
grad_step = 000252, loss = 0.001583
grad_step = 000253, loss = 0.001576
grad_step = 000254, loss = 0.001566
grad_step = 000255, loss = 0.001560
grad_step = 000256, loss = 0.001559
grad_step = 000257, loss = 0.001558
grad_step = 000258, loss = 0.001554
grad_step = 000259, loss = 0.001547
grad_step = 000260, loss = 0.001540
grad_step = 000261, loss = 0.001535
grad_step = 000262, loss = 0.001532
grad_step = 000263, loss = 0.001529
grad_step = 000264, loss = 0.001527
grad_step = 000265, loss = 0.001523
grad_step = 000266, loss = 0.001518
grad_step = 000267, loss = 0.001512
grad_step = 000268, loss = 0.001507
grad_step = 000269, loss = 0.001502
grad_step = 000270, loss = 0.001496
grad_step = 000271, loss = 0.001491
grad_step = 000272, loss = 0.001487
grad_step = 000273, loss = 0.001484
grad_step = 000274, loss = 0.001481
grad_step = 000275, loss = 0.001480
grad_step = 000276, loss = 0.001481
grad_step = 000277, loss = 0.001488
grad_step = 000278, loss = 0.001500
grad_step = 000279, loss = 0.001523
grad_step = 000280, loss = 0.001544
grad_step = 000281, loss = 0.001554
grad_step = 000282, loss = 0.001519
grad_step = 000283, loss = 0.001466
grad_step = 000284, loss = 0.001437
grad_step = 000285, loss = 0.001451
grad_step = 000286, loss = 0.001480
grad_step = 000287, loss = 0.001476
grad_step = 000288, loss = 0.001444
grad_step = 000289, loss = 0.001420
grad_step = 000290, loss = 0.001429
grad_step = 000291, loss = 0.001447
grad_step = 000292, loss = 0.001444
grad_step = 000293, loss = 0.001427
grad_step = 000294, loss = 0.001415
grad_step = 000295, loss = 0.001410
grad_step = 000296, loss = 0.001412
grad_step = 000297, loss = 0.001416
grad_step = 000298, loss = 0.001416
grad_step = 000299, loss = 0.001413
grad_step = 000300, loss = 0.001404
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001397
grad_step = 000302, loss = 0.001392
grad_step = 000303, loss = 0.001392
grad_step = 000304, loss = 0.001397
grad_step = 000305, loss = 0.001401
grad_step = 000306, loss = 0.001401
grad_step = 000307, loss = 0.001400
grad_step = 000308, loss = 0.001400
grad_step = 000309, loss = 0.001399
grad_step = 000310, loss = 0.001395
grad_step = 000311, loss = 0.001389
grad_step = 000312, loss = 0.001384
grad_step = 000313, loss = 0.001380
grad_step = 000314, loss = 0.001375
grad_step = 000315, loss = 0.001372
grad_step = 000316, loss = 0.001370
grad_step = 000317, loss = 0.001368
grad_step = 000318, loss = 0.001365
grad_step = 000319, loss = 0.001362
grad_step = 000320, loss = 0.001361
grad_step = 000321, loss = 0.001360
grad_step = 000322, loss = 0.001359
grad_step = 000323, loss = 0.001360
grad_step = 000324, loss = 0.001361
grad_step = 000325, loss = 0.001369
grad_step = 000326, loss = 0.001393
grad_step = 000327, loss = 0.001450
grad_step = 000328, loss = 0.001574
grad_step = 000329, loss = 0.001716
grad_step = 000330, loss = 0.001689
grad_step = 000331, loss = 0.001514
grad_step = 000332, loss = 0.001371
grad_step = 000333, loss = 0.001474
grad_step = 000334, loss = 0.001525
grad_step = 000335, loss = 0.001399
grad_step = 000336, loss = 0.001394
grad_step = 000337, loss = 0.001447
grad_step = 000338, loss = 0.001425
grad_step = 000339, loss = 0.001369
grad_step = 000340, loss = 0.001400
grad_step = 000341, loss = 0.001433
grad_step = 000342, loss = 0.001347
grad_step = 000343, loss = 0.001389
grad_step = 000344, loss = 0.001411
grad_step = 000345, loss = 0.001353
grad_step = 000346, loss = 0.001374
grad_step = 000347, loss = 0.001374
grad_step = 000348, loss = 0.001376
grad_step = 000349, loss = 0.001343
grad_step = 000350, loss = 0.001351
grad_step = 000351, loss = 0.001377
grad_step = 000352, loss = 0.001329
grad_step = 000353, loss = 0.001339
grad_step = 000354, loss = 0.001354
grad_step = 000355, loss = 0.001331
grad_step = 000356, loss = 0.001334
grad_step = 000357, loss = 0.001325
grad_step = 000358, loss = 0.001330
grad_step = 000359, loss = 0.001329
grad_step = 000360, loss = 0.001313
grad_step = 000361, loss = 0.001319
grad_step = 000362, loss = 0.001322
grad_step = 000363, loss = 0.001309
grad_step = 000364, loss = 0.001312
grad_step = 000365, loss = 0.001309
grad_step = 000366, loss = 0.001307
grad_step = 000367, loss = 0.001309
grad_step = 000368, loss = 0.001301
grad_step = 000369, loss = 0.001301
grad_step = 000370, loss = 0.001303
grad_step = 000371, loss = 0.001298
grad_step = 000372, loss = 0.001297
grad_step = 000373, loss = 0.001296
grad_step = 000374, loss = 0.001293
grad_step = 000375, loss = 0.001294
grad_step = 000376, loss = 0.001292
grad_step = 000377, loss = 0.001288
grad_step = 000378, loss = 0.001289
grad_step = 000379, loss = 0.001287
grad_step = 000380, loss = 0.001285
grad_step = 000381, loss = 0.001285
grad_step = 000382, loss = 0.001282
grad_step = 000383, loss = 0.001280
grad_step = 000384, loss = 0.001280
grad_step = 000385, loss = 0.001279
grad_step = 000386, loss = 0.001277
grad_step = 000387, loss = 0.001276
grad_step = 000388, loss = 0.001275
grad_step = 000389, loss = 0.001273
grad_step = 000390, loss = 0.001272
grad_step = 000391, loss = 0.001271
grad_step = 000392, loss = 0.001269
grad_step = 000393, loss = 0.001268
grad_step = 000394, loss = 0.001267
grad_step = 000395, loss = 0.001265
grad_step = 000396, loss = 0.001264
grad_step = 000397, loss = 0.001263
grad_step = 000398, loss = 0.001261
grad_step = 000399, loss = 0.001260
grad_step = 000400, loss = 0.001259
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001258
grad_step = 000402, loss = 0.001257
grad_step = 000403, loss = 0.001256
grad_step = 000404, loss = 0.001254
grad_step = 000405, loss = 0.001253
grad_step = 000406, loss = 0.001252
grad_step = 000407, loss = 0.001250
grad_step = 000408, loss = 0.001249
grad_step = 000409, loss = 0.001248
grad_step = 000410, loss = 0.001247
grad_step = 000411, loss = 0.001246
grad_step = 000412, loss = 0.001244
grad_step = 000413, loss = 0.001243
grad_step = 000414, loss = 0.001242
grad_step = 000415, loss = 0.001241
grad_step = 000416, loss = 0.001240
grad_step = 000417, loss = 0.001240
grad_step = 000418, loss = 0.001240
grad_step = 000419, loss = 0.001241
grad_step = 000420, loss = 0.001246
grad_step = 000421, loss = 0.001256
grad_step = 000422, loss = 0.001277
grad_step = 000423, loss = 0.001317
grad_step = 000424, loss = 0.001382
grad_step = 000425, loss = 0.001476
grad_step = 000426, loss = 0.001530
grad_step = 000427, loss = 0.001507
grad_step = 000428, loss = 0.001344
grad_step = 000429, loss = 0.001235
grad_step = 000430, loss = 0.001277
grad_step = 000431, loss = 0.001361
grad_step = 000432, loss = 0.001344
grad_step = 000433, loss = 0.001250
grad_step = 000434, loss = 0.001236
grad_step = 000435, loss = 0.001298
grad_step = 000436, loss = 0.001313
grad_step = 000437, loss = 0.001254
grad_step = 000438, loss = 0.001220
grad_step = 000439, loss = 0.001251
grad_step = 000440, loss = 0.001277
grad_step = 000441, loss = 0.001251
grad_step = 000442, loss = 0.001217
grad_step = 000443, loss = 0.001226
grad_step = 000444, loss = 0.001247
grad_step = 000445, loss = 0.001240
grad_step = 000446, loss = 0.001217
grad_step = 000447, loss = 0.001212
grad_step = 000448, loss = 0.001225
grad_step = 000449, loss = 0.001228
grad_step = 000450, loss = 0.001216
grad_step = 000451, loss = 0.001204
grad_step = 000452, loss = 0.001208
grad_step = 000453, loss = 0.001216
grad_step = 000454, loss = 0.001213
grad_step = 000455, loss = 0.001203
grad_step = 000456, loss = 0.001197
grad_step = 000457, loss = 0.001200
grad_step = 000458, loss = 0.001204
grad_step = 000459, loss = 0.001203
grad_step = 000460, loss = 0.001195
grad_step = 000461, loss = 0.001191
grad_step = 000462, loss = 0.001191
grad_step = 000463, loss = 0.001194
grad_step = 000464, loss = 0.001193
grad_step = 000465, loss = 0.001190
grad_step = 000466, loss = 0.001185
grad_step = 000467, loss = 0.001183
grad_step = 000468, loss = 0.001184
grad_step = 000469, loss = 0.001184
grad_step = 000470, loss = 0.001183
grad_step = 000471, loss = 0.001181
grad_step = 000472, loss = 0.001178
grad_step = 000473, loss = 0.001176
grad_step = 000474, loss = 0.001176
grad_step = 000475, loss = 0.001176
grad_step = 000476, loss = 0.001175
grad_step = 000477, loss = 0.001173
grad_step = 000478, loss = 0.001171
grad_step = 000479, loss = 0.001169
grad_step = 000480, loss = 0.001168
grad_step = 000481, loss = 0.001167
grad_step = 000482, loss = 0.001167
grad_step = 000483, loss = 0.001166
grad_step = 000484, loss = 0.001165
grad_step = 000485, loss = 0.001163
grad_step = 000486, loss = 0.001162
grad_step = 000487, loss = 0.001160
grad_step = 000488, loss = 0.001159
grad_step = 000489, loss = 0.001158
grad_step = 000490, loss = 0.001156
grad_step = 000491, loss = 0.001155
grad_step = 000492, loss = 0.001154
grad_step = 000493, loss = 0.001153
grad_step = 000494, loss = 0.001152
grad_step = 000495, loss = 0.001151
grad_step = 000496, loss = 0.001150
grad_step = 000497, loss = 0.001149
grad_step = 000498, loss = 0.001148
grad_step = 000499, loss = 0.001147
grad_step = 000500, loss = 0.001146
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001145
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

  date_run                              2020-05-09 21:12:06.169212
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.28639
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 21:12:06.174200
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    0.2094
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 21:12:06.180691
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.157708
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 21:12:06.185357
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -2.1819
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
0   2020-05-09 21:11:42.060755  ...    mean_absolute_error
1   2020-05-09 21:11:42.064049  ...     mean_squared_error
2   2020-05-09 21:11:42.067011  ...  median_absolute_error
3   2020-05-09 21:11:42.069997  ...               r2_score
4   2020-05-09 21:11:49.626383  ...    mean_absolute_error
5   2020-05-09 21:11:49.629745  ...     mean_squared_error
6   2020-05-09 21:11:49.632322  ...  median_absolute_error
7   2020-05-09 21:11:49.635168  ...               r2_score
8   2020-05-09 21:12:06.169212  ...    mean_absolute_error
9   2020-05-09 21:12:06.174200  ...     mean_squared_error
10  2020-05-09 21:12:06.180691  ...  median_absolute_error
11  2020-05-09 21:12:06.185357  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3842048/9912422 [00:00<00:00, 38270784.01it/s]9920512it [00:00, 36165002.10it/s]                             
0it [00:00, ?it/s]32768it [00:00, 603673.52it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 899407.52it/s]1654784it [00:00, 12610865.29it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191322.16it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d97eee780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d35631c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d97ea4e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d35631da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d97ea4e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d97eeee80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d97eee780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d4a89fcc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d35631da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d4a89fcc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d97ea4e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe904701208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=40a923faa2a53352f25c447d12bdf0b663ade6a99ee2cbeba1bf43790aac20bf
  Stored in directory: /tmp/pip-ephem-wheel-cache-9pvdzvv5/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe89d3e6cc0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1499136/17464789 [=>............................] - ETA: 0s
 3784704/17464789 [=====>........................] - ETA: 0s
10625024/17464789 [=================>............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 21:13:29.446882: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 21:13:29.451285: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095105000 Hz
2020-05-09 21:13:29.451406: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56143de65170 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 21:13:29.451417: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.5133 - accuracy: 0.5100
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5210 - accuracy: 0.5095 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5095 - accuracy: 0.5102
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5695 - accuracy: 0.5063
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6149 - accuracy: 0.5034
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5883 - accuracy: 0.5051
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
11000/25000 [============>.................] - ETA: 3s - loss: 7.5760 - accuracy: 0.5059
12000/25000 [=============>................] - ETA: 2s - loss: 7.6002 - accuracy: 0.5043
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5923 - accuracy: 0.5048
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6151 - accuracy: 0.5034
15000/25000 [=================>............] - ETA: 2s - loss: 7.5961 - accuracy: 0.5046
16000/25000 [==================>...........] - ETA: 1s - loss: 7.5919 - accuracy: 0.5049
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5945 - accuracy: 0.5047
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6096 - accuracy: 0.5037
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6245 - accuracy: 0.5027
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6367 - accuracy: 0.5020
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 6s 249us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 21:13:41.521353
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 21:13:41.521353  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 21:13:46.822029: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 21:13:46.827174: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095105000 Hz
2020-05-09 21:13:46.827312: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562b9fb73630 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 21:13:46.827323: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f09b77f0b70> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2684 - crf_viterbi_accuracy: 0.6533 - val_loss: 1.1751 - val_crf_viterbi_accuracy: 0.6800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f09bef4e0b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4136 - accuracy: 0.5165 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.3702 - accuracy: 0.5193
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4481 - accuracy: 0.5142
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.4182 - accuracy: 0.5162
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.4698 - accuracy: 0.5128
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.4804 - accuracy: 0.5121
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.4960 - accuracy: 0.5111
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5031 - accuracy: 0.5107
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5010 - accuracy: 0.5108
11000/25000 [============>.................] - ETA: 3s - loss: 7.5454 - accuracy: 0.5079
12000/25000 [=============>................] - ETA: 2s - loss: 7.5657 - accuracy: 0.5066
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5463 - accuracy: 0.5078
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5878 - accuracy: 0.5051
15000/25000 [=================>............] - ETA: 2s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6062 - accuracy: 0.5039
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5927 - accuracy: 0.5048
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6019 - accuracy: 0.5042
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6101 - accuracy: 0.5037
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6283 - accuracy: 0.5025
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6257 - accuracy: 0.5027
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6420 - accuracy: 0.5016
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6519 - accuracy: 0.5010
25000/25000 [==============================] - 6s 259us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f094e4c8d68> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:16:20, 9.87kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:13:33, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:06:44, 19.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:29:10, 28.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:55:29, 40.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.92M/862M [00:01<4:07:07, 57.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.5M/862M [00:01<2:52:10, 82.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.7M/862M [00:01<2:00:02, 117kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:01<1:23:36, 167kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:01<58:24, 238kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.3M/862M [00:02<40:47, 340kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:02<28:30, 483kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.9M/862M [00:02<19:56, 687kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<13:59, 974kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:02<09:50, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<07:03, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:04<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:41, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<05:08, 2.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<06:08, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<06:10, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<04:42, 2.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:08<05:50, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<05:43, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<04:25, 3.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<05:46, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<07:00, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<05:31, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<04:05, 3.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<07:05, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<06:22, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<04:48, 2.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<06:22, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<05:46, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<04:19, 3.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<06:07, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<05:38, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<04:16, 3.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<05:33, 2.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<04:13, 3.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<05:59, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<05:31, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<04:11, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<05:29, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:09, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<06:46, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:19, 2.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<03:51, 3.29MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<13:31, 938kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<10:48, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:49, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:25, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:30, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:32, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:41, 2.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<28:45, 436kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<21:24, 586kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<15:16, 819kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<13:35, 918kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:47, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:48, 1.59MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<08:23, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:08, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:18, 2.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:37, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:54, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:23, 2.80MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:00, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:27, 2.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:04, 3.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:45, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:15, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:56, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:38, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:27, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:08, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:32, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<04:57, 2.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<03:43, 3.23MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<02:46, 4.32MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<25:15, 475kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<18:56, 633kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:29, 887kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<12:11, 978kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<09:45, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:05, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:44, 1.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:39, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:56, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<06:14, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:35, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:12, 2.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:42, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:25, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:05, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:26, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:00, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:47, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:22, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:08, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:54, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:34, 3.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:24:36, 136kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:00:25, 191kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<42:26, 271kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<32:18, 354kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<24:58, 459kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<18:02, 634kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<12:41, 897kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<56:16, 202kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<40:33, 280kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<28:35, 397kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<22:35, 501kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<16:48, 673kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<12:02, 937kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<11:01, 1.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:53, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:27, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<07:07, 1.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:09, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:33, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:46, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:12, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:55, 2.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:18, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:52, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:41, 2.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:07, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:45, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:36, 3.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:02, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:33, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:59, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:43, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:29, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:16, 3.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:58, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:24, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:24, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<06:15, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:30, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:04, 2.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<05:20, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:50, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<03:37, 2.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:27<04:59, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:36, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<03:26, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:50, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:30, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:22, 3.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:48, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:27, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:20, 3.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:45, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:11, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:13, 2.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:13, 3.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:34, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:15, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:14, 3.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:38, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:18, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:16, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:38, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:25, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:08, 3.22MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:06:40, 151kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<47:42, 211kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<33:34, 299kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<25:44, 389kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<19:03, 525kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<13:31, 738kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:45, 846kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<10:24, 955kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:43, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:31, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:33, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:01, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:46<05:10, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:52, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:08, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:51, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:56, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:28, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:20, 2.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:35, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:18, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:08, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:02, 3.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:31, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:43, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:30, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<02:35, 3.69MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<25:45, 370kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<19:01, 501kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<13:31, 703kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<11:39, 812kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<09:08, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<06:36, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:48, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:43, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:07, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:43, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:53, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:06, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:48, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:15, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:06, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:58, 3.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:58, 1.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:32, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:48, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:28, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:47, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:35, 2.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:35, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:10, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:08, 2.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:17, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<03:47, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<02:49, 3.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:06, 4.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<24:50, 359kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<18:18, 487kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<13:00, 683kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<11:08, 794kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<08:34, 1.03MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:12, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<04:26, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<25:09, 349kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<18:30, 474kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<13:06, 667kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<11:10, 780kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<09:40, 899kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<07:09, 1.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<05:07, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<07:16, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:59, 1.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:22, 1.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:03, 1.69MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:27, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:17, 2.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:17, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:54, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:00, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:41, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:45, 3.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:53, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:35, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:42, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:48, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:32, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:41, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:46, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:30, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<02:37, 3.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:44, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:19, 2.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:32, 3.19MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<01:52, 4.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<21:42, 373kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<16:01, 505kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<11:23, 707kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<09:48, 819kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:41, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:44, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:50, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:35, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:20, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:52, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:53, 2.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:50, 2.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:30, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:39, 2.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:39, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:22, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:31, 3.06MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<01:55, 4.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:22, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:40, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:12, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<03:03, 2.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:59, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:14, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:53, 1.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:50, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:52, 2.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:33, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:19, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:29, 2.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:20, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:52, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:05, 2.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:13, 3.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<2:22:11, 51.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<1:40:12, 73.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<1:10:05, 104kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<50:30, 144kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<36:05, 201kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<25:22, 285kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<19:20, 372kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<15:04, 477kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<10:54, 658kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<07:39, 931kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<29:35, 241kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<21:26, 332kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<15:08, 469kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<12:12, 579kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<09:16, 760kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:39, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<06:16, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:06, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:44, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:13, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:41, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:45, 2.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:30, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:51, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:03, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:12, 3.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<49:54, 136kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<35:36, 190kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<24:58, 270kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<18:58, 354kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<13:58, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<09:55, 674kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<08:26, 787kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<07:20, 906kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:28, 1.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<03:53, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<49:34, 133kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<35:21, 186kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<24:48, 264kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<18:47, 347kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<14:32, 448kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<10:27, 622kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<07:22, 876kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<07:43, 835kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<06:04, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:24, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:33, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:51, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:50, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:26, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:46, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:56, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:06, 2.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<14:33, 429kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<10:49, 576kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<07:41, 808kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<06:48, 907kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:48, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:35, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:21, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:40, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:12, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:23, 2.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:04, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:05, 2.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<03:13, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:30, 2.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:50, 3.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:36, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:08, 1.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:20, 2.50MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:59, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:42, 2.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:02, 2.84MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:45, 2.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:31, 2.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:53, 3.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:38, 2.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:25, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:50, 3.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:24, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:49, 3.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:33, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:22, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:48, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:31, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:20, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:46, 3.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:29, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:18, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<01:45, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:27, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:16, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:43, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:26, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:13, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:35, 3.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<26:23, 198kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<18:59, 274kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<13:22, 388kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<10:29, 491kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<07:52, 653kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:36, 910kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:04, 999kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:05, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:13, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:47, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:03, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:34, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:18, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:44, 2.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:20, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:35, 3.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:14, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:36, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:02, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:29, 3.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:02, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:37, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:56, 2.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:25, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:06, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:34, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:09, 3.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<12:24, 371kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<09:40, 475kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<06:59, 655kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:18<04:54, 924kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<35:37, 127kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<25:22, 178kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<17:46, 253kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<13:22, 333kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<09:49, 453kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<06:55, 639kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<05:50, 751kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:32, 964kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<03:16, 1.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:16, 1.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:10, 1.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:24, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:43, 2.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:18, 1.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:45, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:00, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:22, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:05, 2.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:33, 2.66MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:02, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:51, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:24, 2.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:54, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:45, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:19, 3.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:42, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:17, 3.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:47, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:05, 1.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:38, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:11, 3.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:04, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:32, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<01:52, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:10, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:20, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:49, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:18, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<14:15, 260kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<10:20, 357kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<07:16, 505kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<05:53, 617kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<04:29, 807kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:13, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:04, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:31, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:50, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:05, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:49, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:21, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:44, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:55, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:31, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:05, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<24:42, 136kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<17:36, 190kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<12:19, 270kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<09:17, 354kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<07:10, 458kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<05:10, 633kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<03:36, 894kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<25:30, 126kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<18:09, 177kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<12:40, 252kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<09:30, 332kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<06:58, 451kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<04:55, 635kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<04:07, 748kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<03:12, 961kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:18, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:17, 1.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:55, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:24, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:39, 1.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:27, 2.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:05, 2.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:25, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:18, 2.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<00:57, 2.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:19, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<00:54, 3.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:16, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:28, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:10, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:50, 3.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<10:13, 261kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<07:25, 359kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<05:13, 507kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<04:12, 620kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:12, 810kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<02:17, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:10, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:44, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:17, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<00:55, 2.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<06:36, 374kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:52, 506kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<03:26, 709kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:55, 820kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:17, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:39, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:40, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:40, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:17, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:54, 2.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<16:49, 135kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<11:58, 188kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<08:20, 267kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<06:15, 351kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:35, 476kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<03:14, 669kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:43, 780kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:20, 904kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:54, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:22, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:24, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:11, 1.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:52, 2.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:04, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:57, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:42, 2.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:56, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:51, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:38, 2.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:52, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:48, 2.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:36, 3.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:45, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:34, 3.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:47, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:55, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:43, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:30, 3.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<12:05, 136kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<08:34, 191kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<05:56, 272kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<04:05, 386kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<06:49, 231kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:06, 308kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:37, 432kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:30, 610kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<02:13, 679kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:42, 881kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:12, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:09, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:57, 1.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:41, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:50, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:27, 2.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<09:35, 136kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<06:48, 190kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<04:42, 270kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<03:29, 354kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<02:33, 480kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:46, 676kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<01:13, 953kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<05:28, 213kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<03:55, 295kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<02:43, 417kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<02:06, 523kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:34, 693kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:06, 968kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:59, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:47, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:31, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:22, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:18, 2.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:21, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:15, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:24, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:18, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:13, 3.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:27, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:23, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:16, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:12, 2.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:10, 3.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:08, 3.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:06, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:03, 3.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:47, 263kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:33, 361kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:20, 510kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:13, 622kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 812kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:05, 1.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:03, 1.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.93MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.68MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1017/400000 [00:00<00:39, 10164.68it/s]  1%|          | 2029/400000 [00:00<00:39, 10150.03it/s]  1%|          | 3090/400000 [00:00<00:38, 10282.30it/s]  1%|          | 4113/400000 [00:00<00:38, 10263.73it/s]  1%|▏         | 5084/400000 [00:00<00:39, 10090.24it/s]  2%|▏         | 6120/400000 [00:00<00:38, 10169.17it/s]  2%|▏         | 7091/400000 [00:00<00:39, 10025.04it/s]  2%|▏         | 8011/400000 [00:00<00:40, 9756.01it/s]   2%|▏         | 9001/400000 [00:00<00:39, 9797.06it/s]  2%|▏         | 9942/400000 [00:01<00:40, 9666.38it/s]  3%|▎         | 11029/400000 [00:01<00:38, 9997.66it/s]  3%|▎         | 12055/400000 [00:01<00:38, 10074.34it/s]  3%|▎         | 13062/400000 [00:01<00:38, 10072.79it/s]  4%|▎         | 14114/400000 [00:01<00:37, 10200.21it/s]  4%|▍         | 15129/400000 [00:01<00:37, 10184.24it/s]  4%|▍         | 16179/400000 [00:01<00:37, 10275.56it/s]  4%|▍         | 17205/400000 [00:01<00:37, 10168.63it/s]  5%|▍         | 18246/400000 [00:01<00:37, 10238.98it/s]  5%|▍         | 19300/400000 [00:01<00:36, 10327.08it/s]  5%|▌         | 20333/400000 [00:02<00:36, 10286.55it/s]  5%|▌         | 21394/400000 [00:02<00:36, 10380.37it/s]  6%|▌         | 22459/400000 [00:02<00:36, 10458.29it/s]  6%|▌         | 23505/400000 [00:02<00:36, 10452.72it/s]  6%|▌         | 24551/400000 [00:02<00:36, 10375.05it/s]  6%|▋         | 25589/400000 [00:02<00:36, 10132.76it/s]  7%|▋         | 26610/400000 [00:02<00:36, 10154.83it/s]  7%|▋         | 27659/400000 [00:02<00:36, 10252.99it/s]  7%|▋         | 28686/400000 [00:02<00:36, 10170.36it/s]  7%|▋         | 29704/400000 [00:02<00:36, 10021.77it/s]  8%|▊         | 30725/400000 [00:03<00:36, 10076.35it/s]  8%|▊         | 31777/400000 [00:03<00:36, 10203.45it/s]  8%|▊         | 32828/400000 [00:03<00:35, 10292.11it/s]  8%|▊         | 33859/400000 [00:03<00:35, 10172.02it/s]  9%|▊         | 34878/400000 [00:03<00:35, 10170.07it/s]  9%|▉         | 35896/400000 [00:03<00:36, 10082.71it/s]  9%|▉         | 36952/400000 [00:03<00:35, 10219.35it/s]  9%|▉         | 37983/400000 [00:03<00:35, 10245.60it/s] 10%|▉         | 39052/400000 [00:03<00:34, 10374.24it/s] 10%|█         | 40091/400000 [00:03<00:35, 10181.42it/s] 10%|█         | 41111/400000 [00:04<00:36, 9826.60it/s]  11%|█         | 42098/400000 [00:04<00:37, 9628.96it/s] 11%|█         | 43103/400000 [00:04<00:36, 9749.58it/s] 11%|█         | 44147/400000 [00:04<00:35, 9945.83it/s] 11%|█▏        | 45154/400000 [00:04<00:35, 9976.29it/s] 12%|█▏        | 46154/400000 [00:04<00:35, 9929.16it/s] 12%|█▏        | 47156/400000 [00:04<00:35, 9954.05it/s] 12%|█▏        | 48153/400000 [00:04<00:35, 9831.64it/s] 12%|█▏        | 49138/400000 [00:04<00:36, 9660.06it/s] 13%|█▎        | 50106/400000 [00:04<00:36, 9544.72it/s] 13%|█▎        | 51082/400000 [00:05<00:36, 9605.99it/s] 13%|█▎        | 52044/400000 [00:05<00:36, 9428.31it/s] 13%|█▎        | 53046/400000 [00:05<00:36, 9595.98it/s] 14%|█▎        | 54054/400000 [00:05<00:35, 9733.94it/s] 14%|█▍        | 55056/400000 [00:05<00:35, 9816.29it/s] 14%|█▍        | 56053/400000 [00:05<00:34, 9860.49it/s] 14%|█▍        | 57102/400000 [00:05<00:34, 10041.10it/s] 15%|█▍        | 58108/400000 [00:05<00:35, 9626.26it/s]  15%|█▍        | 59076/400000 [00:05<00:36, 9465.48it/s] 15%|█▌        | 60027/400000 [00:06<00:36, 9335.05it/s] 15%|█▌        | 61029/400000 [00:06<00:35, 9529.38it/s] 16%|█▌        | 62098/400000 [00:06<00:34, 9847.17it/s] 16%|█▌        | 63150/400000 [00:06<00:33, 10039.80it/s] 16%|█▌        | 64159/400000 [00:06<00:33, 9921.64it/s]  16%|█▋        | 65155/400000 [00:06<00:33, 9931.66it/s] 17%|█▋        | 66151/400000 [00:06<00:33, 9865.33it/s] 17%|█▋        | 67197/400000 [00:06<00:33, 10034.49it/s] 17%|█▋        | 68209/400000 [00:06<00:32, 10059.02it/s] 17%|█▋        | 69249/400000 [00:06<00:32, 10158.43it/s] 18%|█▊        | 70267/400000 [00:07<00:32, 10094.85it/s] 18%|█▊        | 71278/400000 [00:07<00:33, 9841.17it/s]  18%|█▊        | 72265/400000 [00:07<00:33, 9832.69it/s] 18%|█▊        | 73286/400000 [00:07<00:32, 9942.47it/s] 19%|█▊        | 74282/400000 [00:07<00:33, 9782.01it/s] 19%|█▉        | 75262/400000 [00:07<00:33, 9759.48it/s] 19%|█▉        | 76304/400000 [00:07<00:32, 9946.60it/s] 19%|█▉        | 77315/400000 [00:07<00:32, 9991.85it/s] 20%|█▉        | 78375/400000 [00:07<00:31, 10164.74it/s] 20%|█▉        | 79436/400000 [00:07<00:31, 10292.49it/s] 20%|██        | 80467/400000 [00:08<00:31, 10171.60it/s] 20%|██        | 81565/400000 [00:08<00:30, 10399.53it/s] 21%|██        | 82644/400000 [00:08<00:30, 10511.78it/s] 21%|██        | 83698/400000 [00:08<00:30, 10465.61it/s] 21%|██        | 84774/400000 [00:08<00:29, 10552.09it/s] 21%|██▏       | 85831/400000 [00:08<00:30, 10400.37it/s] 22%|██▏       | 86873/400000 [00:08<00:31, 9968.13it/s]  22%|██▏       | 87909/400000 [00:08<00:30, 10079.68it/s] 22%|██▏       | 88942/400000 [00:08<00:30, 10152.81it/s] 23%|██▎       | 90005/400000 [00:08<00:30, 10289.41it/s] 23%|██▎       | 91037/400000 [00:09<00:30, 10131.78it/s] 23%|██▎       | 92053/400000 [00:09<00:31, 9884.16it/s]  23%|██▎       | 93088/400000 [00:09<00:30, 10018.73it/s] 24%|██▎       | 94174/400000 [00:09<00:29, 10256.38it/s] 24%|██▍       | 95203/400000 [00:09<00:30, 10066.72it/s] 24%|██▍       | 96213/400000 [00:09<00:30, 10013.35it/s] 24%|██▍       | 97235/400000 [00:09<00:30, 10072.91it/s] 25%|██▍       | 98280/400000 [00:09<00:29, 10183.07it/s] 25%|██▍       | 99303/400000 [00:09<00:29, 10194.77it/s] 25%|██▌       | 100324/400000 [00:09<00:30, 9986.10it/s] 25%|██▌       | 101330/400000 [00:10<00:29, 10005.79it/s] 26%|██▌       | 102332/400000 [00:10<00:30, 9875.41it/s]  26%|██▌       | 103374/400000 [00:10<00:29, 10031.25it/s] 26%|██▌       | 104445/400000 [00:10<00:28, 10223.04it/s] 26%|██▋       | 105470/400000 [00:10<00:28, 10182.80it/s] 27%|██▋       | 106490/400000 [00:10<00:29, 10018.70it/s] 27%|██▋       | 107494/400000 [00:10<00:29, 10016.95it/s] 27%|██▋       | 108497/400000 [00:10<00:29, 9720.85it/s]  27%|██▋       | 109472/400000 [00:10<00:29, 9722.89it/s] 28%|██▊       | 110501/400000 [00:11<00:29, 9884.03it/s] 28%|██▊       | 111493/400000 [00:11<00:29, 9891.98it/s] 28%|██▊       | 112500/400000 [00:11<00:28, 9941.83it/s] 28%|██▊       | 113544/400000 [00:11<00:28, 10085.82it/s] 29%|██▊       | 114609/400000 [00:11<00:27, 10248.56it/s] 29%|██▉       | 115636/400000 [00:11<00:28, 10061.82it/s] 29%|██▉       | 116644/400000 [00:11<00:28, 9916.08it/s]  29%|██▉       | 117638/400000 [00:11<00:28, 9798.36it/s] 30%|██▉       | 118620/400000 [00:11<00:29, 9620.46it/s] 30%|██▉       | 119643/400000 [00:11<00:28, 9794.15it/s] 30%|███       | 120634/400000 [00:12<00:28, 9827.56it/s] 30%|███       | 121619/400000 [00:12<00:28, 9626.35it/s] 31%|███       | 122614/400000 [00:12<00:28, 9719.84it/s] 31%|███       | 123646/400000 [00:12<00:27, 9891.40it/s] 31%|███       | 124637/400000 [00:12<00:27, 9867.00it/s] 31%|███▏      | 125625/400000 [00:12<00:28, 9795.29it/s] 32%|███▏      | 126611/400000 [00:12<00:27, 9813.54it/s] 32%|███▏      | 127605/400000 [00:12<00:27, 9850.68it/s] 32%|███▏      | 128620/400000 [00:12<00:27, 9936.67it/s] 32%|███▏      | 129615/400000 [00:12<00:28, 9556.93it/s] 33%|███▎      | 130575/400000 [00:13<00:29, 9242.56it/s] 33%|███▎      | 131504/400000 [00:13<00:30, 8875.20it/s] 33%|███▎      | 132429/400000 [00:13<00:29, 8981.61it/s] 33%|███▎      | 133362/400000 [00:13<00:29, 9083.15it/s] 34%|███▎      | 134274/400000 [00:13<00:29, 9071.18it/s] 34%|███▍      | 135269/400000 [00:13<00:28, 9317.57it/s] 34%|███▍      | 136206/400000 [00:13<00:28, 9330.97it/s] 34%|███▍      | 137172/400000 [00:13<00:27, 9427.07it/s] 35%|███▍      | 138175/400000 [00:13<00:27, 9598.77it/s] 35%|███▍      | 139187/400000 [00:13<00:26, 9746.54it/s] 35%|███▌      | 140164/400000 [00:14<00:26, 9710.29it/s] 35%|███▌      | 141137/400000 [00:14<00:26, 9656.72it/s] 36%|███▌      | 142104/400000 [00:14<00:27, 9542.42it/s] 36%|███▌      | 143060/400000 [00:14<00:27, 9437.83it/s] 36%|███▌      | 144005/400000 [00:14<00:27, 9369.76it/s] 36%|███▌      | 144943/400000 [00:14<00:27, 9332.74it/s] 36%|███▋      | 145877/400000 [00:14<00:27, 9133.21it/s] 37%|███▋      | 146794/400000 [00:14<00:27, 9142.07it/s] 37%|███▋      | 147751/400000 [00:14<00:27, 9265.39it/s] 37%|███▋      | 148694/400000 [00:15<00:26, 9309.95it/s] 37%|███▋      | 149655/400000 [00:15<00:26, 9393.25it/s] 38%|███▊      | 150697/400000 [00:15<00:25, 9677.29it/s] 38%|███▊      | 151736/400000 [00:15<00:25, 9880.10it/s] 38%|███▊      | 152727/400000 [00:15<00:25, 9821.80it/s] 38%|███▊      | 153712/400000 [00:15<00:25, 9701.76it/s] 39%|███▊      | 154684/400000 [00:15<00:25, 9505.00it/s] 39%|███▉      | 155637/400000 [00:15<00:26, 9355.45it/s] 39%|███▉      | 156575/400000 [00:15<00:26, 9293.02it/s] 39%|███▉      | 157512/400000 [00:15<00:26, 9313.48it/s] 40%|███▉      | 158496/400000 [00:16<00:25, 9464.48it/s] 40%|███▉      | 159479/400000 [00:16<00:25, 9569.83it/s] 40%|████      | 160438/400000 [00:16<00:25, 9546.58it/s] 40%|████      | 161394/400000 [00:16<00:25, 9471.88it/s] 41%|████      | 162349/400000 [00:16<00:25, 9494.99it/s] 41%|████      | 163363/400000 [00:16<00:24, 9677.02it/s] 41%|████      | 164337/400000 [00:16<00:24, 9694.60it/s] 41%|████▏     | 165308/400000 [00:16<00:24, 9647.96it/s] 42%|████▏     | 166336/400000 [00:16<00:23, 9827.96it/s] 42%|████▏     | 167326/400000 [00:16<00:23, 9848.75it/s] 42%|████▏     | 168328/400000 [00:17<00:23, 9897.69it/s] 42%|████▏     | 169371/400000 [00:17<00:22, 10051.24it/s] 43%|████▎     | 170378/400000 [00:17<00:23, 9963.92it/s]  43%|████▎     | 171395/400000 [00:17<00:22, 10022.72it/s] 43%|████▎     | 172422/400000 [00:17<00:22, 10094.38it/s] 43%|████▎     | 173433/400000 [00:17<00:22, 10062.66it/s] 44%|████▎     | 174463/400000 [00:17<00:22, 10130.56it/s] 44%|████▍     | 175477/400000 [00:17<00:22, 9959.06it/s]  44%|████▍     | 176488/400000 [00:17<00:22, 10001.72it/s] 44%|████▍     | 177489/400000 [00:17<00:22, 9900.01it/s]  45%|████▍     | 178480/400000 [00:18<00:22, 9779.68it/s] 45%|████▍     | 179531/400000 [00:18<00:22, 9987.86it/s] 45%|████▌     | 180532/400000 [00:18<00:22, 9864.77it/s] 45%|████▌     | 181556/400000 [00:18<00:21, 9972.81it/s] 46%|████▌     | 182562/400000 [00:18<00:21, 9998.79it/s] 46%|████▌     | 183589/400000 [00:18<00:21, 10078.31it/s] 46%|████▌     | 184642/400000 [00:18<00:21, 10207.87it/s] 46%|████▋     | 185664/400000 [00:18<00:21, 10107.91it/s] 47%|████▋     | 186676/400000 [00:18<00:21, 10035.01it/s] 47%|████▋     | 187720/400000 [00:18<00:20, 10150.43it/s] 47%|████▋     | 188736/400000 [00:19<00:21, 9795.43it/s]  47%|████▋     | 189719/400000 [00:19<00:21, 9594.10it/s] 48%|████▊     | 190682/400000 [00:19<00:22, 9399.16it/s] 48%|████▊     | 191680/400000 [00:19<00:21, 9562.51it/s] 48%|████▊     | 192674/400000 [00:19<00:21, 9670.94it/s] 48%|████▊     | 193664/400000 [00:19<00:21, 9738.52it/s] 49%|████▊     | 194640/400000 [00:19<00:21, 9729.34it/s] 49%|████▉     | 195642/400000 [00:19<00:20, 9814.45it/s] 49%|████▉     | 196701/400000 [00:19<00:20, 10034.74it/s] 49%|████▉     | 197707/400000 [00:19<00:20, 10042.15it/s] 50%|████▉     | 198725/400000 [00:20<00:19, 10081.80it/s] 50%|████▉     | 199778/400000 [00:20<00:19, 10209.11it/s] 50%|█████     | 200800/400000 [00:20<00:19, 10041.60it/s] 50%|█████     | 201808/400000 [00:20<00:19, 10050.07it/s] 51%|█████     | 202817/400000 [00:20<00:19, 10059.81it/s] 51%|█████     | 203824/400000 [00:20<00:19, 9956.26it/s]  51%|█████     | 204821/400000 [00:20<00:20, 9692.77it/s] 51%|█████▏    | 205793/400000 [00:20<00:20, 9273.88it/s] 52%|█████▏    | 206726/400000 [00:20<00:21, 9177.79it/s] 52%|█████▏    | 207710/400000 [00:21<00:20, 9366.36it/s] 52%|█████▏    | 208712/400000 [00:21<00:20, 9551.49it/s] 52%|█████▏    | 209671/400000 [00:21<00:20, 9446.84it/s] 53%|█████▎    | 210629/400000 [00:21<00:19, 9485.78it/s] 53%|█████▎    | 211665/400000 [00:21<00:19, 9731.90it/s] 53%|█████▎    | 212673/400000 [00:21<00:19, 9832.44it/s] 53%|█████▎    | 213739/400000 [00:21<00:18, 10066.57it/s] 54%|█████▎    | 214749/400000 [00:21<00:18, 10036.10it/s] 54%|█████▍    | 215755/400000 [00:21<00:18, 9916.46it/s]  54%|█████▍    | 216771/400000 [00:21<00:18, 9986.80it/s] 54%|█████▍    | 217772/400000 [00:22<00:18, 9989.86it/s] 55%|█████▍    | 218807/400000 [00:22<00:17, 10094.96it/s] 55%|█████▍    | 219818/400000 [00:22<00:17, 10035.28it/s] 55%|█████▌    | 220823/400000 [00:22<00:18, 9744.26it/s]  55%|█████▌    | 221840/400000 [00:22<00:18, 9867.18it/s] 56%|█████▌    | 222829/400000 [00:22<00:18, 9693.92it/s] 56%|█████▌    | 223801/400000 [00:22<00:18, 9486.70it/s] 56%|█████▌    | 224753/400000 [00:22<00:18, 9487.42it/s] 56%|█████▋    | 225724/400000 [00:22<00:18, 9551.48it/s] 57%|█████▋    | 226681/400000 [00:22<00:18, 9495.42it/s] 57%|█████▋    | 227710/400000 [00:23<00:17, 9720.29it/s] 57%|█████▋    | 228770/400000 [00:23<00:17, 9966.82it/s] 57%|█████▋    | 229770/400000 [00:23<00:17, 9918.05it/s] 58%|█████▊    | 230794/400000 [00:23<00:16, 10011.25it/s] 58%|█████▊    | 231807/400000 [00:23<00:16, 10045.90it/s] 58%|█████▊    | 232841/400000 [00:23<00:16, 10130.31it/s] 58%|█████▊    | 233879/400000 [00:23<00:16, 10203.01it/s] 59%|█████▊    | 234901/400000 [00:23<00:16, 10007.09it/s] 59%|█████▉    | 235904/400000 [00:23<00:16, 9831.38it/s]  59%|█████▉    | 236902/400000 [00:23<00:16, 9875.19it/s] 59%|█████▉    | 237891/400000 [00:24<00:17, 9515.10it/s] 60%|█████▉    | 238847/400000 [00:24<00:17, 9136.38it/s] 60%|█████▉    | 239767/400000 [00:24<00:17, 8910.92it/s] 60%|██████    | 240664/400000 [00:24<00:18, 8851.05it/s] 60%|██████    | 241639/400000 [00:24<00:17, 9101.62it/s] 61%|██████    | 242597/400000 [00:24<00:17, 9239.42it/s] 61%|██████    | 243583/400000 [00:24<00:16, 9417.05it/s] 61%|██████    | 244603/400000 [00:24<00:16, 9638.84it/s] 61%|██████▏   | 245617/400000 [00:24<00:15, 9781.64it/s] 62%|██████▏   | 246615/400000 [00:25<00:15, 9839.02it/s] 62%|██████▏   | 247619/400000 [00:25<00:15, 9895.97it/s] 62%|██████▏   | 248666/400000 [00:25<00:15, 10058.87it/s] 62%|██████▏   | 249674/400000 [00:25<00:14, 10030.49it/s] 63%|██████▎   | 250679/400000 [00:25<00:14, 10012.56it/s] 63%|██████▎   | 251699/400000 [00:25<00:14, 10066.54it/s] 63%|██████▎   | 252707/400000 [00:25<00:14, 9928.19it/s]  63%|██████▎   | 253701/400000 [00:25<00:15, 9665.79it/s] 64%|██████▎   | 254670/400000 [00:25<00:15, 9437.29it/s] 64%|██████▍   | 255624/400000 [00:25<00:15, 9467.20it/s] 64%|██████▍   | 256620/400000 [00:26<00:14, 9608.80it/s] 64%|██████▍   | 257658/400000 [00:26<00:14, 9826.69it/s] 65%|██████▍   | 258686/400000 [00:26<00:14, 9958.29it/s] 65%|██████▍   | 259696/400000 [00:26<00:14, 9997.42it/s] 65%|██████▌   | 260698/400000 [00:26<00:13, 9987.01it/s] 65%|██████▌   | 261705/400000 [00:26<00:13, 10010.69it/s] 66%|██████▌   | 262707/400000 [00:26<00:13, 9879.12it/s]  66%|██████▌   | 263705/400000 [00:26<00:13, 9908.10it/s] 66%|██████▌   | 264697/400000 [00:26<00:13, 9870.90it/s] 66%|██████▋   | 265728/400000 [00:26<00:13, 9997.06it/s] 67%|██████▋   | 266771/400000 [00:27<00:13, 10120.98it/s] 67%|██████▋   | 267850/400000 [00:27<00:12, 10312.59it/s] 67%|██████▋   | 268883/400000 [00:27<00:12, 10187.41it/s] 67%|██████▋   | 269904/400000 [00:27<00:13, 9823.18it/s]  68%|██████▊   | 270905/400000 [00:27<00:13, 9877.23it/s] 68%|██████▊   | 271934/400000 [00:27<00:12, 9996.33it/s] 68%|██████▊   | 272979/400000 [00:27<00:12, 10127.35it/s] 69%|██████▊   | 274011/400000 [00:27<00:12, 10182.60it/s] 69%|██████▉   | 275031/400000 [00:27<00:12, 10187.25it/s] 69%|██████▉   | 276051/400000 [00:27<00:12, 9889.39it/s]  69%|██████▉   | 277043/400000 [00:28<00:12, 9683.14it/s] 70%|██████▉   | 278118/400000 [00:28<00:12, 9979.71it/s] 70%|██████▉   | 279145/400000 [00:28<00:12, 10064.94it/s] 70%|███████   | 280155/400000 [00:28<00:11, 10072.92it/s] 70%|███████   | 281165/400000 [00:28<00:12, 9807.89it/s]  71%|███████   | 282149/400000 [00:28<00:12, 9771.61it/s] 71%|███████   | 283143/400000 [00:28<00:11, 9819.48it/s] 71%|███████   | 284161/400000 [00:28<00:11, 9923.26it/s] 71%|███████▏  | 285172/400000 [00:28<00:11, 9977.95it/s] 72%|███████▏  | 286171/400000 [00:29<00:11, 9886.61it/s] 72%|███████▏  | 287161/400000 [00:29<00:11, 9591.12it/s] 72%|███████▏  | 288123/400000 [00:29<00:11, 9423.27it/s] 72%|███████▏  | 289135/400000 [00:29<00:11, 9620.71it/s] 73%|███████▎  | 290139/400000 [00:29<00:11, 9739.91it/s] 73%|███████▎  | 291161/400000 [00:29<00:11, 9876.49it/s] 73%|███████▎  | 292153/400000 [00:29<00:10, 9889.07it/s] 73%|███████▎  | 293189/400000 [00:29<00:10, 10025.21it/s] 74%|███████▎  | 294234/400000 [00:29<00:10, 10148.96it/s] 74%|███████▍  | 295251/400000 [00:29<00:10, 10150.26it/s] 74%|███████▍  | 296267/400000 [00:30<00:10, 10048.54it/s] 74%|███████▍  | 297273/400000 [00:30<00:10, 9943.49it/s]  75%|███████▍  | 298290/400000 [00:30<00:10, 10010.16it/s] 75%|███████▍  | 299292/400000 [00:30<00:10, 10000.98it/s] 75%|███████▌  | 300293/400000 [00:30<00:10, 9918.38it/s]  75%|███████▌  | 301286/400000 [00:30<00:10, 9844.09it/s] 76%|███████▌  | 302271/400000 [00:30<00:09, 9839.48it/s] 76%|███████▌  | 303256/400000 [00:30<00:09, 9727.35it/s] 76%|███████▌  | 304230/400000 [00:30<00:10, 9533.17it/s] 76%|███████▋  | 305185/400000 [00:30<00:09, 9496.64it/s] 77%|███████▋  | 306209/400000 [00:31<00:09, 9707.09it/s] 77%|███████▋  | 307219/400000 [00:31<00:09, 9819.89it/s] 77%|███████▋  | 308219/400000 [00:31<00:09, 9872.52it/s] 77%|███████▋  | 309208/400000 [00:31<00:09, 9829.69it/s] 78%|███████▊  | 310192/400000 [00:31<00:09, 9745.78it/s] 78%|███████▊  | 311195/400000 [00:31<00:09, 9829.00it/s] 78%|███████▊  | 312224/400000 [00:31<00:08, 9962.63it/s] 78%|███████▊  | 313222/400000 [00:31<00:08, 9726.18it/s] 79%|███████▊  | 314204/400000 [00:31<00:08, 9752.55it/s] 79%|███████▉  | 315213/400000 [00:31<00:08, 9851.01it/s] 79%|███████▉  | 316244/400000 [00:32<00:08, 9983.27it/s] 79%|███████▉  | 317288/400000 [00:32<00:08, 10115.29it/s] 80%|███████▉  | 318301/400000 [00:32<00:08, 10061.35it/s] 80%|███████▉  | 319309/400000 [00:32<00:08, 10010.69it/s] 80%|████████  | 320311/400000 [00:32<00:08, 9853.40it/s]  80%|████████  | 321328/400000 [00:32<00:07, 9946.29it/s] 81%|████████  | 322387/400000 [00:32<00:07, 10130.04it/s] 81%|████████  | 323433/400000 [00:32<00:07, 10225.82it/s] 81%|████████  | 324486/400000 [00:32<00:07, 10312.32it/s] 81%|████████▏ | 325519/400000 [00:32<00:07, 10183.68it/s] 82%|████████▏ | 326539/400000 [00:33<00:07, 9989.58it/s]  82%|████████▏ | 327578/400000 [00:33<00:07, 10105.22it/s] 82%|████████▏ | 328590/400000 [00:33<00:07, 9998.52it/s]  82%|████████▏ | 329592/400000 [00:33<00:07, 9989.32it/s] 83%|████████▎ | 330609/400000 [00:33<00:06, 10041.55it/s] 83%|████████▎ | 331652/400000 [00:33<00:06, 10154.71it/s] 83%|████████▎ | 332678/400000 [00:33<00:06, 10186.05it/s] 83%|████████▎ | 333734/400000 [00:33<00:06, 10294.28it/s] 84%|████████▎ | 334770/400000 [00:33<00:06, 10311.12it/s] 84%|████████▍ | 335802/400000 [00:33<00:06, 10237.61it/s] 84%|████████▍ | 336827/400000 [00:34<00:06, 9989.73it/s]  84%|████████▍ | 337828/400000 [00:34<00:06, 9732.55it/s] 85%|████████▍ | 338825/400000 [00:34<00:06, 9799.38it/s] 85%|████████▍ | 339807/400000 [00:34<00:06, 9724.61it/s] 85%|████████▌ | 340781/400000 [00:34<00:06, 9620.64it/s] 85%|████████▌ | 341745/400000 [00:34<00:06, 9599.85it/s] 86%|████████▌ | 342721/400000 [00:34<00:05, 9645.95it/s] 86%|████████▌ | 343687/400000 [00:34<00:05, 9645.96it/s] 86%|████████▌ | 344653/400000 [00:34<00:05, 9557.85it/s] 86%|████████▋ | 345626/400000 [00:35<00:05, 9607.06it/s] 87%|████████▋ | 346669/400000 [00:35<00:05, 9838.53it/s] 87%|████████▋ | 347655/400000 [00:35<00:05, 9652.23it/s] 87%|████████▋ | 348684/400000 [00:35<00:05, 9834.28it/s] 87%|████████▋ | 349714/400000 [00:35<00:05, 9968.77it/s] 88%|████████▊ | 350713/400000 [00:35<00:04, 9917.21it/s] 88%|████████▊ | 351709/400000 [00:35<00:04, 9929.58it/s] 88%|████████▊ | 352703/400000 [00:35<00:04, 9754.29it/s] 88%|████████▊ | 353680/400000 [00:35<00:04, 9547.61it/s] 89%|████████▊ | 354675/400000 [00:35<00:04, 9664.42it/s] 89%|████████▉ | 355672/400000 [00:36<00:04, 9750.27it/s] 89%|████████▉ | 356660/400000 [00:36<00:04, 9787.86it/s] 89%|████████▉ | 357730/400000 [00:36<00:04, 10044.44it/s] 90%|████████▉ | 358799/400000 [00:36<00:04, 10228.22it/s] 90%|████████▉ | 359832/400000 [00:36<00:03, 10257.19it/s] 90%|█████████ | 360860/400000 [00:36<00:03, 10225.95it/s] 90%|█████████ | 361934/400000 [00:36<00:03, 10372.54it/s] 91%|█████████ | 362973/400000 [00:36<00:03, 10321.35it/s] 91%|█████████ | 364007/400000 [00:36<00:03, 10323.76it/s] 91%|█████████▏| 365041/400000 [00:36<00:03, 10217.59it/s] 92%|█████████▏| 366064/400000 [00:37<00:03, 10137.71it/s] 92%|█████████▏| 367113/400000 [00:37<00:03, 10239.76it/s] 92%|█████████▏| 368138/400000 [00:37<00:03, 10017.59it/s] 92%|█████████▏| 369142/400000 [00:37<00:03, 9799.26it/s]  93%|█████████▎| 370172/400000 [00:37<00:03, 9942.59it/s] 93%|█████████▎| 371191/400000 [00:37<00:02, 10012.11it/s] 93%|█████████▎| 372207/400000 [00:37<00:02, 10053.84it/s] 93%|█████████▎| 373237/400000 [00:37<00:02, 10125.59it/s] 94%|█████████▎| 374251/400000 [00:37<00:02, 10068.86it/s] 94%|█████████▍| 375259/400000 [00:37<00:02, 10052.36it/s] 94%|█████████▍| 376288/400000 [00:38<00:02, 10122.00it/s] 94%|█████████▍| 377313/400000 [00:38<00:02, 10159.77it/s] 95%|█████████▍| 378330/400000 [00:38<00:02, 9911.62it/s]  95%|█████████▍| 379323/400000 [00:38<00:02, 9894.88it/s] 95%|█████████▌| 380314/400000 [00:38<00:02, 9798.28it/s] 95%|█████████▌| 381295/400000 [00:38<00:01, 9643.15it/s] 96%|█████████▌| 382261/400000 [00:38<00:01, 9642.58it/s] 96%|█████████▌| 383227/400000 [00:38<00:01, 9463.41it/s] 96%|█████████▌| 384175/400000 [00:38<00:01, 9345.37it/s] 96%|█████████▋| 385133/400000 [00:38<00:01, 9412.38it/s] 97%|█████████▋| 386076/400000 [00:39<00:01, 9253.56it/s] 97%|█████████▋| 387016/400000 [00:39<00:01, 9296.46it/s] 97%|█████████▋| 387947/400000 [00:39<00:01, 9258.38it/s] 97%|█████████▋| 388884/400000 [00:39<00:01, 9290.10it/s] 97%|█████████▋| 389821/400000 [00:39<00:01, 9313.82it/s] 98%|█████████▊| 390779/400000 [00:39<00:00, 9389.83it/s] 98%|█████████▊| 391719/400000 [00:39<00:00, 9365.50it/s] 98%|█████████▊| 392658/400000 [00:39<00:00, 9371.01it/s] 98%|█████████▊| 393614/400000 [00:39<00:00, 9424.89it/s] 99%|█████████▊| 394562/400000 [00:40<00:00, 9439.25it/s] 99%|█████████▉| 395537/400000 [00:40<00:00, 9528.89it/s] 99%|█████████▉| 396531/400000 [00:40<00:00, 9646.24it/s] 99%|█████████▉| 397497/400000 [00:40<00:00, 9583.74it/s]100%|█████████▉| 398456/400000 [00:40<00:00, 9567.86it/s]100%|█████████▉| 399429/400000 [00:40<00:00, 9614.25it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9861.11it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f09931a4128> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011132686861253774 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011267642711715953 	 Accuracy: 55

  model saves at 55% accuracy 

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

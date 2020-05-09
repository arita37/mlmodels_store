
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f0c44f0b470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 22:12:32.339216
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 22:12:32.343725
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 22:12:32.347453
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 22:12:32.352097
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0c3896e0f0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 348606.0938
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 238943.0156
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 147437.2344
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 87859.9219
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 53808.4688
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 34382.0742
Epoch 7/10

1/1 [==============================] - 0s 115ms/step - loss: 23120.2734
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 16281.3789
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 11940.1494
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 9102.0713

  #### Inference Need return ypred, ytrue ######################### 
[[-5.70251107e-01 -7.77534366e-01  3.70640635e-01  9.52334881e-01
  -7.17321396e-01 -1.84944093e-01 -1.17391467e+00  6.41512871e-01
   2.60796547e-02  1.44111586e+00  1.62996054e-02 -9.15150464e-01
  -1.28672183e+00 -4.65992749e-01  1.68161607e+00 -7.74761915e-01
   1.39064097e+00  1.08175766e+00 -1.13209808e+00  6.74785197e-01
   4.86517578e-01 -6.92900181e-01 -1.56832933e-02  3.93548340e-01
   4.21543002e-01  3.19430470e-01 -8.86096120e-01 -8.46700311e-01
  -8.46251011e-01  7.26260245e-02  1.53801882e+00 -5.98549545e-01
   3.02031547e-01  4.67295110e-01 -1.28948188e+00  2.91818142e-01
   1.12632203e+00  1.17694154e-01 -3.92892659e-01  2.53246367e-01
   1.12305117e+00 -1.47232294e-01 -3.70023191e-01 -1.13697064e+00
  -1.94469079e-01  7.95023680e-01  3.00003469e-01 -5.49302697e-01
  -1.18368566e-01  3.73355746e-02  9.04555798e-01  8.27513456e-01
  -1.20991111e+00 -1.35832357e+00  5.42117357e-01  1.33180469e-01
  -1.34321675e-01 -1.80829453e+00 -9.39821154e-02  6.84401572e-01
  -1.93840548e-01  7.31054163e+00  4.98375511e+00  5.56698799e+00
   6.37633991e+00  7.63183069e+00  6.25324249e+00  5.08055162e+00
   5.57883549e+00  6.03233433e+00  6.08048248e+00  5.95956182e+00
   6.95540953e+00  5.57420111e+00  6.74881744e+00  4.83790016e+00
   6.94100714e+00  5.93996906e+00  5.49758768e+00  5.97628975e+00
   7.34662056e+00  7.21589470e+00  4.52873087e+00  6.22936583e+00
   6.98865843e+00  7.84697962e+00  5.88138390e+00  6.39840937e+00
   6.02184486e+00  6.37208509e+00  5.92398834e+00  4.91307831e+00
   6.50568914e+00  7.95587301e+00  6.59302950e+00  7.28382015e+00
   5.63584328e+00  6.62776136e+00  6.14395237e+00  7.33596134e+00
   8.01059723e+00  7.43081093e+00  7.17216825e+00  5.74271154e+00
   5.88617134e+00  5.87668133e+00  5.89294386e+00  5.78874779e+00
   7.30755424e+00  5.14575195e+00  6.75861359e+00  5.09641981e+00
   6.71205759e+00  6.12970591e+00  5.48407078e+00  7.74927139e+00
   7.28996420e+00  6.47509575e+00  6.98136377e+00  7.33204603e+00
  -6.33029163e-01 -8.52552414e-01  6.84495032e-01 -8.79756987e-01
  -7.60849535e-01  3.13026607e-01 -3.13730538e-01 -2.71371096e-01
  -6.98170662e-02  5.28474212e-01  1.09958023e-01  7.05753267e-03
  -2.12656915e-01  3.53511781e-01 -3.54408622e-02 -3.48199069e-01
  -6.53455555e-01  5.27525485e-01  1.12335205e+00 -4.83928770e-01
  -1.28616405e+00  5.80503345e-02  4.54523683e-01  3.23328435e-01
  -1.26549387e+00 -8.18521261e-01 -5.72340310e-01  4.24367845e-01
  -1.10073078e+00  2.46643707e-01 -2.42943466e-01 -1.06975830e+00
  -2.00618476e-01 -9.84904170e-01  8.83764565e-01  6.95202589e-01
  -4.27697122e-01  1.49179712e-01 -9.52634335e-01 -9.19127390e-02
   9.81020108e-02  6.79309070e-02  7.50718892e-01 -1.67150021e+00
   1.15055668e+00 -2.24173725e-01  9.45804358e-01 -9.25867736e-01
  -6.98007643e-01  2.91933715e-02 -1.69836223e-01 -2.97331929e-01
  -9.04510617e-01 -2.03630257e+00 -2.79021412e-02  7.41294563e-01
   3.67771477e-01  1.64721429e+00 -6.11709476e-01  4.09963757e-01
   1.90228534e+00  2.11626744e+00  7.46377707e-01  1.34297252e+00
   1.49474955e+00  1.09530985e+00  1.02130580e+00  9.75044906e-01
   8.83102655e-01  1.93944025e+00  1.62372136e+00  1.04240632e+00
   3.75947118e-01  1.91805053e+00  2.92570770e-01  1.40662456e+00
   1.23483384e+00  7.46020138e-01  9.56998408e-01  1.62713587e+00
   9.99708056e-01  6.77129567e-01  2.73037386e+00  2.39457369e+00
   3.65783334e-01  2.01335907e+00  1.58426332e+00  3.06798792e+00
   2.68284273e+00  1.28933179e+00  6.61601841e-01  8.74710321e-01
   3.14134359e-01  3.35979998e-01  1.49239957e-01  5.58448434e-01
   1.92186546e+00  2.74889827e-01  2.19185174e-01  1.62566614e+00
   2.15691423e+00  2.40893841e-01  3.90338302e-01  1.15549088e+00
   1.15732992e+00  1.93214691e+00  1.21751022e+00  5.35513818e-01
   4.99569237e-01  1.10181415e+00  1.36415434e+00  1.06724310e+00
   1.99538016e+00  6.32184207e-01  2.32526720e-01  4.80546653e-01
   3.22369099e-01  1.10389996e+00  1.70716405e+00  5.83364367e-01
   6.53884411e-02  8.31170559e+00  8.18004417e+00  7.40283251e+00
   6.43043995e+00  7.12353230e+00  7.05621481e+00  6.99714756e+00
   7.15347624e+00  8.05675793e+00  7.84118652e+00  7.54111433e+00
   6.67110968e+00  6.56438589e+00  6.43462515e+00  7.70649862e+00
   7.53528881e+00  7.11700678e+00  8.15691090e+00  7.09367180e+00
   5.81959295e+00  7.10957384e+00  7.26569176e+00  6.87888575e+00
   6.99213314e+00  7.82827234e+00  6.01526785e+00  7.63427973e+00
   7.25246668e+00  7.20416832e+00  7.97751379e+00  7.21332407e+00
   6.59972811e+00  7.51041079e+00  7.06480265e+00  7.35720062e+00
   5.61643028e+00  7.16341734e+00  7.57970428e+00  7.36008120e+00
   7.79306698e+00  6.22074318e+00  7.60233164e+00  6.41513062e+00
   7.64940500e+00  7.93369818e+00  7.13906240e+00  6.35746765e+00
   6.86535788e+00  7.93200016e+00  8.12122440e+00  5.80540419e+00
   7.90814877e+00  6.82747507e+00  6.77490902e+00  7.55004644e+00
   6.21696758e+00  6.64509678e+00  6.39601326e+00  5.95632458e+00
   5.70559561e-01  1.22500086e+00  2.22590065e+00  4.36935663e-01
   1.30930090e+00  1.28218448e+00  1.08213639e+00  4.91244674e-01
   4.03761804e-01  2.21822023e-01  1.94812608e+00  1.09279621e+00
   1.55179143e-01  2.79659128e+00  2.91582346e-01  1.63768840e+00
   1.01074886e+00  1.42412353e+00  8.66627157e-01  9.31194186e-01
   4.14736390e-01  6.07445836e-01  3.71767282e-01  7.26584196e-01
   7.64821529e-01  7.34443188e-01  4.19675052e-01  5.30851126e-01
   1.55971646e+00  1.07861567e+00  1.85084486e+00  1.15025306e+00
   6.89734280e-01  1.24750745e+00  6.07281089e-01  1.16917205e+00
   1.60635650e-01  2.15215063e+00  1.31370807e+00  6.83172703e-01
   1.80026209e+00  6.58379316e-01  1.40594244e+00  2.45744824e-01
   6.04444921e-01  1.94270134e+00  9.87163424e-01  3.08280611e+00
   2.58763337e+00  5.89562654e-01  2.79020786e+00  1.07779753e+00
   1.49859011e+00  1.45781887e+00  8.32981467e-01  1.08940268e+00
   1.14461410e+00  1.73482776e+00  5.66535652e-01  1.58819628e+00
  -9.33593559e+00  6.36762333e+00  1.45561635e-01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 22:12:42.831901
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.9706
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 22:12:42.836287
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9234.25
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 22:12:42.840206
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.9981
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 22:12:42.843945
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -825.992
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139689984087656
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139689181701008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139689181701512
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139689181702016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139689181702520
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139689181703024

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f0c309ce128> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.669669
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.632043
grad_step = 000002, loss = 0.605340
grad_step = 000003, loss = 0.576630
grad_step = 000004, loss = 0.544476
grad_step = 000005, loss = 0.507693
grad_step = 000006, loss = 0.481353
grad_step = 000007, loss = 0.474441
grad_step = 000008, loss = 0.474899
grad_step = 000009, loss = 0.456662
grad_step = 000010, loss = 0.435383
grad_step = 000011, loss = 0.421196
grad_step = 000012, loss = 0.412418
grad_step = 000013, loss = 0.404816
grad_step = 000014, loss = 0.395883
grad_step = 000015, loss = 0.384785
grad_step = 000016, loss = 0.371741
grad_step = 000017, loss = 0.357736
grad_step = 000018, loss = 0.344577
grad_step = 000019, loss = 0.333875
grad_step = 000020, loss = 0.325578
grad_step = 000021, loss = 0.317204
grad_step = 000022, loss = 0.306687
grad_step = 000023, loss = 0.294788
grad_step = 000024, loss = 0.283680
grad_step = 000025, loss = 0.274339
grad_step = 000026, loss = 0.265997
grad_step = 000027, loss = 0.257508
grad_step = 000028, loss = 0.248424
grad_step = 000029, loss = 0.238908
grad_step = 000030, loss = 0.229403
grad_step = 000031, loss = 0.220488
grad_step = 000032, loss = 0.212442
grad_step = 000033, loss = 0.204791
grad_step = 000034, loss = 0.196831
grad_step = 000035, loss = 0.188499
grad_step = 000036, loss = 0.180334
grad_step = 000037, loss = 0.172776
grad_step = 000038, loss = 0.165699
grad_step = 000039, loss = 0.158682
grad_step = 000040, loss = 0.151523
grad_step = 000041, loss = 0.144420
grad_step = 000042, loss = 0.137692
grad_step = 000043, loss = 0.131324
grad_step = 000044, loss = 0.125069
grad_step = 000045, loss = 0.118874
grad_step = 000046, loss = 0.112867
grad_step = 000047, loss = 0.107135
grad_step = 000048, loss = 0.101600
grad_step = 000049, loss = 0.096169
grad_step = 000050, loss = 0.090893
grad_step = 000051, loss = 0.085841
grad_step = 000052, loss = 0.081035
grad_step = 000053, loss = 0.076392
grad_step = 000054, loss = 0.071838
grad_step = 000055, loss = 0.067440
grad_step = 000056, loss = 0.063307
grad_step = 000057, loss = 0.059384
grad_step = 000058, loss = 0.055560
grad_step = 000059, loss = 0.051851
grad_step = 000060, loss = 0.048365
grad_step = 000061, loss = 0.045094
grad_step = 000062, loss = 0.041957
grad_step = 000063, loss = 0.038935
grad_step = 000064, loss = 0.036079
grad_step = 000065, loss = 0.033431
grad_step = 000066, loss = 0.030943
grad_step = 000067, loss = 0.028550
grad_step = 000068, loss = 0.026305
grad_step = 000069, loss = 0.024238
grad_step = 000070, loss = 0.022315
grad_step = 000071, loss = 0.020500
grad_step = 000072, loss = 0.018807
grad_step = 000073, loss = 0.017251
grad_step = 000074, loss = 0.015820
grad_step = 000075, loss = 0.014489
grad_step = 000076, loss = 0.013262
grad_step = 000077, loss = 0.012143
grad_step = 000078, loss = 0.011113
grad_step = 000079, loss = 0.010171
grad_step = 000080, loss = 0.009317
grad_step = 000081, loss = 0.008539
grad_step = 000082, loss = 0.007828
grad_step = 000083, loss = 0.007186
grad_step = 000084, loss = 0.006611
grad_step = 000085, loss = 0.006088
grad_step = 000086, loss = 0.005614
grad_step = 000087, loss = 0.005193
grad_step = 000088, loss = 0.004816
grad_step = 000089, loss = 0.004475
grad_step = 000090, loss = 0.004170
grad_step = 000091, loss = 0.003901
grad_step = 000092, loss = 0.003660
grad_step = 000093, loss = 0.003445
grad_step = 000094, loss = 0.003261
grad_step = 000095, loss = 0.003106
grad_step = 000096, loss = 0.002965
grad_step = 000097, loss = 0.002829
grad_step = 000098, loss = 0.002694
grad_step = 000099, loss = 0.002582
grad_step = 000100, loss = 0.002498
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002431
grad_step = 000102, loss = 0.002371
grad_step = 000103, loss = 0.002305
grad_step = 000104, loss = 0.002238
grad_step = 000105, loss = 0.002180
grad_step = 000106, loss = 0.002135
grad_step = 000107, loss = 0.002104
grad_step = 000108, loss = 0.002082
grad_step = 000109, loss = 0.002068
grad_step = 000110, loss = 0.002057
grad_step = 000111, loss = 0.002048
grad_step = 000112, loss = 0.002022
grad_step = 000113, loss = 0.001991
grad_step = 000114, loss = 0.001953
grad_step = 000115, loss = 0.001921
grad_step = 000116, loss = 0.001899
grad_step = 000117, loss = 0.001890
grad_step = 000118, loss = 0.001889
grad_step = 000119, loss = 0.001895
grad_step = 000120, loss = 0.001909
grad_step = 000121, loss = 0.001918
grad_step = 000122, loss = 0.001928
grad_step = 000123, loss = 0.001900
grad_step = 000124, loss = 0.001861
grad_step = 000125, loss = 0.001815
grad_step = 000126, loss = 0.001791
grad_step = 000127, loss = 0.001792
grad_step = 000128, loss = 0.001807
grad_step = 000129, loss = 0.001828
grad_step = 000130, loss = 0.001834
grad_step = 000131, loss = 0.001826
grad_step = 000132, loss = 0.001783
grad_step = 000133, loss = 0.001745
grad_step = 000134, loss = 0.001725
grad_step = 000135, loss = 0.001728
grad_step = 000136, loss = 0.001745
grad_step = 000137, loss = 0.001753
grad_step = 000138, loss = 0.001756
grad_step = 000139, loss = 0.001734
grad_step = 000140, loss = 0.001711
grad_step = 000141, loss = 0.001685
grad_step = 000142, loss = 0.001670
grad_step = 000143, loss = 0.001664
grad_step = 000144, loss = 0.001667
grad_step = 000145, loss = 0.001674
grad_step = 000146, loss = 0.001688
grad_step = 000147, loss = 0.001717
grad_step = 000148, loss = 0.001748
grad_step = 000149, loss = 0.001798
grad_step = 000150, loss = 0.001787
grad_step = 000151, loss = 0.001749
grad_step = 000152, loss = 0.001662
grad_step = 000153, loss = 0.001616
grad_step = 000154, loss = 0.001634
grad_step = 000155, loss = 0.001673
grad_step = 000156, loss = 0.001698
grad_step = 000157, loss = 0.001668
grad_step = 000158, loss = 0.001625
grad_step = 000159, loss = 0.001595
grad_step = 000160, loss = 0.001593
grad_step = 000161, loss = 0.001611
grad_step = 000162, loss = 0.001628
grad_step = 000163, loss = 0.001634
grad_step = 000164, loss = 0.001621
grad_step = 000165, loss = 0.001598
grad_step = 000166, loss = 0.001578
grad_step = 000167, loss = 0.001566
grad_step = 000168, loss = 0.001566
grad_step = 000169, loss = 0.001572
grad_step = 000170, loss = 0.001583
grad_step = 000171, loss = 0.001590
grad_step = 000172, loss = 0.001597
grad_step = 000173, loss = 0.001604
grad_step = 000174, loss = 0.001608
grad_step = 000175, loss = 0.001614
grad_step = 000176, loss = 0.001603
grad_step = 000177, loss = 0.001583
grad_step = 000178, loss = 0.001553
grad_step = 000179, loss = 0.001537
grad_step = 000180, loss = 0.001546
grad_step = 000181, loss = 0.001576
grad_step = 000182, loss = 0.001604
grad_step = 000183, loss = 0.001626
grad_step = 000184, loss = 0.001616
grad_step = 000185, loss = 0.001595
grad_step = 000186, loss = 0.001557
grad_step = 000187, loss = 0.001537
grad_step = 000188, loss = 0.001545
grad_step = 000189, loss = 0.001575
grad_step = 000190, loss = 0.001585
grad_step = 000191, loss = 0.001571
grad_step = 000192, loss = 0.001528
grad_step = 000193, loss = 0.001499
grad_step = 000194, loss = 0.001500
grad_step = 000195, loss = 0.001522
grad_step = 000196, loss = 0.001539
grad_step = 000197, loss = 0.001530
grad_step = 000198, loss = 0.001508
grad_step = 000199, loss = 0.001488
grad_step = 000200, loss = 0.001484
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001495
grad_step = 000202, loss = 0.001511
grad_step = 000203, loss = 0.001527
grad_step = 000204, loss = 0.001544
grad_step = 000205, loss = 0.001588
grad_step = 000206, loss = 0.001698
grad_step = 000207, loss = 0.001967
grad_step = 000208, loss = 0.002128
grad_step = 000209, loss = 0.002101
grad_step = 000210, loss = 0.001567
grad_step = 000211, loss = 0.001575
grad_step = 000212, loss = 0.001913
grad_step = 000213, loss = 0.001686
grad_step = 000214, loss = 0.001487
grad_step = 000215, loss = 0.001731
grad_step = 000216, loss = 0.001644
grad_step = 000217, loss = 0.001478
grad_step = 000218, loss = 0.001589
grad_step = 000219, loss = 0.001612
grad_step = 000220, loss = 0.001476
grad_step = 000221, loss = 0.001501
grad_step = 000222, loss = 0.001569
grad_step = 000223, loss = 0.001506
grad_step = 000224, loss = 0.001455
grad_step = 000225, loss = 0.001512
grad_step = 000226, loss = 0.001524
grad_step = 000227, loss = 0.001453
grad_step = 000228, loss = 0.001468
grad_step = 000229, loss = 0.001502
grad_step = 000230, loss = 0.001467
grad_step = 000231, loss = 0.001439
grad_step = 000232, loss = 0.001471
grad_step = 000233, loss = 0.001467
grad_step = 000234, loss = 0.001438
grad_step = 000235, loss = 0.001442
grad_step = 000236, loss = 0.001458
grad_step = 000237, loss = 0.001442
grad_step = 000238, loss = 0.001427
grad_step = 000239, loss = 0.001439
grad_step = 000240, loss = 0.001441
grad_step = 000241, loss = 0.001426
grad_step = 000242, loss = 0.001421
grad_step = 000243, loss = 0.001429
grad_step = 000244, loss = 0.001427
grad_step = 000245, loss = 0.001416
grad_step = 000246, loss = 0.001414
grad_step = 000247, loss = 0.001418
grad_step = 000248, loss = 0.001417
grad_step = 000249, loss = 0.001409
grad_step = 000250, loss = 0.001406
grad_step = 000251, loss = 0.001408
grad_step = 000252, loss = 0.001409
grad_step = 000253, loss = 0.001405
grad_step = 000254, loss = 0.001403
grad_step = 000255, loss = 0.001409
grad_step = 000256, loss = 0.001423
grad_step = 000257, loss = 0.001453
grad_step = 000258, loss = 0.001503
grad_step = 000259, loss = 0.001596
grad_step = 000260, loss = 0.001656
grad_step = 000261, loss = 0.001639
grad_step = 000262, loss = 0.001506
grad_step = 000263, loss = 0.001404
grad_step = 000264, loss = 0.001423
grad_step = 000265, loss = 0.001490
grad_step = 000266, loss = 0.001504
grad_step = 000267, loss = 0.001446
grad_step = 000268, loss = 0.001411
grad_step = 000269, loss = 0.001417
grad_step = 000270, loss = 0.001425
grad_step = 000271, loss = 0.001425
grad_step = 000272, loss = 0.001421
grad_step = 000273, loss = 0.001414
grad_step = 000274, loss = 0.001393
grad_step = 000275, loss = 0.001380
grad_step = 000276, loss = 0.001392
grad_step = 000277, loss = 0.001409
grad_step = 000278, loss = 0.001405
grad_step = 000279, loss = 0.001376
grad_step = 000280, loss = 0.001359
grad_step = 000281, loss = 0.001369
grad_step = 000282, loss = 0.001384
grad_step = 000283, loss = 0.001383
grad_step = 000284, loss = 0.001368
grad_step = 000285, loss = 0.001358
grad_step = 000286, loss = 0.001358
grad_step = 000287, loss = 0.001359
grad_step = 000288, loss = 0.001357
grad_step = 000289, loss = 0.001354
grad_step = 000290, loss = 0.001354
grad_step = 000291, loss = 0.001357
grad_step = 000292, loss = 0.001355
grad_step = 000293, loss = 0.001348
grad_step = 000294, loss = 0.001341
grad_step = 000295, loss = 0.001338
grad_step = 000296, loss = 0.001338
grad_step = 000297, loss = 0.001337
grad_step = 000298, loss = 0.001335
grad_step = 000299, loss = 0.001332
grad_step = 000300, loss = 0.001330
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001331
grad_step = 000302, loss = 0.001333
grad_step = 000303, loss = 0.001337
grad_step = 000304, loss = 0.001344
grad_step = 000305, loss = 0.001361
grad_step = 000306, loss = 0.001398
grad_step = 000307, loss = 0.001466
grad_step = 000308, loss = 0.001569
grad_step = 000309, loss = 0.001682
grad_step = 000310, loss = 0.001726
grad_step = 000311, loss = 0.001653
grad_step = 000312, loss = 0.001451
grad_step = 000313, loss = 0.001325
grad_step = 000314, loss = 0.001359
grad_step = 000315, loss = 0.001465
grad_step = 000316, loss = 0.001497
grad_step = 000317, loss = 0.001401
grad_step = 000318, loss = 0.001317
grad_step = 000319, loss = 0.001337
grad_step = 000320, loss = 0.001406
grad_step = 000321, loss = 0.001421
grad_step = 000322, loss = 0.001357
grad_step = 000323, loss = 0.001306
grad_step = 000324, loss = 0.001319
grad_step = 000325, loss = 0.001361
grad_step = 000326, loss = 0.001374
grad_step = 000327, loss = 0.001340
grad_step = 000328, loss = 0.001304
grad_step = 000329, loss = 0.001301
grad_step = 000330, loss = 0.001324
grad_step = 000331, loss = 0.001342
grad_step = 000332, loss = 0.001331
grad_step = 000333, loss = 0.001307
grad_step = 000334, loss = 0.001292
grad_step = 000335, loss = 0.001297
grad_step = 000336, loss = 0.001311
grad_step = 000337, loss = 0.001316
grad_step = 000338, loss = 0.001309
grad_step = 000339, loss = 0.001295
grad_step = 000340, loss = 0.001286
grad_step = 000341, loss = 0.001287
grad_step = 000342, loss = 0.001293
grad_step = 000343, loss = 0.001298
grad_step = 000344, loss = 0.001297
grad_step = 000345, loss = 0.001291
grad_step = 000346, loss = 0.001284
grad_step = 000347, loss = 0.001279
grad_step = 000348, loss = 0.001278
grad_step = 000349, loss = 0.001281
grad_step = 000350, loss = 0.001283
grad_step = 000351, loss = 0.001284
grad_step = 000352, loss = 0.001283
grad_step = 000353, loss = 0.001281
grad_step = 000354, loss = 0.001277
grad_step = 000355, loss = 0.001274
grad_step = 000356, loss = 0.001271
grad_step = 000357, loss = 0.001269
grad_step = 000358, loss = 0.001269
grad_step = 000359, loss = 0.001268
grad_step = 000360, loss = 0.001269
grad_step = 000361, loss = 0.001269
grad_step = 000362, loss = 0.001270
grad_step = 000363, loss = 0.001271
grad_step = 000364, loss = 0.001273
grad_step = 000365, loss = 0.001276
grad_step = 000366, loss = 0.001280
grad_step = 000367, loss = 0.001288
grad_step = 000368, loss = 0.001299
grad_step = 000369, loss = 0.001317
grad_step = 000370, loss = 0.001342
grad_step = 000371, loss = 0.001380
grad_step = 000372, loss = 0.001425
grad_step = 000373, loss = 0.001479
grad_step = 000374, loss = 0.001508
grad_step = 000375, loss = 0.001511
grad_step = 000376, loss = 0.001448
grad_step = 000377, loss = 0.001360
grad_step = 000378, loss = 0.001280
grad_step = 000379, loss = 0.001253
grad_step = 000380, loss = 0.001278
grad_step = 000381, loss = 0.001321
grad_step = 000382, loss = 0.001345
grad_step = 000383, loss = 0.001328
grad_step = 000384, loss = 0.001289
grad_step = 000385, loss = 0.001257
grad_step = 000386, loss = 0.001252
grad_step = 000387, loss = 0.001268
grad_step = 000388, loss = 0.001285
grad_step = 000389, loss = 0.001290
grad_step = 000390, loss = 0.001277
grad_step = 000391, loss = 0.001260
grad_step = 000392, loss = 0.001246
grad_step = 000393, loss = 0.001243
grad_step = 000394, loss = 0.001249
grad_step = 000395, loss = 0.001257
grad_step = 000396, loss = 0.001262
grad_step = 000397, loss = 0.001259
grad_step = 000398, loss = 0.001251
grad_step = 000399, loss = 0.001241
grad_step = 000400, loss = 0.001234
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001232
grad_step = 000402, loss = 0.001234
grad_step = 000403, loss = 0.001238
grad_step = 000404, loss = 0.001241
grad_step = 000405, loss = 0.001242
grad_step = 000406, loss = 0.001240
grad_step = 000407, loss = 0.001236
grad_step = 000408, loss = 0.001231
grad_step = 000409, loss = 0.001228
grad_step = 000410, loss = 0.001225
grad_step = 000411, loss = 0.001223
grad_step = 000412, loss = 0.001222
grad_step = 000413, loss = 0.001221
grad_step = 000414, loss = 0.001220
grad_step = 000415, loss = 0.001220
grad_step = 000416, loss = 0.001219
grad_step = 000417, loss = 0.001218
grad_step = 000418, loss = 0.001219
grad_step = 000419, loss = 0.001219
grad_step = 000420, loss = 0.001220
grad_step = 000421, loss = 0.001221
grad_step = 000422, loss = 0.001223
grad_step = 000423, loss = 0.001225
grad_step = 000424, loss = 0.001229
grad_step = 000425, loss = 0.001234
grad_step = 000426, loss = 0.001243
grad_step = 000427, loss = 0.001256
grad_step = 000428, loss = 0.001275
grad_step = 000429, loss = 0.001301
grad_step = 000430, loss = 0.001338
grad_step = 000431, loss = 0.001379
grad_step = 000432, loss = 0.001423
grad_step = 000433, loss = 0.001447
grad_step = 000434, loss = 0.001445
grad_step = 000435, loss = 0.001395
grad_step = 000436, loss = 0.001320
grad_step = 000437, loss = 0.001244
grad_step = 000438, loss = 0.001206
grad_step = 000439, loss = 0.001214
grad_step = 000440, loss = 0.001244
grad_step = 000441, loss = 0.001267
grad_step = 000442, loss = 0.001261
grad_step = 000443, loss = 0.001238
grad_step = 000444, loss = 0.001213
grad_step = 000445, loss = 0.001204
grad_step = 000446, loss = 0.001209
grad_step = 000447, loss = 0.001216
grad_step = 000448, loss = 0.001218
grad_step = 000449, loss = 0.001208
grad_step = 000450, loss = 0.001197
grad_step = 000451, loss = 0.001189
grad_step = 000452, loss = 0.001188
grad_step = 000453, loss = 0.001192
grad_step = 000454, loss = 0.001196
grad_step = 000455, loss = 0.001197
grad_step = 000456, loss = 0.001192
grad_step = 000457, loss = 0.001185
grad_step = 000458, loss = 0.001178
grad_step = 000459, loss = 0.001173
grad_step = 000460, loss = 0.001171
grad_step = 000461, loss = 0.001171
grad_step = 000462, loss = 0.001173
grad_step = 000463, loss = 0.001173
grad_step = 000464, loss = 0.001173
grad_step = 000465, loss = 0.001171
grad_step = 000466, loss = 0.001168
grad_step = 000467, loss = 0.001164
grad_step = 000468, loss = 0.001161
grad_step = 000469, loss = 0.001158
grad_step = 000470, loss = 0.001156
grad_step = 000471, loss = 0.001155
grad_step = 000472, loss = 0.001154
grad_step = 000473, loss = 0.001154
grad_step = 000474, loss = 0.001153
grad_step = 000475, loss = 0.001153
grad_step = 000476, loss = 0.001152
grad_step = 000477, loss = 0.001151
grad_step = 000478, loss = 0.001150
grad_step = 000479, loss = 0.001149
grad_step = 000480, loss = 0.001148
grad_step = 000481, loss = 0.001147
grad_step = 000482, loss = 0.001146
grad_step = 000483, loss = 0.001145
grad_step = 000484, loss = 0.001145
grad_step = 000485, loss = 0.001145
grad_step = 000486, loss = 0.001146
grad_step = 000487, loss = 0.001149
grad_step = 000488, loss = 0.001153
grad_step = 000489, loss = 0.001160
grad_step = 000490, loss = 0.001172
grad_step = 000491, loss = 0.001190
grad_step = 000492, loss = 0.001217
grad_step = 000493, loss = 0.001252
grad_step = 000494, loss = 0.001296
grad_step = 000495, loss = 0.001340
grad_step = 000496, loss = 0.001366
grad_step = 000497, loss = 0.001368
grad_step = 000498, loss = 0.001313
grad_step = 000499, loss = 0.001249
grad_step = 000500, loss = 0.001181
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001162
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

  date_run                              2020-05-09 22:13:04.909120
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.271481
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 22:13:04.915821
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.212143
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 22:13:04.923836
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.143611
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 22:13:04.930905
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.22359
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
0   2020-05-09 22:12:32.339216  ...    mean_absolute_error
1   2020-05-09 22:12:32.343725  ...     mean_squared_error
2   2020-05-09 22:12:32.347453  ...  median_absolute_error
3   2020-05-09 22:12:32.352097  ...               r2_score
4   2020-05-09 22:12:42.831901  ...    mean_absolute_error
5   2020-05-09 22:12:42.836287  ...     mean_squared_error
6   2020-05-09 22:12:42.840206  ...  median_absolute_error
7   2020-05-09 22:12:42.843945  ...               r2_score
8   2020-05-09 22:13:04.909120  ...    mean_absolute_error
9   2020-05-09 22:13:04.915821  ...     mean_squared_error
10  2020-05-09 22:13:04.923836  ...  median_absolute_error
11  2020-05-09 22:13:04.930905  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3407872/9912422 [00:00<00:00, 34075177.87it/s]9920512it [00:00, 35472750.85it/s]                             
0it [00:00, ?it/s]32768it [00:00, 554894.13it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148735.73it/s]1654784it [00:00, 10308921.78it/s]                         
0it [00:00, ?it/s]8192it [00:00, 177619.28it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f17217f6780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16bef39be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f17217ade48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16bef39d68> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f17217ade48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16c865c470> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16bef3c080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16d41a8cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16bef3c080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f16d41a8cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f17217ade48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f85461701d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=df74f615f0c097c6d1fbf5bd9de794bf9b0a6a71ab42c1212ce2d16724c5daca
  Stored in directory: /tmp/pip-ephem-wheel-cache-0y8d1qde/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f84dee55c88> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1712128/17464789 [=>............................] - ETA: 0s
 6676480/17464789 [==========>...................] - ETA: 0s
12673024/17464789 [====================>.........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 22:14:32.572821: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 22:14:32.576740: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-09 22:14:32.576894: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556a27ad1930 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 22:14:32.576915: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8430 - accuracy: 0.4885 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8353 - accuracy: 0.4890
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7625 - accuracy: 0.4938
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7464 - accuracy: 0.4948
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7535 - accuracy: 0.4943
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7411 - accuracy: 0.4951
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7548 - accuracy: 0.4942
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7007 - accuracy: 0.4978
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6912 - accuracy: 0.4984
11000/25000 [============>.................] - ETA: 4s - loss: 7.6792 - accuracy: 0.4992
12000/25000 [=============>................] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6619 - accuracy: 0.5003
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6502 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 2s - loss: 7.6339 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6427 - accuracy: 0.5016
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6414 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6408 - accuracy: 0.5017
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6520 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6464 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6566 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 8s 338us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 22:14:47.819443
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 22:14:47.819443  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 22:14:53.812433: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 22:14:53.817134: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-09 22:14:53.817285: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5651440dc6c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 22:14:53.817300: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f06fdcaad30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 976ms/step - loss: 1.8925 - crf_viterbi_accuracy: 0.2800 - val_loss: 1.7315 - val_crf_viterbi_accuracy: 0.3333

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f06f3052f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4264 - accuracy: 0.5157
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5388 - accuracy: 0.5083
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5768 - accuracy: 0.5059
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5708 - accuracy: 0.5063
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5712 - accuracy: 0.5062
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5578 - accuracy: 0.5071
11000/25000 [============>.................] - ETA: 4s - loss: 7.5649 - accuracy: 0.5066
12000/25000 [=============>................] - ETA: 3s - loss: 7.5912 - accuracy: 0.5049
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5758 - accuracy: 0.5059
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5648 - accuracy: 0.5066
15000/25000 [=================>............] - ETA: 2s - loss: 7.5808 - accuracy: 0.5056
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6287 - accuracy: 0.5025
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6240 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6384 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6567 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6534 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6620 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 8s 339us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f06c0505898> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:05:37, 9.94kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:05:50, 14.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:01:17, 19.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:25:20, 28.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:01<5:52:48, 40.6kB/s].vector_cache/glove.6B.zip:   1%|          | 8.37M/862M [00:01<4:05:41, 57.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:51:09, 82.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.1M/862M [00:01<1:59:29, 118kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:23:12, 168kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:01<58:07, 240kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<40:57, 340kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:02<28:31, 485kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:02<20:01, 688kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:02<14:02, 975kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<09:51, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.1M/862M [00:02<06:58, 1.94MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<05:40, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<05:52, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:23, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:02, 2.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<05:49, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<05:45, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:26, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<05:46, 2.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<06:51, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<05:24, 2.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<03:57, 3.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<08:22, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<07:12, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<05:22, 2.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<06:51, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<06:08, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<04:37, 2.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<06:20, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<07:07, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<05:38, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:02, 2.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:17<05:34, 2.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<04:10, 3.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<05:57, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<05:19, 2.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.2M/862M [00:19<03:57, 3.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:55, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<04:28, 2.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:09, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<06:54, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:23<05:23, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:23<03:54, 3.26MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<11:42, 1.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<09:31, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:25<06:58, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:50, 1.62MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:10, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:23, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:38, 2.72MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:03, 1.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:39, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<05:37, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:52, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:27, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:10, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:21, 1.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:55, 1.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<07:12, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:56, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:05, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:17, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:22, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:43, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<04:19, 2.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:52, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:36, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<03:47, 3.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:47:03, 17.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<8:15:45, 24.6kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<5:46:35, 35.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<4:02:07, 50.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<3:12:18, 63.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<2:17:04, 88.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<1:36:22, 126kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<1:07:25, 179kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<53:40, 225kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<38:35, 312kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<27:14, 441kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<19:10, 625kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:01:40, 194kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<45:37, 263kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<32:29, 368kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<24:35, 485kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<18:26, 646kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<13:08, 904kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<11:58, 990kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:36, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<07:00, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:40, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:46, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:02, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:07, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:30, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<04:05, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:18, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:55, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:35, 3.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<09:50, 1.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<08:03, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:55, 1.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:49, 1.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:56, 1.93MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:27, 2.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:48, 1.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:24, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:05, 2.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:23, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:55, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:44, 3.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:15, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:53, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:42, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:05, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:33, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:25, 3.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<02:33, 4.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<28:41, 389kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<21:12, 526kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<15:05, 737kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<13:09, 843kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<11:30, 964kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:36, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:47, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<06:33, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:52, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:57, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:00, 2.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<02:56, 3.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:23:06, 131kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<59:14, 184kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<41:38, 261kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<31:37, 342kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<23:13, 466kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<16:26, 656kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<14:02, 765kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<12:01, 893kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:53, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:20, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<11:13, 952kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:58, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<06:29, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:01, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:07, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:30, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:33, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:57, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:41, 2.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:02, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:35, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:21, 2.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<03:10, 3.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:49, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:33, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:50, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:49, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:58, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:49, 2.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<02:46, 3.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<46:21, 221kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<34:33, 297kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<24:35, 417kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<17:17, 591kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<17:49, 572kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<13:31, 754kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<09:41, 1.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<09:08, 1.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:25, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:10, 1.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<06:23, 1.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:58, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:05, 1.96MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:36, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:26, 2.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:43, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:19, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:13, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:03, 3.22MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<10:05:01, 16.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<7:05:30, 23.2kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<4:57:54, 33.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<3:28:07, 47.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<2:28:29, 65.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<1:44:54, 93.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<1:13:29, 133kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<53:31, 182kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<39:25, 246kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<28:03, 346kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<21:06, 457kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<15:45, 612kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<11:14, 855kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<10:05, 949kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<09:02, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:48, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<06:16, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:21, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:59, 2.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:59, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:27, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:20, 2.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:33, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:06, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:02, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:55, 3.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<9:01:00, 17.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<6:19:23, 24.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<4:25:02, 35.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<3:06:59, 49.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<2:12:43, 69.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<1:33:11, 99.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<1:05:03, 141kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<51:54, 177kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<37:16, 246kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<26:15, 348kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<20:26, 445kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<16:10, 562kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<11:42, 776kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<08:16, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<10:28, 863kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:05, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:50, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:11, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<39:00, 230kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<29:07, 308kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<20:49, 430kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<15:56, 558kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<12:04, 737kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:39, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<08:05, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:27, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:01, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:28, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:09, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:10, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:30, 1.93MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:21, 2.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:20, 1.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:47, 1.80MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:46, 2.28MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:42, 3.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<18:33, 461kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<13:52, 616kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<09:52, 863kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:49, 961kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<07:54, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:57, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:30, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:42, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:30, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:23, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:56, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:57, 2.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:00, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:38, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:45, 2.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:51, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:32, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:40, 3.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:47, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:18, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:25, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:41, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:25, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:35, 3.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:41, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:17, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:24, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<02:28, 3.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<58:22, 136kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<41:38, 190kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<29:15, 270kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<22:14, 354kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<17:09, 458kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<12:23, 634kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<09:52, 789kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<07:43, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:35, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:41, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<05:33, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:12, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:02, 2.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:14, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:10, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:46, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:24, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:51, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:53, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:47, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:11, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:18, 2.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:30, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:12, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:25, 3.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:09, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<02:22, 3.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:22, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:50, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:00, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<02:11, 3.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:48, 1.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:30, 1.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:59, 1.80MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:28, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:35, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:34, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<03:38, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:16, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:27, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:21, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:47, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:58, 2.37MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:08, 3.26MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<10:38, 656kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<08:09, 854kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:52, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:42, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:24, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:07, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:58, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:28, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:35, 2.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:23, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:45, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:55, 2.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:07, 3.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:02, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:55, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:36, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:08, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:16, 1.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:15, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:21, 2.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:48, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:02, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:59, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:36, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:10, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<02:21, 2.74MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:10, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:31, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:47, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:58, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:43, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:03, 3.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:22, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:40, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<01:56, 3.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<45:51, 136kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<32:42, 190kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<22:55, 270kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<17:24, 353kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<13:25, 458kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<09:41, 633kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<07:43, 788kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:00, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:20, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:26, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:19, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:19, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:16, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:53, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:08, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:52, 2.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:36, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:58, 2.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:44, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:06, 1.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:25, 2.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:45, 3.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:11, 1.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:31, 1.63MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:36, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:07, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:46, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:04, 2.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:46, 2.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:30, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<01:53, 2.94MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<02:37, 2.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<02:24, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:48, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<02:33, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<02:54, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:15, 2.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:38, 3.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<04:13, 1.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:30, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:33, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<03:04, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:13, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:31, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:36, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:22, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<01:47, 2.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:26, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:46, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:11, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:21, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:10, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:38, 3.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:19, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:39, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:06, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:16, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:06, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<01:35, 3.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:00, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<01:29, 3.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:06, 4.38MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<20:19, 239kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<14:42, 329kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<10:21, 465kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<08:19, 574kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<06:14, 766kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:31, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:11, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<33:27, 141kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<24:21, 193kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<17:12, 273kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<12:00, 388kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<10:52, 427kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<08:04, 574kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<05:44, 803kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<05:03, 904kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:28, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:21, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<41:32, 108kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<29:08, 154kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<20:13, 220kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<30:17, 147kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<21:32, 206kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<15:05, 292kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<10:32, 415kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<23:53, 183kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<17:36, 248kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<12:30, 348kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<08:42, 494kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<36:48, 117kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<26:10, 164kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<18:19, 233kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<13:42, 309kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<10:00, 422kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<07:03, 595kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<05:52, 708kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<04:31, 919kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<03:15, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:32<03:13, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:40, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:57, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:18, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:26, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:54, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:57, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:46, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:19, 2.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:49, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:40, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:15, 3.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:46, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:01, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:36, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:42, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:34, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:11, 3.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:41, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:33, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:10, 3.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:39, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:31, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:09, 3.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:38, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:51, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:27, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:03, 3.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:15, 1.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:25, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:47, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:56, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:30, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:05, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:13, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:35, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:53, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:04, 1.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:06, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:38, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:09, 2.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<3:05:58, 17.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<2:10:14, 24.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<1:30:30, 35.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<1:03:22, 49.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<44:35, 70.2kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<31:01, 100kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<22:11, 138kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<16:08, 190kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<11:24, 267kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<08:19, 360kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<06:07, 488kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<04:19, 686kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:40, 798kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<03:09, 925kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:20, 1.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:38, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<11:39, 245kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<08:26, 338kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<05:55, 477kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:45, 587kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<03:53, 717kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:51, 973kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:24, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:57, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:24, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:36, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:22, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:01, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:19, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:11, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:53, 2.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:12, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:21, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:04, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:07, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:02, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:46, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:05, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:13, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:58, 2.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:02, 2.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:57, 2.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:43, 3.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:01, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:10, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:55, 2.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:59, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:54, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:41, 3.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:57, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:06, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:52, 2.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:55, 2.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:51, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:38, 3.13MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:54, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:49, 2.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:37, 3.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:52, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:48, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:36, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:46, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:34, 3.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:43, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:32, 3.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:46, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:53, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:42, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:44, 2.20MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:40, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:30, 3.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:42, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:37, 2.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:27, 3.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:02, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:52, 1.69MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:38, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:46, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:49, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:37, 2.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:27, 3.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:44, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:28, 2.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:37, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:41, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:32, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:33, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:30, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:23, 3.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:36, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:27, 2.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:20, 3.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:37, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:32, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:23, 2.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:30, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:19, 2.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:26, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:30, 1.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:16, 3.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:37, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:31, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:22, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:28, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:22, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:15, 2.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<02:50, 259kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<02:02, 356kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<01:23, 503kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:04, 615kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:53, 746kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:38, 1.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:30, 1.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:24, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:18, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:27, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:21, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:14, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:08, 2.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:06, 2.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:02, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 3.14MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 704/400000 [00:00<00:56, 7034.32it/s]  0%|          | 1436/400000 [00:00<00:56, 7116.32it/s]  1%|          | 2198/400000 [00:00<00:54, 7259.55it/s]  1%|          | 3019/400000 [00:00<00:52, 7520.71it/s]  1%|          | 3835/400000 [00:00<00:51, 7698.81it/s]  1%|          | 4609/400000 [00:00<00:51, 7710.43it/s]  1%|▏         | 5382/400000 [00:00<00:51, 7715.76it/s]  2%|▏         | 6098/400000 [00:00<00:52, 7527.19it/s]  2%|▏         | 6878/400000 [00:00<00:51, 7602.51it/s]  2%|▏         | 7684/400000 [00:01<00:50, 7732.51it/s]  2%|▏         | 8479/400000 [00:01<00:50, 7795.49it/s]  2%|▏         | 9246/400000 [00:01<00:50, 7671.68it/s]  3%|▎         | 10049/400000 [00:01<00:50, 7773.09it/s]  3%|▎         | 10841/400000 [00:01<00:49, 7814.75it/s]  3%|▎         | 11652/400000 [00:01<00:49, 7898.58it/s]  3%|▎         | 12443/400000 [00:01<00:49, 7901.00it/s]  3%|▎         | 13232/400000 [00:01<00:49, 7748.61it/s]  4%|▎         | 14007/400000 [00:01<00:50, 7719.21it/s]  4%|▎         | 14821/400000 [00:01<00:49, 7839.97it/s]  4%|▍         | 15606/400000 [00:02<00:49, 7722.99it/s]  4%|▍         | 16389/400000 [00:02<00:49, 7752.29it/s]  4%|▍         | 17165/400000 [00:02<00:49, 7656.93it/s]  4%|▍         | 17945/400000 [00:02<00:49, 7697.99it/s]  5%|▍         | 18716/400000 [00:02<00:49, 7639.78it/s]  5%|▍         | 19502/400000 [00:02<00:49, 7704.23it/s]  5%|▌         | 20279/400000 [00:02<00:49, 7721.53it/s]  5%|▌         | 21052/400000 [00:02<00:49, 7623.16it/s]  5%|▌         | 21842/400000 [00:02<00:49, 7702.35it/s]  6%|▌         | 22657/400000 [00:02<00:48, 7830.77it/s]  6%|▌         | 23441/400000 [00:03<00:48, 7791.57it/s]  6%|▌         | 24233/400000 [00:03<00:48, 7827.00it/s]  6%|▋         | 25017/400000 [00:03<00:48, 7783.92it/s]  6%|▋         | 25796/400000 [00:03<00:49, 7521.68it/s]  7%|▋         | 26551/400000 [00:03<00:50, 7444.17it/s]  7%|▋         | 27335/400000 [00:03<00:49, 7555.46it/s]  7%|▋         | 28127/400000 [00:03<00:48, 7659.53it/s]  7%|▋         | 28903/400000 [00:03<00:48, 7687.64it/s]  7%|▋         | 29673/400000 [00:03<00:48, 7635.68it/s]  8%|▊         | 30468/400000 [00:03<00:47, 7726.11it/s]  8%|▊         | 31242/400000 [00:04<00:48, 7656.22it/s]  8%|▊         | 32025/400000 [00:04<00:47, 7707.19it/s]  8%|▊         | 32805/400000 [00:04<00:47, 7734.18it/s]  8%|▊         | 33600/400000 [00:04<00:47, 7795.70it/s]  9%|▊         | 34402/400000 [00:04<00:46, 7860.11it/s]  9%|▉         | 35208/400000 [00:04<00:46, 7917.44it/s]  9%|▉         | 36017/400000 [00:04<00:45, 7966.30it/s]  9%|▉         | 36814/400000 [00:04<00:45, 7921.13it/s]  9%|▉         | 37622/400000 [00:04<00:45, 7968.07it/s] 10%|▉         | 38420/400000 [00:04<00:45, 7870.48it/s] 10%|▉         | 39208/400000 [00:05<00:46, 7770.10it/s] 10%|▉         | 39986/400000 [00:05<00:46, 7732.20it/s] 10%|█         | 40760/400000 [00:05<00:46, 7731.37it/s] 10%|█         | 41563/400000 [00:05<00:45, 7817.93it/s] 11%|█         | 42377/400000 [00:05<00:45, 7911.84it/s] 11%|█         | 43175/400000 [00:05<00:44, 7932.08it/s] 11%|█         | 43974/400000 [00:05<00:44, 7948.45it/s] 11%|█         | 44770/400000 [00:05<00:45, 7833.48it/s] 11%|█▏        | 45554/400000 [00:05<00:45, 7803.23it/s] 12%|█▏        | 46351/400000 [00:05<00:45, 7850.92it/s] 12%|█▏        | 47137/400000 [00:06<00:45, 7698.53it/s] 12%|█▏        | 47908/400000 [00:06<00:45, 7666.51it/s] 12%|█▏        | 48676/400000 [00:06<00:45, 7666.38it/s] 12%|█▏        | 49473/400000 [00:06<00:45, 7753.43it/s] 13%|█▎        | 50249/400000 [00:06<00:45, 7717.26it/s] 13%|█▎        | 51022/400000 [00:06<00:48, 7133.51it/s] 13%|█▎        | 51745/400000 [00:06<00:49, 7080.73it/s] 13%|█▎        | 52478/400000 [00:06<00:48, 7153.10it/s] 13%|█▎        | 53280/400000 [00:06<00:46, 7389.12it/s] 14%|█▎        | 54087/400000 [00:07<00:45, 7580.82it/s] 14%|█▎        | 54850/400000 [00:07<00:47, 7334.11it/s] 14%|█▍        | 55594/400000 [00:07<00:46, 7364.04it/s] 14%|█▍        | 56335/400000 [00:07<00:47, 7207.50it/s] 14%|█▍        | 57080/400000 [00:07<00:47, 7277.68it/s] 14%|█▍        | 57850/400000 [00:07<00:46, 7399.28it/s] 15%|█▍        | 58623/400000 [00:07<00:45, 7494.63it/s] 15%|█▍        | 59375/400000 [00:07<00:46, 7339.74it/s] 15%|█▌        | 60111/400000 [00:07<00:46, 7337.76it/s] 15%|█▌        | 60866/400000 [00:07<00:45, 7397.84it/s] 15%|█▌        | 61653/400000 [00:08<00:44, 7532.23it/s] 16%|█▌        | 62435/400000 [00:08<00:44, 7614.81it/s] 16%|█▌        | 63246/400000 [00:08<00:43, 7753.62it/s] 16%|█▌        | 64023/400000 [00:08<00:43, 7685.22it/s] 16%|█▌        | 64809/400000 [00:08<00:43, 7736.79it/s] 16%|█▋        | 65601/400000 [00:08<00:42, 7788.30it/s] 17%|█▋        | 66381/400000 [00:08<00:44, 7574.35it/s] 17%|█▋        | 67199/400000 [00:08<00:42, 7744.37it/s] 17%|█▋        | 67985/400000 [00:08<00:42, 7776.41it/s] 17%|█▋        | 68803/400000 [00:08<00:41, 7890.60it/s] 17%|█▋        | 69594/400000 [00:09<00:42, 7768.01it/s] 18%|█▊        | 70399/400000 [00:09<00:41, 7850.25it/s] 18%|█▊        | 71227/400000 [00:09<00:41, 7973.45it/s] 18%|█▊        | 72026/400000 [00:09<00:41, 7957.69it/s] 18%|█▊        | 72852/400000 [00:09<00:40, 8043.34it/s] 18%|█▊        | 73670/400000 [00:09<00:40, 8082.44it/s] 19%|█▊        | 74491/400000 [00:09<00:40, 8119.47it/s] 19%|█▉        | 75316/400000 [00:09<00:39, 8157.81it/s] 19%|█▉        | 76133/400000 [00:09<00:39, 8106.67it/s] 19%|█▉        | 76956/400000 [00:09<00:39, 8141.19it/s] 19%|█▉        | 77771/400000 [00:10<00:40, 7879.64it/s] 20%|█▉        | 78561/400000 [00:10<00:40, 7851.58it/s] 20%|█▉        | 79348/400000 [00:10<00:40, 7833.24it/s] 20%|██        | 80151/400000 [00:10<00:40, 7890.58it/s] 20%|██        | 80977/400000 [00:10<00:39, 7996.15it/s] 20%|██        | 81789/400000 [00:10<00:39, 8031.38it/s] 21%|██        | 82617/400000 [00:10<00:39, 8103.43it/s] 21%|██        | 83436/400000 [00:10<00:38, 8126.28it/s] 21%|██        | 84250/400000 [00:10<00:39, 8071.20it/s] 21%|██▏       | 85065/400000 [00:10<00:38, 8092.29it/s] 21%|██▏       | 85877/400000 [00:11<00:38, 8100.00it/s] 22%|██▏       | 86688/400000 [00:11<00:38, 8088.10it/s] 22%|██▏       | 87505/400000 [00:11<00:38, 8110.83it/s] 22%|██▏       | 88317/400000 [00:11<00:38, 8046.30it/s] 22%|██▏       | 89133/400000 [00:11<00:38, 8078.21it/s] 22%|██▏       | 89949/400000 [00:11<00:38, 8100.77it/s] 23%|██▎       | 90760/400000 [00:11<00:39, 7850.86it/s] 23%|██▎       | 91547/400000 [00:11<00:39, 7806.31it/s] 23%|██▎       | 92349/400000 [00:11<00:39, 7868.54it/s] 23%|██▎       | 93152/400000 [00:12<00:38, 7915.55it/s] 23%|██▎       | 93953/400000 [00:12<00:38, 7943.24it/s] 24%|██▎       | 94748/400000 [00:12<00:39, 7737.10it/s] 24%|██▍       | 95524/400000 [00:12<00:39, 7633.28it/s] 24%|██▍       | 96289/400000 [00:12<00:40, 7556.92it/s] 24%|██▍       | 97071/400000 [00:12<00:39, 7631.95it/s] 24%|██▍       | 97883/400000 [00:12<00:38, 7770.31it/s] 25%|██▍       | 98680/400000 [00:12<00:38, 7828.58it/s] 25%|██▍       | 99479/400000 [00:12<00:38, 7875.60it/s] 25%|██▌       | 100268/400000 [00:12<00:38, 7828.25it/s] 25%|██▌       | 101052/400000 [00:13<00:39, 7588.39it/s] 25%|██▌       | 101813/400000 [00:13<00:39, 7496.22it/s] 26%|██▌       | 102565/400000 [00:13<00:40, 7435.74it/s] 26%|██▌       | 103310/400000 [00:13<00:40, 7323.09it/s] 26%|██▌       | 104044/400000 [00:13<00:41, 7191.83it/s] 26%|██▌       | 104768/400000 [00:13<00:40, 7204.41it/s] 26%|██▋       | 105573/400000 [00:13<00:39, 7436.88it/s] 27%|██▋       | 106322/400000 [00:13<00:39, 7450.10it/s] 27%|██▋       | 107069/400000 [00:13<00:39, 7431.70it/s] 27%|██▋       | 107814/400000 [00:13<00:40, 7290.45it/s] 27%|██▋       | 108567/400000 [00:14<00:39, 7360.44it/s] 27%|██▋       | 109305/400000 [00:14<00:39, 7351.07it/s] 28%|██▊       | 110055/400000 [00:14<00:39, 7394.76it/s] 28%|██▊       | 110841/400000 [00:14<00:38, 7526.63it/s] 28%|██▊       | 111607/400000 [00:14<00:38, 7566.06it/s] 28%|██▊       | 112416/400000 [00:14<00:37, 7713.43it/s] 28%|██▊       | 113217/400000 [00:14<00:36, 7797.91it/s] 29%|██▊       | 114024/400000 [00:14<00:36, 7875.65it/s] 29%|██▊       | 114832/400000 [00:14<00:35, 7934.69it/s] 29%|██▉       | 115627/400000 [00:14<00:36, 7843.73it/s] 29%|██▉       | 116413/400000 [00:15<00:36, 7831.24it/s] 29%|██▉       | 117205/400000 [00:15<00:35, 7855.52it/s] 29%|██▉       | 117991/400000 [00:15<00:36, 7832.32it/s] 30%|██▉       | 118775/400000 [00:15<00:38, 7399.83it/s] 30%|██▉       | 119521/400000 [00:15<00:38, 7277.84it/s] 30%|███       | 120323/400000 [00:15<00:37, 7484.99it/s] 30%|███       | 121113/400000 [00:15<00:36, 7603.43it/s] 30%|███       | 121911/400000 [00:15<00:36, 7711.25it/s] 31%|███       | 122685/400000 [00:15<00:36, 7660.63it/s] 31%|███       | 123478/400000 [00:15<00:35, 7738.75it/s] 31%|███       | 124275/400000 [00:16<00:35, 7804.15it/s] 31%|███▏      | 125096/400000 [00:16<00:34, 7920.14it/s] 31%|███▏      | 125890/400000 [00:16<00:34, 7863.79it/s] 32%|███▏      | 126698/400000 [00:16<00:34, 7925.49it/s] 32%|███▏      | 127492/400000 [00:16<00:34, 7888.86it/s] 32%|███▏      | 128309/400000 [00:16<00:34, 7969.46it/s] 32%|███▏      | 129120/400000 [00:16<00:33, 8010.11it/s] 32%|███▏      | 129922/400000 [00:16<00:33, 7949.89it/s] 33%|███▎      | 130719/400000 [00:16<00:33, 7954.52it/s] 33%|███▎      | 131515/400000 [00:17<00:34, 7691.81it/s] 33%|███▎      | 132287/400000 [00:17<00:35, 7583.24it/s] 33%|███▎      | 133048/400000 [00:17<00:35, 7566.44it/s] 33%|███▎      | 133838/400000 [00:17<00:34, 7660.90it/s] 34%|███▎      | 134606/400000 [00:17<00:35, 7526.40it/s] 34%|███▍      | 135360/400000 [00:17<00:36, 7174.80it/s] 34%|███▍      | 136082/400000 [00:17<00:36, 7164.89it/s] 34%|███▍      | 136892/400000 [00:17<00:35, 7419.96it/s] 34%|███▍      | 137682/400000 [00:17<00:34, 7555.79it/s] 35%|███▍      | 138498/400000 [00:17<00:33, 7726.27it/s] 35%|███▍      | 139275/400000 [00:18<00:33, 7724.35it/s] 35%|███▌      | 140091/400000 [00:18<00:33, 7847.90it/s] 35%|███▌      | 140904/400000 [00:18<00:32, 7927.71it/s] 35%|███▌      | 141699/400000 [00:18<00:32, 7896.27it/s] 36%|███▌      | 142505/400000 [00:18<00:32, 7944.53it/s] 36%|███▌      | 143301/400000 [00:18<00:32, 7831.62it/s] 36%|███▌      | 144103/400000 [00:18<00:32, 7885.61it/s] 36%|███▌      | 144893/400000 [00:18<00:33, 7656.90it/s] 36%|███▋      | 145661/400000 [00:18<00:33, 7633.85it/s] 37%|███▋      | 146461/400000 [00:18<00:32, 7739.77it/s] 37%|███▋      | 147256/400000 [00:19<00:32, 7801.08it/s] 37%|███▋      | 148081/400000 [00:19<00:31, 7927.97it/s] 37%|███▋      | 148876/400000 [00:19<00:31, 7889.87it/s] 37%|███▋      | 149666/400000 [00:19<00:31, 7829.35it/s] 38%|███▊      | 150472/400000 [00:19<00:31, 7895.23it/s] 38%|███▊      | 151263/400000 [00:19<00:31, 7897.52it/s] 38%|███▊      | 152080/400000 [00:19<00:31, 7975.05it/s] 38%|███▊      | 152902/400000 [00:19<00:30, 8045.89it/s] 38%|███▊      | 153726/400000 [00:19<00:30, 8100.87it/s] 39%|███▊      | 154543/400000 [00:19<00:30, 8119.95it/s] 39%|███▉      | 155356/400000 [00:20<00:30, 8009.59it/s] 39%|███▉      | 156159/400000 [00:20<00:30, 8015.70it/s] 39%|███▉      | 156961/400000 [00:20<00:31, 7794.67it/s] 39%|███▉      | 157743/400000 [00:20<00:31, 7712.47it/s] 40%|███▉      | 158516/400000 [00:20<00:32, 7529.74it/s] 40%|███▉      | 159305/400000 [00:20<00:31, 7633.44it/s] 40%|████      | 160128/400000 [00:20<00:30, 7802.18it/s] 40%|████      | 160938/400000 [00:20<00:30, 7886.77it/s] 40%|████      | 161747/400000 [00:20<00:29, 7946.32it/s] 41%|████      | 162543/400000 [00:20<00:30, 7895.37it/s] 41%|████      | 163338/400000 [00:21<00:29, 7910.92it/s] 41%|████      | 164151/400000 [00:21<00:29, 7974.16it/s] 41%|████      | 164965/400000 [00:21<00:29, 8019.76it/s] 41%|████▏     | 165768/400000 [00:21<00:29, 8013.13it/s] 42%|████▏     | 166570/400000 [00:21<00:29, 8007.24it/s] 42%|████▏     | 167371/400000 [00:21<00:29, 7855.21it/s] 42%|████▏     | 168158/400000 [00:21<00:29, 7829.47it/s] 42%|████▏     | 168974/400000 [00:21<00:29, 7923.12it/s] 42%|████▏     | 169782/400000 [00:21<00:28, 7967.42it/s] 43%|████▎     | 170580/400000 [00:22<00:29, 7834.90it/s] 43%|████▎     | 171365/400000 [00:22<00:30, 7534.09it/s] 43%|████▎     | 172164/400000 [00:22<00:29, 7663.58it/s] 43%|████▎     | 172951/400000 [00:22<00:29, 7722.60it/s] 43%|████▎     | 173764/400000 [00:22<00:28, 7838.05it/s] 44%|████▎     | 174585/400000 [00:22<00:28, 7945.32it/s] 44%|████▍     | 175382/400000 [00:22<00:28, 7879.48it/s] 44%|████▍     | 176172/400000 [00:22<00:28, 7828.82it/s] 44%|████▍     | 176994/400000 [00:22<00:28, 7941.90it/s] 44%|████▍     | 177824/400000 [00:22<00:27, 8045.67it/s] 45%|████▍     | 178630/400000 [00:23<00:27, 8015.74it/s] 45%|████▍     | 179433/400000 [00:23<00:27, 7925.70it/s] 45%|████▌     | 180227/400000 [00:23<00:27, 7913.70it/s] 45%|████▌     | 181031/400000 [00:23<00:27, 7950.01it/s] 45%|████▌     | 181845/400000 [00:23<00:27, 8004.65it/s] 46%|████▌     | 182652/400000 [00:23<00:27, 8022.12it/s] 46%|████▌     | 183455/400000 [00:23<00:27, 7994.15it/s] 46%|████▌     | 184255/400000 [00:23<00:27, 7823.23it/s] 46%|████▋     | 185039/400000 [00:23<00:27, 7763.36it/s] 46%|████▋     | 185868/400000 [00:23<00:27, 7913.33it/s] 47%|████▋     | 186698/400000 [00:24<00:26, 8022.89it/s] 47%|████▋     | 187502/400000 [00:24<00:26, 7987.15it/s] 47%|████▋     | 188321/400000 [00:24<00:26, 8044.94it/s] 47%|████▋     | 189127/400000 [00:24<00:26, 7987.25it/s] 47%|████▋     | 189927/400000 [00:24<00:26, 7908.60it/s] 48%|████▊     | 190735/400000 [00:24<00:26, 7958.05it/s] 48%|████▊     | 191532/400000 [00:24<00:26, 7853.95it/s] 48%|████▊     | 192345/400000 [00:24<00:26, 7933.29it/s] 48%|████▊     | 193175/400000 [00:24<00:25, 8038.69it/s] 48%|████▊     | 193987/400000 [00:24<00:25, 8062.03it/s] 49%|████▊     | 194821/400000 [00:25<00:25, 8141.52it/s] 49%|████▉     | 195636/400000 [00:25<00:25, 7949.65it/s] 49%|████▉     | 196436/400000 [00:25<00:25, 7964.25it/s] 49%|████▉     | 197234/400000 [00:25<00:25, 7957.60it/s] 50%|████▉     | 198031/400000 [00:25<00:25, 7771.96it/s] 50%|████▉     | 198822/400000 [00:25<00:25, 7812.39it/s] 50%|████▉     | 199605/400000 [00:25<00:25, 7768.59it/s] 50%|█████     | 200429/400000 [00:25<00:25, 7902.34it/s] 50%|█████     | 201252/400000 [00:25<00:24, 7996.69it/s] 51%|█████     | 202072/400000 [00:25<00:24, 8055.35it/s] 51%|█████     | 202892/400000 [00:26<00:24, 8096.69it/s] 51%|█████     | 203703/400000 [00:26<00:24, 7880.11it/s] 51%|█████     | 204493/400000 [00:26<00:25, 7766.69it/s] 51%|█████▏    | 205300/400000 [00:26<00:24, 7852.45it/s] 52%|█████▏    | 206107/400000 [00:26<00:24, 7913.59it/s] 52%|█████▏    | 206924/400000 [00:26<00:24, 7988.38it/s] 52%|█████▏    | 207724/400000 [00:26<00:24, 7986.88it/s] 52%|█████▏    | 208541/400000 [00:26<00:23, 8039.68it/s] 52%|█████▏    | 209347/400000 [00:26<00:23, 8043.67it/s] 53%|█████▎    | 210152/400000 [00:26<00:23, 8020.75it/s] 53%|█████▎    | 210955/400000 [00:27<00:24, 7803.01it/s] 53%|█████▎    | 211737/400000 [00:27<00:24, 7763.45it/s] 53%|█████▎    | 212529/400000 [00:27<00:24, 7807.12it/s] 53%|█████▎    | 213347/400000 [00:27<00:23, 7913.37it/s] 54%|█████▎    | 214140/400000 [00:27<00:23, 7819.42it/s] 54%|█████▎    | 214923/400000 [00:27<00:23, 7792.87it/s] 54%|█████▍    | 215703/400000 [00:27<00:23, 7696.32it/s] 54%|█████▍    | 216482/400000 [00:27<00:23, 7721.83it/s] 54%|█████▍    | 217255/400000 [00:27<00:24, 7550.79it/s] 55%|█████▍    | 218033/400000 [00:28<00:23, 7617.02it/s] 55%|█████▍    | 218840/400000 [00:28<00:23, 7746.88it/s] 55%|█████▍    | 219634/400000 [00:28<00:23, 7802.46it/s] 55%|█████▌    | 220439/400000 [00:28<00:22, 7873.48it/s] 55%|█████▌    | 221255/400000 [00:28<00:22, 7954.74it/s] 56%|█████▌    | 222052/400000 [00:28<00:22, 7905.56it/s] 56%|█████▌    | 222854/400000 [00:28<00:22, 7937.23it/s] 56%|█████▌    | 223649/400000 [00:28<00:22, 7727.29it/s] 56%|█████▌    | 224424/400000 [00:28<00:22, 7644.54it/s] 56%|█████▋    | 225237/400000 [00:28<00:22, 7782.38it/s] 57%|█████▋    | 226059/400000 [00:29<00:22, 7903.63it/s] 57%|█████▋    | 226878/400000 [00:29<00:21, 7985.01it/s] 57%|█████▋    | 227678/400000 [00:29<00:21, 7923.94it/s] 57%|█████▋    | 228472/400000 [00:29<00:21, 7874.72it/s] 57%|█████▋    | 229294/400000 [00:29<00:21, 7972.74it/s] 58%|█████▊    | 230125/400000 [00:29<00:21, 8068.79it/s] 58%|█████▊    | 230949/400000 [00:29<00:20, 8116.79it/s] 58%|█████▊    | 231762/400000 [00:29<00:21, 7994.51it/s] 58%|█████▊    | 232563/400000 [00:29<00:21, 7841.49it/s] 58%|█████▊    | 233349/400000 [00:29<00:21, 7778.95it/s] 59%|█████▊    | 234128/400000 [00:30<00:21, 7548.48it/s] 59%|█████▊    | 234938/400000 [00:30<00:21, 7705.73it/s] 59%|█████▉    | 235721/400000 [00:30<00:21, 7742.50it/s] 59%|█████▉    | 236497/400000 [00:30<00:21, 7582.14it/s] 59%|█████▉    | 237284/400000 [00:30<00:21, 7665.82it/s] 60%|█████▉    | 238094/400000 [00:30<00:20, 7788.64it/s] 60%|█████▉    | 238913/400000 [00:30<00:20, 7903.43it/s] 60%|█████▉    | 239731/400000 [00:30<00:20, 7984.05it/s] 60%|██████    | 240531/400000 [00:30<00:19, 7977.27it/s] 60%|██████    | 241330/400000 [00:30<00:19, 7937.01it/s] 61%|██████    | 242138/400000 [00:31<00:19, 7977.64it/s] 61%|██████    | 242952/400000 [00:31<00:19, 8023.94it/s] 61%|██████    | 243761/400000 [00:31<00:19, 8041.41it/s] 61%|██████    | 244566/400000 [00:31<00:19, 8012.18it/s] 61%|██████▏   | 245388/400000 [00:31<00:19, 8070.95it/s] 62%|██████▏   | 246210/400000 [00:31<00:18, 8114.46it/s] 62%|██████▏   | 247038/400000 [00:31<00:18, 8161.82it/s] 62%|██████▏   | 247855/400000 [00:31<00:18, 8146.20it/s] 62%|██████▏   | 248670/400000 [00:31<00:18, 8113.70it/s] 62%|██████▏   | 249482/400000 [00:31<00:18, 8015.25it/s] 63%|██████▎   | 250284/400000 [00:32<00:19, 7847.44it/s] 63%|██████▎   | 251091/400000 [00:32<00:18, 7912.19it/s] 63%|██████▎   | 251885/400000 [00:32<00:18, 7919.63it/s] 63%|██████▎   | 252678/400000 [00:32<00:18, 7886.58it/s] 63%|██████▎   | 253497/400000 [00:32<00:18, 7972.73it/s] 64%|██████▎   | 254326/400000 [00:32<00:18, 8063.18it/s] 64%|██████▍   | 255134/400000 [00:32<00:18, 7977.29it/s] 64%|██████▍   | 255940/400000 [00:32<00:18, 8001.23it/s] 64%|██████▍   | 256741/400000 [00:32<00:17, 7995.10it/s] 64%|██████▍   | 257541/400000 [00:33<00:17, 7981.06it/s] 65%|██████▍   | 258340/400000 [00:33<00:17, 7923.18it/s] 65%|██████▍   | 259152/400000 [00:33<00:17, 7979.24it/s] 65%|██████▍   | 259963/400000 [00:33<00:17, 8015.00it/s] 65%|██████▌   | 260765/400000 [00:33<00:17, 7946.35it/s] 65%|██████▌   | 261573/400000 [00:33<00:17, 7983.31it/s] 66%|██████▌   | 262380/400000 [00:33<00:17, 8009.08it/s] 66%|██████▌   | 263182/400000 [00:33<00:17, 7918.52it/s] 66%|██████▌   | 263975/400000 [00:33<00:17, 7831.01it/s] 66%|██████▌   | 264759/400000 [00:33<00:17, 7829.43it/s] 66%|██████▋   | 265563/400000 [00:34<00:17, 7889.03it/s] 67%|██████▋   | 266375/400000 [00:34<00:16, 7954.26it/s] 67%|██████▋   | 267196/400000 [00:34<00:16, 8028.13it/s] 67%|██████▋   | 268016/400000 [00:34<00:16, 8078.01it/s] 67%|██████▋   | 268825/400000 [00:34<00:16, 8050.23it/s] 67%|██████▋   | 269648/400000 [00:34<00:16, 8103.31it/s] 68%|██████▊   | 270461/400000 [00:34<00:15, 8106.34it/s] 68%|██████▊   | 271272/400000 [00:34<00:15, 8095.66it/s] 68%|██████▊   | 272092/400000 [00:34<00:15, 8124.93it/s] 68%|██████▊   | 272905/400000 [00:34<00:15, 8035.79it/s] 68%|██████▊   | 273725/400000 [00:35<00:15, 8082.29it/s] 69%|██████▊   | 274534/400000 [00:35<00:15, 8068.34it/s] 69%|██████▉   | 275342/400000 [00:35<00:15, 7979.11it/s] 69%|██████▉   | 276141/400000 [00:35<00:15, 7869.60it/s] 69%|██████▉   | 276929/400000 [00:35<00:16, 7670.31it/s] 69%|██████▉   | 277698/400000 [00:35<00:15, 7667.33it/s] 70%|██████▉   | 278522/400000 [00:35<00:15, 7829.44it/s] 70%|██████▉   | 279356/400000 [00:35<00:15, 7973.65it/s] 70%|███████   | 280186/400000 [00:35<00:14, 8067.69it/s] 70%|███████   | 280995/400000 [00:35<00:14, 7984.57it/s] 70%|███████   | 281795/400000 [00:36<00:14, 7969.47it/s] 71%|███████   | 282593/400000 [00:36<00:14, 7954.93it/s] 71%|███████   | 283406/400000 [00:36<00:14, 8006.53it/s] 71%|███████   | 284230/400000 [00:36<00:14, 8072.80it/s] 71%|███████▏  | 285038/400000 [00:36<00:14, 8039.65it/s] 71%|███████▏  | 285846/400000 [00:36<00:14, 8049.33it/s] 72%|███████▏  | 286653/400000 [00:36<00:14, 8055.29it/s] 72%|███████▏  | 287459/400000 [00:36<00:14, 7978.44it/s] 72%|███████▏  | 288277/400000 [00:36<00:13, 8037.22it/s] 72%|███████▏  | 289082/400000 [00:36<00:13, 8002.52it/s] 72%|███████▏  | 289883/400000 [00:37<00:14, 7780.53it/s] 73%|███████▎  | 290663/400000 [00:37<00:14, 7774.29it/s] 73%|███████▎  | 291472/400000 [00:37<00:13, 7864.99it/s] 73%|███████▎  | 292299/400000 [00:37<00:13, 7981.16it/s] 73%|███████▎  | 293104/400000 [00:37<00:13, 7998.82it/s] 73%|███████▎  | 293909/400000 [00:37<00:13, 8013.04it/s] 74%|███████▎  | 294711/400000 [00:37<00:13, 7993.49it/s] 74%|███████▍  | 295511/400000 [00:37<00:13, 7966.56it/s] 74%|███████▍  | 296340/400000 [00:37<00:12, 8060.12it/s] 74%|███████▍  | 297147/400000 [00:37<00:12, 8054.13it/s] 74%|███████▍  | 297953/400000 [00:38<00:12, 8015.08it/s] 75%|███████▍  | 298755/400000 [00:38<00:12, 7969.15it/s] 75%|███████▍  | 299553/400000 [00:38<00:12, 7916.74it/s] 75%|███████▌  | 300368/400000 [00:38<00:12, 7983.26it/s] 75%|███████▌  | 301167/400000 [00:38<00:12, 7949.59it/s] 75%|███████▌  | 301987/400000 [00:38<00:12, 8020.31it/s] 76%|███████▌  | 302790/400000 [00:38<00:12, 7897.56it/s] 76%|███████▌  | 303581/400000 [00:38<00:12, 7697.65it/s] 76%|███████▌  | 304395/400000 [00:38<00:12, 7824.20it/s] 76%|███████▋  | 305198/400000 [00:38<00:12, 7882.95it/s] 77%|███████▋  | 306004/400000 [00:39<00:11, 7935.21it/s] 77%|███████▋  | 306828/400000 [00:39<00:11, 8022.52it/s] 77%|███████▋  | 307651/400000 [00:39<00:11, 8082.44it/s] 77%|███████▋  | 308460/400000 [00:39<00:11, 8079.71it/s] 77%|███████▋  | 309269/400000 [00:39<00:11, 7972.18it/s] 78%|███████▊  | 310083/400000 [00:39<00:11, 8020.06it/s] 78%|███████▊  | 310886/400000 [00:39<00:11, 7934.81it/s] 78%|███████▊  | 311681/400000 [00:39<00:11, 7868.79it/s] 78%|███████▊  | 312497/400000 [00:39<00:11, 7953.55it/s] 78%|███████▊  | 313300/400000 [00:40<00:10, 7975.97it/s] 79%|███████▊  | 314099/400000 [00:40<00:10, 7905.02it/s] 79%|███████▊  | 314907/400000 [00:40<00:10, 7954.49it/s] 79%|███████▉  | 315703/400000 [00:40<00:10, 7936.04it/s] 79%|███████▉  | 316497/400000 [00:40<00:10, 7697.90it/s] 79%|███████▉  | 317269/400000 [00:40<00:10, 7533.70it/s] 80%|███████▉  | 318025/400000 [00:40<00:11, 7445.75it/s] 80%|███████▉  | 318834/400000 [00:40<00:10, 7627.03it/s] 80%|███████▉  | 319652/400000 [00:40<00:10, 7782.75it/s] 80%|████████  | 320479/400000 [00:40<00:10, 7922.64it/s] 80%|████████  | 321274/400000 [00:41<00:10, 7851.76it/s] 81%|████████  | 322071/400000 [00:41<00:09, 7886.00it/s] 81%|████████  | 322864/400000 [00:41<00:09, 7896.84it/s] 81%|████████  | 323672/400000 [00:41<00:09, 7948.49it/s] 81%|████████  | 324497/400000 [00:41<00:09, 8033.79it/s] 81%|████████▏ | 325302/400000 [00:41<00:09, 8006.06it/s] 82%|████████▏ | 326126/400000 [00:41<00:09, 8073.03it/s] 82%|████████▏ | 326944/400000 [00:41<00:09, 8103.34it/s] 82%|████████▏ | 327766/400000 [00:41<00:08, 8135.80it/s] 82%|████████▏ | 328580/400000 [00:41<00:08, 8130.93it/s] 82%|████████▏ | 329394/400000 [00:42<00:08, 7951.77it/s] 83%|████████▎ | 330191/400000 [00:42<00:09, 7621.34it/s] 83%|████████▎ | 330981/400000 [00:42<00:08, 7702.37it/s] 83%|████████▎ | 331770/400000 [00:42<00:08, 7756.76it/s] 83%|████████▎ | 332574/400000 [00:42<00:08, 7837.07it/s] 83%|████████▎ | 333367/400000 [00:42<00:08, 7863.30it/s] 84%|████████▎ | 334179/400000 [00:42<00:08, 7938.59it/s] 84%|████████▎ | 334974/400000 [00:42<00:08, 7925.27it/s] 84%|████████▍ | 335782/400000 [00:42<00:08, 7970.29it/s] 84%|████████▍ | 336599/400000 [00:42<00:07, 8027.07it/s] 84%|████████▍ | 337403/400000 [00:43<00:07, 7989.11it/s] 85%|████████▍ | 338204/400000 [00:43<00:07, 7994.04it/s] 85%|████████▍ | 339004/400000 [00:43<00:07, 7963.59it/s] 85%|████████▍ | 339801/400000 [00:43<00:07, 7814.02it/s] 85%|████████▌ | 340584/400000 [00:43<00:07, 7776.32it/s] 85%|████████▌ | 341363/400000 [00:43<00:07, 7730.30it/s] 86%|████████▌ | 342173/400000 [00:43<00:07, 7834.87it/s] 86%|████████▌ | 342958/400000 [00:43<00:07, 7800.35it/s] 86%|████████▌ | 343740/400000 [00:43<00:07, 7804.11it/s] 86%|████████▌ | 344551/400000 [00:43<00:07, 7893.15it/s] 86%|████████▋ | 345341/400000 [00:44<00:06, 7881.66it/s] 87%|████████▋ | 346146/400000 [00:44<00:06, 7930.81it/s] 87%|████████▋ | 346967/400000 [00:44<00:06, 8011.58it/s] 87%|████████▋ | 347780/400000 [00:44<00:06, 8045.40it/s] 87%|████████▋ | 348597/400000 [00:44<00:06, 8079.49it/s] 87%|████████▋ | 349406/400000 [00:44<00:06, 8031.24it/s] 88%|████████▊ | 350230/400000 [00:44<00:06, 8090.03it/s] 88%|████████▊ | 351047/400000 [00:44<00:06, 8111.49it/s] 88%|████████▊ | 351865/400000 [00:44<00:05, 8130.98it/s] 88%|████████▊ | 352683/400000 [00:44<00:05, 8143.52it/s] 88%|████████▊ | 353498/400000 [00:45<00:05, 8079.35it/s] 89%|████████▊ | 354316/400000 [00:45<00:05, 8106.76it/s] 89%|████████▉ | 355127/400000 [00:45<00:05, 8053.66it/s] 89%|████████▉ | 355933/400000 [00:45<00:05, 7904.49it/s] 89%|████████▉ | 356725/400000 [00:45<00:05, 7658.53it/s] 89%|████████▉ | 357508/400000 [00:45<00:05, 7708.91it/s] 90%|████████▉ | 358299/400000 [00:45<00:05, 7767.89it/s] 90%|████████▉ | 359078/400000 [00:45<00:05, 7673.67it/s] 90%|████████▉ | 359847/400000 [00:45<00:05, 7633.91it/s] 90%|█████████ | 360612/400000 [00:46<00:05, 7554.96it/s] 90%|█████████ | 361388/400000 [00:46<00:05, 7612.90it/s] 91%|█████████ | 362210/400000 [00:46<00:04, 7784.87it/s] 91%|█████████ | 363013/400000 [00:46<00:04, 7855.52it/s] 91%|█████████ | 363800/400000 [00:46<00:04, 7804.18it/s] 91%|█████████ | 364612/400000 [00:46<00:04, 7894.32it/s] 91%|█████████▏| 365403/400000 [00:46<00:04, 7874.12it/s] 92%|█████████▏| 366209/400000 [00:46<00:04, 7927.76it/s] 92%|█████████▏| 367003/400000 [00:46<00:04, 7875.90it/s] 92%|█████████▏| 367816/400000 [00:46<00:04, 7949.47it/s] 92%|█████████▏| 368612/400000 [00:47<00:03, 7897.46it/s] 92%|█████████▏| 369403/400000 [00:47<00:03, 7700.17it/s] 93%|█████████▎| 370216/400000 [00:47<00:03, 7823.13it/s] 93%|█████████▎| 371010/400000 [00:47<00:03, 7856.28it/s] 93%|█████████▎| 371797/400000 [00:47<00:03, 7655.10it/s] 93%|█████████▎| 372618/400000 [00:47<00:03, 7813.41it/s] 93%|█████████▎| 373402/400000 [00:47<00:03, 7662.70it/s] 94%|█████████▎| 374171/400000 [00:47<00:03, 7630.14it/s] 94%|█████████▎| 374937/400000 [00:47<00:03, 7637.33it/s] 94%|█████████▍| 375742/400000 [00:47<00:03, 7751.43it/s] 94%|█████████▍| 376519/400000 [00:48<00:03, 7675.45it/s] 94%|█████████▍| 377304/400000 [00:48<00:02, 7725.86it/s] 95%|█████████▍| 378123/400000 [00:48<00:02, 7856.96it/s] 95%|█████████▍| 378947/400000 [00:48<00:02, 7966.79it/s] 95%|█████████▍| 379745/400000 [00:48<00:02, 7929.12it/s] 95%|█████████▌| 380539/400000 [00:48<00:02, 7797.83it/s] 95%|█████████▌| 381320/400000 [00:48<00:02, 7722.84it/s] 96%|█████████▌| 382094/400000 [00:48<00:02, 7488.68it/s] 96%|█████████▌| 382885/400000 [00:48<00:02, 7610.14it/s] 96%|█████████▌| 383650/400000 [00:48<00:02, 7621.06it/s] 96%|█████████▌| 384464/400000 [00:49<00:01, 7768.28it/s] 96%|█████████▋| 385249/400000 [00:49<00:01, 7791.58it/s] 97%|█████████▋| 386061/400000 [00:49<00:01, 7885.07it/s] 97%|█████████▋| 386851/400000 [00:49<00:01, 7881.82it/s] 97%|█████████▋| 387658/400000 [00:49<00:01, 7935.02it/s] 97%|█████████▋| 388457/400000 [00:49<00:01, 7949.36it/s] 97%|█████████▋| 389253/400000 [00:49<00:01, 7941.50it/s] 98%|█████████▊| 390077/400000 [00:49<00:01, 8026.36it/s] 98%|█████████▊| 390892/400000 [00:49<00:01, 8062.50it/s] 98%|█████████▊| 391699/400000 [00:49<00:01, 8041.53it/s] 98%|█████████▊| 392504/400000 [00:50<00:00, 8002.77it/s] 98%|█████████▊| 393305/400000 [00:50<00:00, 7905.57it/s] 99%|█████████▊| 394096/400000 [00:50<00:00, 7847.31it/s] 99%|█████████▊| 394882/400000 [00:50<00:00, 7703.76it/s] 99%|█████████▉| 395683/400000 [00:50<00:00, 7791.72it/s] 99%|█████████▉| 396507/400000 [00:50<00:00, 7920.60it/s] 99%|█████████▉| 397301/400000 [00:50<00:00, 7923.35it/s]100%|█████████▉| 398127/400000 [00:50<00:00, 8018.32it/s]100%|█████████▉| 398941/400000 [00:50<00:00, 8052.95it/s]100%|█████████▉| 399747/400000 [00:50<00:00, 7863.68it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7838.46it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f06f3af0160> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011012267145476116 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010884651571611895 	 Accuracy: 71

  model saves at 71% accuracy 

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

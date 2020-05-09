
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4c487a1470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 23:14:13.061288
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 23:14:13.065152
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 23:14:13.068380
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 23:14:13.071571
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4c40af1438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354702.9688
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 266383.5312
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 146905.0625
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 73038.2812
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 36086.7656
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 19461.0273
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 11568.3799
Epoch 8/10

1/1 [==============================] - 0s 107ms/step - loss: 7511.3853
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 5244.2627
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 3959.4636

  #### Inference Need return ypred, ytrue ######################### 
[[ -0.5573888   12.877762    12.682869    13.005272    11.803536
   13.1597395   10.483778    12.011764    11.985515    12.097491
   12.214314    12.632666    12.287338    11.92303     11.014772
   10.446445    10.970652    12.161601    11.099544    10.307882
   11.124381     9.970307    12.570681    10.884713    10.572218
   10.824586    12.150446    12.164619    11.025105    10.474505
   11.130184    12.562965    11.187426    11.782251    14.623584
   11.523708    13.025284    12.511553    11.01859     12.548501
   12.656008    11.202018    10.663034    14.214154    12.097865
   11.92814     12.893335    12.208223    11.303542    12.549642
   12.113747    12.693276    11.749808    10.922002    12.61356
   10.73822     12.514776    11.170928    12.63697     11.551549
    0.95345765   2.239666     0.79383564   0.9799913   -0.23446026
   -0.69777334   1.4520638   -0.11361772   0.39088377  -0.21046638
    0.8152692   -1.8828778   -0.65615034   0.5279046   -1.1424158
    1.5538776   -1.0055983   -0.742738     0.6519505    1.0478454
    0.9293024   -1.541982     1.528856    -2.105391     0.744774
   -1.7714415   -1.4331269    0.12769313  -1.1553086    0.817999
   -1.2961638    0.4219819    0.614964    -0.6241512    0.82121
    1.0688962   -0.8932531    0.40135396  -2.0600352   -0.03292596
    1.1225674   -0.38546956   0.7463937    0.7705115   -0.2569087
    2.887289    -1.2194395    0.65364397  -0.01849753  -1.002207
    0.10219966  -0.75184923   1.9785609   -0.45662808  -2.5506587
    0.44443312  -2.242987     0.09261623   0.33034837  -1.6458068
   -0.17757162   1.0740563   -0.8891134    0.218523     1.585841
   -0.4122432   -0.67438304  -0.13573152  -0.72143984  -1.3156363
   -0.9727982    1.1987712    0.3793366   -0.8491866    0.54054403
    0.48044586   1.6965365    0.04853463  -1.3618591   -0.33110255
   -0.8137227   -1.9309893   -0.6067233   -0.08749321   1.0827957
   -0.5389396   -0.50870305   0.6173437   -0.5660875   -0.87950456
   -0.55813247  -1.1230334   -0.27253905   0.06102136  -0.62195104
   -1.1366875    0.67085344   1.1857836   -0.5383202   -0.05336219
   -1.1458344    1.8904046    0.13022012  -0.6195744    0.9671363
    0.33845305  -0.09751451  -1.4049687   -2.760155    -2.6131792
    0.6233532    0.65864575   0.27684394  -0.23459485   0.75388575
   -3.6877577    1.2705938   -0.7868542   -1.5832951   -0.53202164
    0.34488177  10.315831    11.501525    11.633456    11.689438
    8.120229    10.651022    10.85998     11.907315     8.786927
   11.811278    11.326006    10.426227     9.856119    11.049722
   10.203391    12.957371    10.595449     9.336638    11.27042
   10.553159     9.164191    13.424192    12.65645     14.304094
    9.122963    10.93292     10.880804    10.753154    11.438597
    9.410286    11.565288     9.129765    11.411933    12.901868
   10.472466    10.81847     10.70065     11.174591    12.018059
   12.383595    10.20371     11.31397     13.766698    12.254012
   11.446243    10.032288    12.661521    10.033832    12.619286
   11.038371    12.871915    12.008907    10.705748    11.974254
   12.54993     10.849551    10.95286     11.391168    10.6677685
    2.963894     0.3189659    1.2852556    1.8031042    0.21571434
    0.1143707    0.7667929    3.02103      0.53507924   0.2587117
    0.4015637    1.627801     0.2639957    0.6381916    1.3719485
    0.06153953   0.42812425   0.38930577   2.2173977    0.7824603
    0.18106318   0.854691     0.45950282   0.91354895   0.8701007
    3.2912984    0.10272914   0.50169843   0.5980535    0.28053582
    0.652724     2.5881386    1.3338649    0.9714922    0.92849696
    2.6721764    1.5652492    1.5613259    1.4751269    2.346368
    1.089358     0.14836252   0.5581259    0.92643124   0.45471025
    0.8704287    1.82532      0.16216516   0.17024106   1.1174848
    1.7455819    1.9761       0.5065221    0.35395968   3.796966
    1.6860502    0.30778933   1.714051     0.19695288   2.273118
    1.8684008    1.8116606    1.9545306    2.0821838    2.0845523
    0.17752886   1.8300085    1.2224135    1.2658257    0.39193833
    1.6292295    2.080317     1.8102794    0.8837985    0.78240097
    0.5029959    0.3379948    1.6532477    0.08682448   1.7039177
    0.13212335   1.7806677    1.2515358    2.2997923    0.3427385
    0.02490759   1.5654048    0.46431077   2.5023036    0.68663085
    1.017153     0.14066142   0.3873847    2.146283     0.20337856
    0.35730624   1.2680532    1.6825211    3.457859     0.20420462
    0.4845177    0.16742921   0.36042833   0.20292586   2.3105698
    1.1098619    2.129641     1.401984     1.8647803    0.60898185
    0.464213     0.3027293    1.4827046    0.31262362   2.6188834
    0.69933426   2.1693792    0.44586802   1.6655505    0.3869887
    6.434759   -12.530979   -10.115805  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 23:14:22.096475
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.6733
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 23:14:22.101020
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8245.46
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 23:14:22.104870
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.0757
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 23:14:22.108663
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -737.44
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139964938507376
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139964047057808
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139964047058312
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139964047058816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139964047059320
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139964047059824

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4c2e51ca20> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.614088
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.585700
grad_step = 000002, loss = 0.560056
grad_step = 000003, loss = 0.532858
grad_step = 000004, loss = 0.502006
grad_step = 000005, loss = 0.469697
grad_step = 000006, loss = 0.441793
grad_step = 000007, loss = 0.418055
grad_step = 000008, loss = 0.388027
grad_step = 000009, loss = 0.369557
grad_step = 000010, loss = 0.351064
grad_step = 000011, loss = 0.325201
grad_step = 000012, loss = 0.302005
grad_step = 000013, loss = 0.286391
grad_step = 000014, loss = 0.275867
grad_step = 000015, loss = 0.266905
grad_step = 000016, loss = 0.259082
grad_step = 000017, loss = 0.252153
grad_step = 000018, loss = 0.243351
grad_step = 000019, loss = 0.232906
grad_step = 000020, loss = 0.222930
grad_step = 000021, loss = 0.213068
grad_step = 000022, loss = 0.203310
grad_step = 000023, loss = 0.193877
grad_step = 000024, loss = 0.185643
grad_step = 000025, loss = 0.178790
grad_step = 000026, loss = 0.172290
grad_step = 000027, loss = 0.165245
grad_step = 000028, loss = 0.157909
grad_step = 000029, loss = 0.150693
grad_step = 000030, loss = 0.143598
grad_step = 000031, loss = 0.136626
grad_step = 000032, loss = 0.129964
grad_step = 000033, loss = 0.123739
grad_step = 000034, loss = 0.118021
grad_step = 000035, loss = 0.112762
grad_step = 000036, loss = 0.107508
grad_step = 000037, loss = 0.102462
grad_step = 000038, loss = 0.097641
grad_step = 000039, loss = 0.092780
grad_step = 000040, loss = 0.087879
grad_step = 000041, loss = 0.083237
grad_step = 000042, loss = 0.078884
grad_step = 000043, loss = 0.074684
grad_step = 000044, loss = 0.070680
grad_step = 000045, loss = 0.066898
grad_step = 000046, loss = 0.063320
grad_step = 000047, loss = 0.059874
grad_step = 000048, loss = 0.056521
grad_step = 000049, loss = 0.053349
grad_step = 000050, loss = 0.050339
grad_step = 000051, loss = 0.047387
grad_step = 000052, loss = 0.044570
grad_step = 000053, loss = 0.041923
grad_step = 000054, loss = 0.039379
grad_step = 000055, loss = 0.036952
grad_step = 000056, loss = 0.034653
grad_step = 000057, loss = 0.032469
grad_step = 000058, loss = 0.030365
grad_step = 000059, loss = 0.028357
grad_step = 000060, loss = 0.026475
grad_step = 000061, loss = 0.024679
grad_step = 000062, loss = 0.022972
grad_step = 000063, loss = 0.021368
grad_step = 000064, loss = 0.019860
grad_step = 000065, loss = 0.018433
grad_step = 000066, loss = 0.017090
grad_step = 000067, loss = 0.015834
grad_step = 000068, loss = 0.014645
grad_step = 000069, loss = 0.013531
grad_step = 000070, loss = 0.012503
grad_step = 000071, loss = 0.011545
grad_step = 000072, loss = 0.010653
grad_step = 000073, loss = 0.009830
grad_step = 000074, loss = 0.009067
grad_step = 000075, loss = 0.008360
grad_step = 000076, loss = 0.007715
grad_step = 000077, loss = 0.007119
grad_step = 000078, loss = 0.006572
grad_step = 000079, loss = 0.006074
grad_step = 000080, loss = 0.005625
grad_step = 000081, loss = 0.005212
grad_step = 000082, loss = 0.004836
grad_step = 000083, loss = 0.004499
grad_step = 000084, loss = 0.004196
grad_step = 000085, loss = 0.003926
grad_step = 000086, loss = 0.003684
grad_step = 000087, loss = 0.003467
grad_step = 000088, loss = 0.003276
grad_step = 000089, loss = 0.003107
grad_step = 000090, loss = 0.002957
grad_step = 000091, loss = 0.002827
grad_step = 000092, loss = 0.002716
grad_step = 000093, loss = 0.002620
grad_step = 000094, loss = 0.002536
grad_step = 000095, loss = 0.002466
grad_step = 000096, loss = 0.002406
grad_step = 000097, loss = 0.002356
grad_step = 000098, loss = 0.002314
grad_step = 000099, loss = 0.002281
grad_step = 000100, loss = 0.002253
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002230
grad_step = 000102, loss = 0.002211
grad_step = 000103, loss = 0.002197
grad_step = 000104, loss = 0.002185
grad_step = 000105, loss = 0.002176
grad_step = 000106, loss = 0.002170
grad_step = 000107, loss = 0.002167
grad_step = 000108, loss = 0.002162
grad_step = 000109, loss = 0.002156
grad_step = 000110, loss = 0.002149
grad_step = 000111, loss = 0.002145
grad_step = 000112, loss = 0.002144
grad_step = 000113, loss = 0.002142
grad_step = 000114, loss = 0.002137
grad_step = 000115, loss = 0.002132
grad_step = 000116, loss = 0.002127
grad_step = 000117, loss = 0.002123
grad_step = 000118, loss = 0.002120
grad_step = 000119, loss = 0.002116
grad_step = 000120, loss = 0.002111
grad_step = 000121, loss = 0.002105
grad_step = 000122, loss = 0.002099
grad_step = 000123, loss = 0.002093
grad_step = 000124, loss = 0.002089
grad_step = 000125, loss = 0.002084
grad_step = 000126, loss = 0.002079
grad_step = 000127, loss = 0.002074
grad_step = 000128, loss = 0.002069
grad_step = 000129, loss = 0.002064
grad_step = 000130, loss = 0.002059
grad_step = 000131, loss = 0.002054
grad_step = 000132, loss = 0.002048
grad_step = 000133, loss = 0.002043
grad_step = 000134, loss = 0.002038
grad_step = 000135, loss = 0.002032
grad_step = 000136, loss = 0.002027
grad_step = 000137, loss = 0.002022
grad_step = 000138, loss = 0.002017
grad_step = 000139, loss = 0.002013
grad_step = 000140, loss = 0.002010
grad_step = 000141, loss = 0.002007
grad_step = 000142, loss = 0.002005
grad_step = 000143, loss = 0.002001
grad_step = 000144, loss = 0.001997
grad_step = 000145, loss = 0.001990
grad_step = 000146, loss = 0.001981
grad_step = 000147, loss = 0.001971
grad_step = 000148, loss = 0.001960
grad_step = 000149, loss = 0.001951
grad_step = 000150, loss = 0.001943
grad_step = 000151, loss = 0.001937
grad_step = 000152, loss = 0.001933
grad_step = 000153, loss = 0.001929
grad_step = 000154, loss = 0.001927
grad_step = 000155, loss = 0.001929
grad_step = 000156, loss = 0.001940
grad_step = 000157, loss = 0.001966
grad_step = 000158, loss = 0.002027
grad_step = 000159, loss = 0.002112
grad_step = 000160, loss = 0.002209
grad_step = 000161, loss = 0.002142
grad_step = 000162, loss = 0.001948
grad_step = 000163, loss = 0.001880
grad_step = 000164, loss = 0.001990
grad_step = 000165, loss = 0.002038
grad_step = 000166, loss = 0.001941
grad_step = 000167, loss = 0.001910
grad_step = 000168, loss = 0.001934
grad_step = 000169, loss = 0.001906
grad_step = 000170, loss = 0.001877
grad_step = 000171, loss = 0.001916
grad_step = 000172, loss = 0.001918
grad_step = 000173, loss = 0.001842
grad_step = 000174, loss = 0.001845
grad_step = 000175, loss = 0.001895
grad_step = 000176, loss = 0.001864
grad_step = 000177, loss = 0.001824
grad_step = 000178, loss = 0.001838
grad_step = 000179, loss = 0.001842
grad_step = 000180, loss = 0.001816
grad_step = 000181, loss = 0.001817
grad_step = 000182, loss = 0.001835
grad_step = 000183, loss = 0.001818
grad_step = 000184, loss = 0.001792
grad_step = 000185, loss = 0.001794
grad_step = 000186, loss = 0.001800
grad_step = 000187, loss = 0.001784
grad_step = 000188, loss = 0.001770
grad_step = 000189, loss = 0.001773
grad_step = 000190, loss = 0.001776
grad_step = 000191, loss = 0.001765
grad_step = 000192, loss = 0.001755
grad_step = 000193, loss = 0.001761
grad_step = 000194, loss = 0.001790
grad_step = 000195, loss = 0.001877
grad_step = 000196, loss = 0.002175
grad_step = 000197, loss = 0.002650
grad_step = 000198, loss = 0.003093
grad_step = 000199, loss = 0.002167
grad_step = 000200, loss = 0.001939
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002357
grad_step = 000202, loss = 0.002110
grad_step = 000203, loss = 0.001956
grad_step = 000204, loss = 0.002010
grad_step = 000205, loss = 0.001997
grad_step = 000206, loss = 0.001909
grad_step = 000207, loss = 0.001866
grad_step = 000208, loss = 0.001949
grad_step = 000209, loss = 0.001799
grad_step = 000210, loss = 0.001852
grad_step = 000211, loss = 0.001888
grad_step = 000212, loss = 0.001737
grad_step = 000213, loss = 0.001880
grad_step = 000214, loss = 0.001791
grad_step = 000215, loss = 0.001754
grad_step = 000216, loss = 0.001851
grad_step = 000217, loss = 0.001723
grad_step = 000218, loss = 0.001784
grad_step = 000219, loss = 0.001774
grad_step = 000220, loss = 0.001717
grad_step = 000221, loss = 0.001775
grad_step = 000222, loss = 0.001720
grad_step = 000223, loss = 0.001737
grad_step = 000224, loss = 0.001737
grad_step = 000225, loss = 0.001709
grad_step = 000226, loss = 0.001736
grad_step = 000227, loss = 0.001702
grad_step = 000228, loss = 0.001712
grad_step = 000229, loss = 0.001715
grad_step = 000230, loss = 0.001686
grad_step = 000231, loss = 0.001708
grad_step = 000232, loss = 0.001692
grad_step = 000233, loss = 0.001681
grad_step = 000234, loss = 0.001696
grad_step = 000235, loss = 0.001678
grad_step = 000236, loss = 0.001678
grad_step = 000237, loss = 0.001679
grad_step = 000238, loss = 0.001670
grad_step = 000239, loss = 0.001673
grad_step = 000240, loss = 0.001665
grad_step = 000241, loss = 0.001663
grad_step = 000242, loss = 0.001666
grad_step = 000243, loss = 0.001655
grad_step = 000244, loss = 0.001654
grad_step = 000245, loss = 0.001655
grad_step = 000246, loss = 0.001648
grad_step = 000247, loss = 0.001647
grad_step = 000248, loss = 0.001645
grad_step = 000249, loss = 0.001639
grad_step = 000250, loss = 0.001640
grad_step = 000251, loss = 0.001638
grad_step = 000252, loss = 0.001633
grad_step = 000253, loss = 0.001632
grad_step = 000254, loss = 0.001630
grad_step = 000255, loss = 0.001625
grad_step = 000256, loss = 0.001625
grad_step = 000257, loss = 0.001624
grad_step = 000258, loss = 0.001620
grad_step = 000259, loss = 0.001619
grad_step = 000260, loss = 0.001619
grad_step = 000261, loss = 0.001618
grad_step = 000262, loss = 0.001620
grad_step = 000263, loss = 0.001629
grad_step = 000264, loss = 0.001648
grad_step = 000265, loss = 0.001700
grad_step = 000266, loss = 0.001791
grad_step = 000267, loss = 0.001999
grad_step = 000268, loss = 0.002050
grad_step = 000269, loss = 0.002072
grad_step = 000270, loss = 0.001771
grad_step = 000271, loss = 0.001630
grad_step = 000272, loss = 0.001759
grad_step = 000273, loss = 0.001856
grad_step = 000274, loss = 0.001753
grad_step = 000275, loss = 0.001604
grad_step = 000276, loss = 0.001675
grad_step = 000277, loss = 0.001793
grad_step = 000278, loss = 0.001755
grad_step = 000279, loss = 0.001693
grad_step = 000280, loss = 0.001622
grad_step = 000281, loss = 0.001610
grad_step = 000282, loss = 0.001684
grad_step = 000283, loss = 0.001703
grad_step = 000284, loss = 0.001643
grad_step = 000285, loss = 0.001599
grad_step = 000286, loss = 0.001609
grad_step = 000287, loss = 0.001630
grad_step = 000288, loss = 0.001643
grad_step = 000289, loss = 0.001626
grad_step = 000290, loss = 0.001583
grad_step = 000291, loss = 0.001586
grad_step = 000292, loss = 0.001613
grad_step = 000293, loss = 0.001612
grad_step = 000294, loss = 0.001604
grad_step = 000295, loss = 0.001587
grad_step = 000296, loss = 0.001571
grad_step = 000297, loss = 0.001578
grad_step = 000298, loss = 0.001592
grad_step = 000299, loss = 0.001594
grad_step = 000300, loss = 0.001586
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001579
grad_step = 000302, loss = 0.001566
grad_step = 000303, loss = 0.001562
grad_step = 000304, loss = 0.001568
grad_step = 000305, loss = 0.001571
grad_step = 000306, loss = 0.001574
grad_step = 000307, loss = 0.001576
grad_step = 000308, loss = 0.001574
grad_step = 000309, loss = 0.001568
grad_step = 000310, loss = 0.001565
grad_step = 000311, loss = 0.001560
grad_step = 000312, loss = 0.001555
grad_step = 000313, loss = 0.001552
grad_step = 000314, loss = 0.001550
grad_step = 000315, loss = 0.001547
grad_step = 000316, loss = 0.001546
grad_step = 000317, loss = 0.001546
grad_step = 000318, loss = 0.001545
grad_step = 000319, loss = 0.001545
grad_step = 000320, loss = 0.001546
grad_step = 000321, loss = 0.001548
grad_step = 000322, loss = 0.001553
grad_step = 000323, loss = 0.001565
grad_step = 000324, loss = 0.001586
grad_step = 000325, loss = 0.001637
grad_step = 000326, loss = 0.001711
grad_step = 000327, loss = 0.001860
grad_step = 000328, loss = 0.001957
grad_step = 000329, loss = 0.002040
grad_step = 000330, loss = 0.001853
grad_step = 000331, loss = 0.001630
grad_step = 000332, loss = 0.001536
grad_step = 000333, loss = 0.001632
grad_step = 000334, loss = 0.001745
grad_step = 000335, loss = 0.001681
grad_step = 000336, loss = 0.001558
grad_step = 000337, loss = 0.001546
grad_step = 000338, loss = 0.001628
grad_step = 000339, loss = 0.001648
grad_step = 000340, loss = 0.001566
grad_step = 000341, loss = 0.001528
grad_step = 000342, loss = 0.001574
grad_step = 000343, loss = 0.001602
grad_step = 000344, loss = 0.001565
grad_step = 000345, loss = 0.001523
grad_step = 000346, loss = 0.001539
grad_step = 000347, loss = 0.001570
grad_step = 000348, loss = 0.001556
grad_step = 000349, loss = 0.001523
grad_step = 000350, loss = 0.001519
grad_step = 000351, loss = 0.001540
grad_step = 000352, loss = 0.001548
grad_step = 000353, loss = 0.001527
grad_step = 000354, loss = 0.001510
grad_step = 000355, loss = 0.001516
grad_step = 000356, loss = 0.001528
grad_step = 000357, loss = 0.001528
grad_step = 000358, loss = 0.001513
grad_step = 000359, loss = 0.001504
grad_step = 000360, loss = 0.001506
grad_step = 000361, loss = 0.001513
grad_step = 000362, loss = 0.001515
grad_step = 000363, loss = 0.001508
grad_step = 000364, loss = 0.001500
grad_step = 000365, loss = 0.001496
grad_step = 000366, loss = 0.001497
grad_step = 000367, loss = 0.001501
grad_step = 000368, loss = 0.001502
grad_step = 000369, loss = 0.001499
grad_step = 000370, loss = 0.001494
grad_step = 000371, loss = 0.001490
grad_step = 000372, loss = 0.001487
grad_step = 000373, loss = 0.001486
grad_step = 000374, loss = 0.001487
grad_step = 000375, loss = 0.001487
grad_step = 000376, loss = 0.001488
grad_step = 000377, loss = 0.001488
grad_step = 000378, loss = 0.001487
grad_step = 000379, loss = 0.001485
grad_step = 000380, loss = 0.001483
grad_step = 000381, loss = 0.001481
grad_step = 000382, loss = 0.001479
grad_step = 000383, loss = 0.001477
grad_step = 000384, loss = 0.001475
grad_step = 000385, loss = 0.001474
grad_step = 000386, loss = 0.001473
grad_step = 000387, loss = 0.001472
grad_step = 000388, loss = 0.001473
grad_step = 000389, loss = 0.001474
grad_step = 000390, loss = 0.001478
grad_step = 000391, loss = 0.001485
grad_step = 000392, loss = 0.001501
grad_step = 000393, loss = 0.001523
grad_step = 000394, loss = 0.001569
grad_step = 000395, loss = 0.001617
grad_step = 000396, loss = 0.001709
grad_step = 000397, loss = 0.001737
grad_step = 000398, loss = 0.001769
grad_step = 000399, loss = 0.001654
grad_step = 000400, loss = 0.001537
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001461
grad_step = 000402, loss = 0.001484
grad_step = 000403, loss = 0.001556
grad_step = 000404, loss = 0.001573
grad_step = 000405, loss = 0.001530
grad_step = 000406, loss = 0.001464
grad_step = 000407, loss = 0.001450
grad_step = 000408, loss = 0.001485
grad_step = 000409, loss = 0.001517
grad_step = 000410, loss = 0.001520
grad_step = 000411, loss = 0.001487
grad_step = 000412, loss = 0.001455
grad_step = 000413, loss = 0.001439
grad_step = 000414, loss = 0.001446
grad_step = 000415, loss = 0.001466
grad_step = 000416, loss = 0.001480
grad_step = 000417, loss = 0.001484
grad_step = 000418, loss = 0.001470
grad_step = 000419, loss = 0.001452
grad_step = 000420, loss = 0.001436
grad_step = 000421, loss = 0.001428
grad_step = 000422, loss = 0.001430
grad_step = 000423, loss = 0.001437
grad_step = 000424, loss = 0.001444
grad_step = 000425, loss = 0.001447
grad_step = 000426, loss = 0.001444
grad_step = 000427, loss = 0.001437
grad_step = 000428, loss = 0.001428
grad_step = 000429, loss = 0.001421
grad_step = 000430, loss = 0.001417
grad_step = 000431, loss = 0.001416
grad_step = 000432, loss = 0.001417
grad_step = 000433, loss = 0.001419
grad_step = 000434, loss = 0.001421
grad_step = 000435, loss = 0.001423
grad_step = 000436, loss = 0.001424
grad_step = 000437, loss = 0.001425
grad_step = 000438, loss = 0.001425
grad_step = 000439, loss = 0.001426
grad_step = 000440, loss = 0.001427
grad_step = 000441, loss = 0.001430
grad_step = 000442, loss = 0.001434
grad_step = 000443, loss = 0.001442
grad_step = 000444, loss = 0.001451
grad_step = 000445, loss = 0.001468
grad_step = 000446, loss = 0.001484
grad_step = 000447, loss = 0.001512
grad_step = 000448, loss = 0.001530
grad_step = 000449, loss = 0.001559
grad_step = 000450, loss = 0.001553
grad_step = 000451, loss = 0.001541
grad_step = 000452, loss = 0.001490
grad_step = 000453, loss = 0.001439
grad_step = 000454, loss = 0.001399
grad_step = 000455, loss = 0.001389
grad_step = 000456, loss = 0.001405
grad_step = 000457, loss = 0.001428
grad_step = 000458, loss = 0.001444
grad_step = 000459, loss = 0.001439
grad_step = 000460, loss = 0.001423
grad_step = 000461, loss = 0.001401
grad_step = 000462, loss = 0.001384
grad_step = 000463, loss = 0.001377
grad_step = 000464, loss = 0.001380
grad_step = 000465, loss = 0.001389
grad_step = 000466, loss = 0.001400
grad_step = 000467, loss = 0.001411
grad_step = 000468, loss = 0.001419
grad_step = 000469, loss = 0.001427
grad_step = 000470, loss = 0.001431
grad_step = 000471, loss = 0.001436
grad_step = 000472, loss = 0.001437
grad_step = 000473, loss = 0.001440
grad_step = 000474, loss = 0.001435
grad_step = 000475, loss = 0.001430
grad_step = 000476, loss = 0.001418
grad_step = 000477, loss = 0.001404
grad_step = 000478, loss = 0.001387
grad_step = 000479, loss = 0.001373
grad_step = 000480, loss = 0.001362
grad_step = 000481, loss = 0.001356
grad_step = 000482, loss = 0.001355
grad_step = 000483, loss = 0.001357
grad_step = 000484, loss = 0.001361
grad_step = 000485, loss = 0.001366
grad_step = 000486, loss = 0.001373
grad_step = 000487, loss = 0.001380
grad_step = 000488, loss = 0.001392
grad_step = 000489, loss = 0.001408
grad_step = 000490, loss = 0.001437
grad_step = 000491, loss = 0.001472
grad_step = 000492, loss = 0.001534
grad_step = 000493, loss = 0.001583
grad_step = 000494, loss = 0.001656
grad_step = 000495, loss = 0.001639
grad_step = 000496, loss = 0.001593
grad_step = 000497, loss = 0.001459
grad_step = 000498, loss = 0.001359
grad_step = 000499, loss = 0.001345
grad_step = 000500, loss = 0.001401
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001457
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

  date_run                              2020-05-09 23:14:40.148196
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.234321
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 23:14:40.153850
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.126173
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 23:14:40.160046
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.143626
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 23:14:40.165038
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.917247
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
0   2020-05-09 23:14:13.061288  ...    mean_absolute_error
1   2020-05-09 23:14:13.065152  ...     mean_squared_error
2   2020-05-09 23:14:13.068380  ...  median_absolute_error
3   2020-05-09 23:14:13.071571  ...               r2_score
4   2020-05-09 23:14:22.096475  ...    mean_absolute_error
5   2020-05-09 23:14:22.101020  ...     mean_squared_error
6   2020-05-09 23:14:22.104870  ...  median_absolute_error
7   2020-05-09 23:14:22.108663  ...               r2_score
8   2020-05-09 23:14:40.148196  ...    mean_absolute_error
9   2020-05-09 23:14:40.153850  ...     mean_squared_error
10  2020-05-09 23:14:40.160046  ...  median_absolute_error
11  2020-05-09 23:14:40.165038  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 17%|█▋        | 1687552/9912422 [00:00<00:00, 16764460.50it/s] 44%|████▎     | 4325376/9912422 [00:00<00:00, 18621975.49it/s] 75%|███████▍  | 7421952/9912422 [00:00<00:00, 21138911.33it/s]9920512it [00:00, 18794547.40it/s]                             
0it [00:00, ?it/s]32768it [00:00, 536732.54it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 483523.42it/s]1654784it [00:00, 11646997.58it/s]                         
0it [00:00, ?it/s]8192it [00:00, 234840.43it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f545060d780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53edd50ac8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54505c4e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53edd50da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54505c4e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f545060de80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f545060d780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f540253be80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f545060d780> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5402fd0c88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f545060d780> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fbbab8741d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=691d8f4d231f7c383b097c50bea1e2440328f1174ff372d63f6847b70beb4c9c
  Stored in directory: /tmp/pip-ephem-wheel-cache-_gmb6vv6/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fbb43459048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  819200/17464789 [>.............................] - ETA: 1s
 3940352/17464789 [=====>........................] - ETA: 0s
 6930432/17464789 [==========>...................] - ETA: 0s
10067968/17464789 [================>.............] - ETA: 0s
13475840/17464789 [======================>.......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 23:16:06.194981: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 23:16:06.199363: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-09 23:16:06.199513: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56304e963cf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 23:16:06.199526: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6360 - accuracy: 0.5020 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6615 - accuracy: 0.5003
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6557 - accuracy: 0.5007
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7109 - accuracy: 0.4971
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6958 - accuracy: 0.4981
11000/25000 [============>.................] - ETA: 3s - loss: 7.7043 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 3s - loss: 7.7318 - accuracy: 0.4958
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7173 - accuracy: 0.4967
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7258 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 2s - loss: 7.7136 - accuracy: 0.4969
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7136 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7090 - accuracy: 0.4972
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7067 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7037 - accuracy: 0.4976
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7009 - accuracy: 0.4978
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7133 - accuracy: 0.4970
23000/25000 [==========================>...] - ETA: 0s - loss: 7.7060 - accuracy: 0.4974
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6852 - accuracy: 0.4988
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 23:16:19.683130
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 23:16:19.683130  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 23:16:25.745100: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 23:16:25.750537: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-09 23:16:25.751096: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559978097d70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 23:16:25.751373: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f57910e7cc0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6939 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.6738 - val_crf_viterbi_accuracy: 0.1867

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f576c02cf60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4750 - accuracy: 0.5125 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5695 - accuracy: 0.5063
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5470 - accuracy: 0.5078
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5184 - accuracy: 0.5097
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5702 - accuracy: 0.5063
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5785 - accuracy: 0.5058
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5831 - accuracy: 0.5054
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5838 - accuracy: 0.5054
11000/25000 [============>.................] - ETA: 3s - loss: 7.6067 - accuracy: 0.5039
12000/25000 [=============>................] - ETA: 3s - loss: 7.6142 - accuracy: 0.5034
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6324 - accuracy: 0.5022
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6721 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6939 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6892 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 7s 282us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f574d4d8240> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:17:22, 10.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:32:21, 14.5kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:38:05, 20.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<8:09:10, 29.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.55M/862M [00:01<5:41:33, 41.9kB/s].vector_cache/glove.6B.zip:   1%|          | 7.47M/862M [00:01<3:58:06, 59.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:45:50, 85.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.9M/862M [00:01<1:55:42, 122kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.4M/862M [00:01<1:20:39, 174kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:01<56:17, 248kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.9M/862M [00:01<39:17, 353kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.0M/862M [00:02<27:28, 503kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<19:13, 715kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:02<13:29, 1.01MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<09:29, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.2M/862M [00:02<06:42, 2.02MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<05:18, 2.55MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:37, 2.39MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<07:37, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:09, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:05<04:32, 2.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<06:53, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<06:29, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<04:53, 2.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<06:07, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<07:08, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<05:36, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:08<04:06, 3.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<08:34, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<07:23, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:10<05:30, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<06:55, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:12, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<04:40, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<06:21, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<05:48, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<04:20, 3.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<06:08, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<05:38, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<04:14, 3.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<06:02, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<05:33, 2.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<04:10, 3.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<05:59, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<05:31, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<04:11, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<04:07, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<05:26, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<04:07, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:53, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:24, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:06, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:05, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:21, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:04, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:19, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:00, 3.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:45, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:58, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:43, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:16, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:57, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:41, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:58, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:39, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:12, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:57, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:37, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:26, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:07, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:31, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:08, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<03:54, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:32, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:06, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:52, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:30, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:18, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:02, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:40, 3.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<1:27:05, 136kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<1:02:08, 191kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<43:42, 271kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<33:16, 354kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<24:29, 481kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<17:21, 677kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:54, 786kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<11:40, 1.00MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:24, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:36, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:25, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:29, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:24, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:40, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:15, 2.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:40, 2.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:09, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:53, 2.95MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:23, 2.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:59, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:46, 3.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:54, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<03:42, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:13, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:51, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<03:41, 3.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:12, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:48, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:38, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:55, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:43, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:05, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:42, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:34, 3.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:05, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:41, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:33, 3.09MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:04, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:29, 2.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:23, 3.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<02:32, 4.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:11, 3.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<9:36:19, 18.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<6:44:13, 26.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<4:42:28, 38.4kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<3:19:52, 54.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<2:22:11, 76.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<1:39:55, 108kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<1:09:55, 154kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<52:29, 205kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<37:52, 284kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<26:40, 402kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<21:07, 506kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<15:52, 673kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<11:21, 938kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:26, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:29, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:11, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:43, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:47, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:16, 2.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:26, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:53, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:38, 2.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:01, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:35, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:28, 2.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:52, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:28, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:23, 3.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:47, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:28, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:21, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:08, 3.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<9:50:32, 17.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<6:54:10, 24.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<4:49:26, 35.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<3:24:16, 49.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<2:23:56, 70.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<1:40:45, 100kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<1:12:38, 139kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<51:51, 194kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<36:27, 275kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<27:46, 360kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<20:17, 493kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<14:41, 680kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<10:20, 961kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<44:42, 222kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<33:19, 298kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<23:48, 416kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<18:10, 543kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<13:35, 725kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:43, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<06:55, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<23:18, 420kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<17:18, 566kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:17, 794kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<10:53, 893kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:36, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<06:13, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<06:38, 1.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:38, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:11, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:11, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:36, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:24, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:37, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:12, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:11, 2.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:25, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:03, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:58, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:53, 2.41MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<02:50, 3.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:29, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:30, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:05, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:01, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:27, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:21, 2.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:30, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:00, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:06<02:53, 3.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<8:48:32, 17.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<6:10:40, 24.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<4:18:58, 35.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<3:02:40, 49.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<2:09:41, 69.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<1:31:04, 99.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:10<1:03:31, 141kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<55:54, 161kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<40:01, 224kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<28:08, 318kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<21:42, 410kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<16:05, 553kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<11:27, 775kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<10:04, 877kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<07:57, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:06, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:10, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:48, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:43, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:12, 2.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:09, 2.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:14, 2.04MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:51, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:54, 2.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:03, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:43, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:48, 3.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:58, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:39, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:45, 3.06MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:28, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:41, 3.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<01:58, 4.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<34:59, 239kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<25:18, 330kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<17:52, 466kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<14:25, 575kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<10:56, 757kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<07:49, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<07:24, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:01, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:24, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:00, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:11, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:58, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:53, 2.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<05:11, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:28, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:20, 2.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:12, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:35, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:34, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:35, 3.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:18, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:05, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:44, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:42, 2.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<33:39, 234kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<25:11, 313kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<17:56, 438kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<12:35, 621kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<14:00, 558kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<10:35, 736kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:33, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:05, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:45, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:11, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:44, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:54, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:46, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:44, 2.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:25, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:34, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:23, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:07, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:39, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:43, 2.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:39, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:19, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:29, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:29, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:12, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<02:24, 3.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:08, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:22, 3.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:22, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:06, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:21, 3.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:20, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:04, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:19, 3.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:18, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:02, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:18, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:01, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:17, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:14, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:15, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:12, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:57, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:13, 3.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:10, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:38, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:53, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:06, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:52, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:10, 3.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:05, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:51, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:09, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:49, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:08, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:02, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:48, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:05, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:46, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:04, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:44, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:04, 3.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:57, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:43, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:01, 3.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:53, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:20, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:36, 2.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<01:55, 3.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:35, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:10, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:22, 2.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:05, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:40, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<01:57, 3.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:00, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:20, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:27, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<01:47, 3.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<13:00, 463kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<10:21, 581kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:32, 796kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<06:12, 961kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:57, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:35, 1.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:52, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:18, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:27, 2.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:05, 1.89MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:45, 2.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:04, 2.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:48, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:15, 1.76MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:33, 2.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:51, 3.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:55, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:20, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:27, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:00, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:15, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:33, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:40, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:26, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:50, 2.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:33, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:58, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:18, 2.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:41, 3.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:13, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:49, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:05, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:41, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:49, 2.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:31, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:51, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:15, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:24, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:13, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:41, 3.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:11, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<01:38, 3.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:18, 2.19MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:37, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:05, 2.42MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:15, 2.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:06, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:35, 3.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:13, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:04, 2.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:34, 3.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:13, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:03, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:32, 3.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:12, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:02, 2.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:31, 3.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:11, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:00, 2.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:30, 3.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:09, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:59, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:28, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:07, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:57, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:27, 3.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:05, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:55, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:27, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:03, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:53, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:26, 3.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:02, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:52, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:24, 3.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:00, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:50, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:23, 3.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:58, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<01:48, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:22, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:46, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:20, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:54, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:44, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:19, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:08, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:40, 2.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:11, 3.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<08:08, 487kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<06:05, 650kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<04:19, 910kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:55, 995kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:32, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:40, 1.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:27, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:07, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:34, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:58, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:09, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:41, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:46, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:36, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:13, 3.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:41, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:54, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:30, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:37, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:30, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:07, 3.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:36, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:50, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:25, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:02, 3.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:11, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:52, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:23, 2.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:44, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:34, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:10, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:35, 2.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:26, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:05, 2.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:30, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:43, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:21, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<00:58, 3.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<23:34, 133kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<16:47, 187kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<11:43, 265kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<08:50, 348kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<06:29, 473kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<04:34, 664kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:52, 775kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:01, 993kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:10, 1.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:11, 1.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:50, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:20, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:36, 1.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:24, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:02, 2.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:23, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:15, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:56, 2.92MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:18, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:11, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:53, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:14, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:08, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:51, 3.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:12, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:06, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:49, 3.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:04, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:48, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:02, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:47, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:06, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:01, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:46, 3.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:04, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:59, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:44, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:11, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:56, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:40, 3.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<2:06:00, 17.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<1:28:07, 24.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<1:01:08, 35.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<42:03, 50.3kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<37:55, 55.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<26:56, 78.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<18:49, 111kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<13:01, 159kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<09:46, 209kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<07:02, 290kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<04:54, 410kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<03:50, 515kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:52, 684kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<02:02, 953kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:51, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:29, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:04, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:10, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:00, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:44, 2.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:55, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:47, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:36, 2.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<00:25, 3.95MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<07:09, 238kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<05:09, 328kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<03:35, 464kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:51, 573kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<02:09, 755kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:31, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:24, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:08, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:49, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:55, 1.62MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:47, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:34, 2.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:29, 2.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:35, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:26, 3.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:36, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:33, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:24, 3.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:34, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:31, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:23, 3.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:29, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:29, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:26, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:18, 3.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:49, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:39, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:19, 2.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<02:36, 363kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:54, 492kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:19, 690kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:05, 799kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:51, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:36, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:35, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:29, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:21, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:24, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:15, 2.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:18, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:13, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:17, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:15, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:11, 3.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:15, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:13, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:09, 3.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:05, 3.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 3.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.08MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 896/400000 [00:00<00:44, 8959.79it/s]  0%|          | 1755/400000 [00:00<00:45, 8845.32it/s]  1%|          | 2636/400000 [00:00<00:44, 8834.56it/s]  1%|          | 3526/400000 [00:00<00:44, 8854.06it/s]  1%|          | 4431/400000 [00:00<00:44, 8908.90it/s]  1%|▏         | 5321/400000 [00:00<00:44, 8904.40it/s]  2%|▏         | 6204/400000 [00:00<00:44, 8879.80it/s]  2%|▏         | 7070/400000 [00:00<00:44, 8812.44it/s]  2%|▏         | 7974/400000 [00:00<00:44, 8878.21it/s]  2%|▏         | 8849/400000 [00:01<00:44, 8837.06it/s]  2%|▏         | 9744/400000 [00:01<00:43, 8869.80it/s]  3%|▎         | 10657/400000 [00:01<00:43, 8944.02it/s]  3%|▎         | 11540/400000 [00:01<00:43, 8884.48it/s]  3%|▎         | 12421/400000 [00:01<00:44, 8765.18it/s]  3%|▎         | 13314/400000 [00:01<00:43, 8812.86it/s]  4%|▎         | 14228/400000 [00:01<00:43, 8907.29it/s]  4%|▍         | 15122/400000 [00:01<00:43, 8915.48it/s]  4%|▍         | 16012/400000 [00:01<00:43, 8737.88it/s]  4%|▍         | 16896/400000 [00:01<00:43, 8767.80it/s]  4%|▍         | 17773/400000 [00:02<00:44, 8670.99it/s]  5%|▍         | 18641/400000 [00:02<00:44, 8664.01it/s]  5%|▍         | 19550/400000 [00:02<00:43, 8785.06it/s]  5%|▌         | 20445/400000 [00:02<00:42, 8831.82it/s]  5%|▌         | 21362/400000 [00:02<00:42, 8929.01it/s]  6%|▌         | 22268/400000 [00:02<00:42, 8967.26it/s]  6%|▌         | 23166/400000 [00:02<00:42, 8951.35it/s]  6%|▌         | 24062/400000 [00:02<00:42, 8943.36it/s]  6%|▌         | 24957/400000 [00:02<00:42, 8807.54it/s]  6%|▋         | 25852/400000 [00:02<00:42, 8849.65it/s]  7%|▋         | 26741/400000 [00:03<00:42, 8860.12it/s]  7%|▋         | 27638/400000 [00:03<00:41, 8892.50it/s]  7%|▋         | 28528/400000 [00:03<00:41, 8887.05it/s]  7%|▋         | 29417/400000 [00:03<00:41, 8883.70it/s]  8%|▊         | 30330/400000 [00:03<00:41, 8955.14it/s]  8%|▊         | 31233/400000 [00:03<00:41, 8976.29it/s]  8%|▊         | 32135/400000 [00:03<00:40, 8986.99it/s]  8%|▊         | 33056/400000 [00:03<00:40, 9049.31it/s]  8%|▊         | 33962/400000 [00:03<00:40, 8973.68it/s]  9%|▊         | 34860/400000 [00:03<00:40, 8915.26it/s]  9%|▉         | 35761/400000 [00:04<00:40, 8941.50it/s]  9%|▉         | 36673/400000 [00:04<00:40, 8993.04it/s]  9%|▉         | 37593/400000 [00:04<00:40, 9053.52it/s] 10%|▉         | 38499/400000 [00:04<00:39, 9048.96it/s] 10%|▉         | 39405/400000 [00:04<00:39, 9047.96it/s] 10%|█         | 40320/400000 [00:04<00:39, 9076.97it/s] 10%|█         | 41228/400000 [00:04<00:39, 9046.44it/s] 11%|█         | 42133/400000 [00:04<00:39, 9005.03it/s] 11%|█         | 43034/400000 [00:04<00:39, 8977.73it/s] 11%|█         | 43932/400000 [00:04<00:40, 8755.95it/s] 11%|█         | 44834/400000 [00:05<00:40, 8833.36it/s] 11%|█▏        | 45719/400000 [00:05<00:41, 8579.03it/s] 12%|█▏        | 46632/400000 [00:05<00:40, 8735.95it/s] 12%|█▏        | 47518/400000 [00:05<00:40, 8770.27it/s] 12%|█▏        | 48424/400000 [00:05<00:39, 8852.97it/s] 12%|█▏        | 49311/400000 [00:05<00:39, 8777.06it/s] 13%|█▎        | 50190/400000 [00:05<00:40, 8684.81it/s] 13%|█▎        | 51061/400000 [00:05<00:40, 8691.67it/s] 13%|█▎        | 51938/400000 [00:05<00:39, 8713.34it/s] 13%|█▎        | 52838/400000 [00:05<00:39, 8795.00it/s] 13%|█▎        | 53748/400000 [00:06<00:38, 8883.98it/s] 14%|█▎        | 54643/400000 [00:06<00:38, 8901.22it/s] 14%|█▍        | 55534/400000 [00:06<00:38, 8881.04it/s] 14%|█▍        | 56423/400000 [00:06<00:38, 8883.16it/s] 14%|█▍        | 57326/400000 [00:06<00:38, 8926.58it/s] 15%|█▍        | 58230/400000 [00:06<00:38, 8957.59it/s] 15%|█▍        | 59162/400000 [00:06<00:37, 9061.82it/s] 15%|█▌        | 60093/400000 [00:06<00:37, 9134.50it/s] 15%|█▌        | 61025/400000 [00:06<00:36, 9188.15it/s] 15%|█▌        | 61945/400000 [00:06<00:36, 9141.86it/s] 16%|█▌        | 62868/400000 [00:07<00:36, 9167.26it/s] 16%|█▌        | 63792/400000 [00:07<00:36, 9188.68it/s] 16%|█▌        | 64749/400000 [00:07<00:36, 9298.81it/s] 16%|█▋        | 65680/400000 [00:07<00:36, 9187.37it/s] 17%|█▋        | 66600/400000 [00:07<00:36, 9098.08it/s] 17%|█▋        | 67511/400000 [00:07<00:36, 9009.92it/s] 17%|█▋        | 68413/400000 [00:07<00:36, 8990.19it/s] 17%|█▋        | 69324/400000 [00:07<00:36, 9024.36it/s] 18%|█▊        | 70227/400000 [00:07<00:36, 9000.02it/s] 18%|█▊        | 71128/400000 [00:07<00:36, 8908.59it/s] 18%|█▊        | 72038/400000 [00:08<00:36, 8964.68it/s] 18%|█▊        | 72952/400000 [00:08<00:36, 9014.11it/s] 18%|█▊        | 73854/400000 [00:08<00:36, 8956.72it/s] 19%|█▊        | 74750/400000 [00:08<00:36, 8942.15it/s] 19%|█▉        | 75657/400000 [00:08<00:36, 8978.40it/s] 19%|█▉        | 76558/400000 [00:08<00:35, 8986.95it/s] 19%|█▉        | 77457/400000 [00:08<00:35, 8960.88it/s] 20%|█▉        | 78368/400000 [00:08<00:35, 9002.95it/s] 20%|█▉        | 79269/400000 [00:08<00:36, 8802.65it/s] 20%|██        | 80151/400000 [00:08<00:36, 8653.23it/s] 20%|██        | 81043/400000 [00:09<00:36, 8729.88it/s] 20%|██        | 81964/400000 [00:09<00:35, 8866.46it/s] 21%|██        | 82877/400000 [00:09<00:35, 8942.39it/s] 21%|██        | 83773/400000 [00:09<00:35, 8898.72it/s] 21%|██        | 84664/400000 [00:09<00:35, 8867.54it/s] 21%|██▏       | 85566/400000 [00:09<00:35, 8910.93it/s] 22%|██▏       | 86475/400000 [00:09<00:34, 8963.69it/s] 22%|██▏       | 87372/400000 [00:09<00:34, 8960.63it/s] 22%|██▏       | 88270/400000 [00:09<00:34, 8962.71it/s] 22%|██▏       | 89167/400000 [00:10<00:34, 8932.68it/s] 23%|██▎       | 90071/400000 [00:10<00:34, 8962.15it/s] 23%|██▎       | 90977/400000 [00:10<00:34, 8991.24it/s] 23%|██▎       | 91890/400000 [00:10<00:34, 9030.25it/s] 23%|██▎       | 92794/400000 [00:10<00:34, 8904.70it/s] 23%|██▎       | 93692/400000 [00:10<00:34, 8926.64it/s] 24%|██▎       | 94600/400000 [00:10<00:34, 8970.28it/s] 24%|██▍       | 95498/400000 [00:10<00:33, 8957.23it/s] 24%|██▍       | 96394/400000 [00:10<00:34, 8849.43it/s] 24%|██▍       | 97280/400000 [00:10<00:34, 8841.77it/s] 25%|██▍       | 98167/400000 [00:11<00:34, 8849.66it/s] 25%|██▍       | 99053/400000 [00:11<00:34, 8822.06it/s] 25%|██▍       | 99963/400000 [00:11<00:33, 8902.96it/s] 25%|██▌       | 100864/400000 [00:11<00:33, 8932.72it/s] 25%|██▌       | 101765/400000 [00:11<00:33, 8953.72it/s] 26%|██▌       | 102661/400000 [00:11<00:33, 8944.12it/s] 26%|██▌       | 103556/400000 [00:11<00:33, 8942.23it/s] 26%|██▌       | 104451/400000 [00:11<00:33, 8940.82it/s] 26%|██▋       | 105348/400000 [00:11<00:32, 8947.57it/s] 27%|██▋       | 106243/400000 [00:11<00:32, 8944.93it/s] 27%|██▋       | 107138/400000 [00:12<00:32, 8881.16it/s] 27%|██▋       | 108027/400000 [00:12<00:32, 8850.76it/s] 27%|██▋       | 108932/400000 [00:12<00:32, 8906.78it/s] 27%|██▋       | 109835/400000 [00:12<00:32, 8940.75it/s] 28%|██▊       | 110737/400000 [00:12<00:32, 8962.54it/s] 28%|██▊       | 111634/400000 [00:12<00:32, 8936.88it/s] 28%|██▊       | 112530/400000 [00:12<00:32, 8940.76it/s] 28%|██▊       | 113438/400000 [00:12<00:31, 8981.02it/s] 29%|██▊       | 114353/400000 [00:12<00:31, 9030.87it/s] 29%|██▉       | 115262/400000 [00:12<00:31, 9046.05it/s] 29%|██▉       | 116167/400000 [00:13<00:31, 8981.90it/s] 29%|██▉       | 117067/400000 [00:13<00:31, 8985.17it/s] 29%|██▉       | 117990/400000 [00:13<00:31, 9055.01it/s] 30%|██▉       | 118913/400000 [00:13<00:30, 9105.01it/s] 30%|██▉       | 119824/400000 [00:13<00:30, 9106.19it/s] 30%|███       | 120735/400000 [00:13<00:30, 9047.00it/s] 30%|███       | 121667/400000 [00:13<00:30, 9126.39it/s] 31%|███       | 122604/400000 [00:13<00:30, 9195.24it/s] 31%|███       | 123540/400000 [00:13<00:29, 9243.21it/s] 31%|███       | 124465/400000 [00:13<00:29, 9224.74it/s] 31%|███▏      | 125388/400000 [00:14<00:30, 9086.55it/s] 32%|███▏      | 126298/400000 [00:14<00:30, 8886.48it/s] 32%|███▏      | 127189/400000 [00:14<00:30, 8846.67it/s] 32%|███▏      | 128107/400000 [00:14<00:30, 8943.85it/s] 32%|███▏      | 129028/400000 [00:14<00:30, 9021.99it/s] 32%|███▏      | 129932/400000 [00:14<00:30, 8979.38it/s] 33%|███▎      | 130839/400000 [00:14<00:29, 9003.69it/s] 33%|███▎      | 131748/400000 [00:14<00:29, 9028.96it/s] 33%|███▎      | 132669/400000 [00:14<00:29, 9080.92it/s] 33%|███▎      | 133578/400000 [00:14<00:29, 9072.56it/s] 34%|███▎      | 134486/400000 [00:15<00:29, 8996.55it/s] 34%|███▍      | 135402/400000 [00:15<00:29, 9043.27it/s] 34%|███▍      | 136326/400000 [00:15<00:28, 9099.64it/s] 34%|███▍      | 137238/400000 [00:15<00:28, 9105.04it/s] 35%|███▍      | 138149/400000 [00:15<00:29, 8998.52it/s] 35%|███▍      | 139051/400000 [00:15<00:28, 9003.00it/s] 35%|███▍      | 139970/400000 [00:15<00:28, 9056.35it/s] 35%|███▌      | 140877/400000 [00:15<00:28, 9059.41it/s] 35%|███▌      | 141784/400000 [00:15<00:28, 9021.83it/s] 36%|███▌      | 142687/400000 [00:15<00:28, 9011.89it/s] 36%|███▌      | 143589/400000 [00:16<00:28, 8963.26it/s] 36%|███▌      | 144486/400000 [00:16<00:29, 8808.65it/s] 36%|███▋      | 145390/400000 [00:16<00:28, 8874.45it/s] 37%|███▋      | 146305/400000 [00:16<00:28, 8954.55it/s] 37%|███▋      | 147215/400000 [00:16<00:28, 8994.66it/s] 37%|███▋      | 148115/400000 [00:16<00:28, 8949.23it/s] 37%|███▋      | 149027/400000 [00:16<00:27, 8998.22it/s] 37%|███▋      | 149928/400000 [00:16<00:27, 8990.19it/s] 38%|███▊      | 150839/400000 [00:16<00:27, 9025.75it/s] 38%|███▊      | 151742/400000 [00:16<00:27, 8990.64it/s] 38%|███▊      | 152642/400000 [00:17<00:27, 8914.18it/s] 38%|███▊      | 153547/400000 [00:17<00:27, 8953.00it/s] 39%|███▊      | 154474/400000 [00:17<00:27, 9043.84it/s] 39%|███▉      | 155385/400000 [00:17<00:26, 9062.02it/s] 39%|███▉      | 156292/400000 [00:17<00:26, 9058.89it/s] 39%|███▉      | 157199/400000 [00:17<00:26, 9015.44it/s] 40%|███▉      | 158106/400000 [00:17<00:26, 9029.06it/s] 40%|███▉      | 159029/400000 [00:17<00:26, 9088.38it/s] 40%|███▉      | 159951/400000 [00:17<00:26, 9126.51it/s] 40%|████      | 160864/400000 [00:17<00:26, 9005.51it/s] 40%|████      | 161766/400000 [00:18<00:26, 8981.99it/s] 41%|████      | 162671/400000 [00:18<00:26, 9000.35it/s] 41%|████      | 163572/400000 [00:18<00:26, 8833.70it/s] 41%|████      | 164467/400000 [00:18<00:26, 8868.20it/s] 41%|████▏     | 165371/400000 [00:18<00:26, 8918.76it/s] 42%|████▏     | 166264/400000 [00:18<00:26, 8887.72it/s] 42%|████▏     | 167165/400000 [00:18<00:26, 8922.74it/s] 42%|████▏     | 168086/400000 [00:18<00:25, 9007.01it/s] 42%|████▏     | 169005/400000 [00:18<00:25, 9059.86it/s] 42%|████▏     | 169920/400000 [00:18<00:25, 9084.46it/s] 43%|████▎     | 170829/400000 [00:19<00:25, 9009.83it/s] 43%|████▎     | 171774/400000 [00:19<00:24, 9135.37it/s] 43%|████▎     | 172722/400000 [00:19<00:24, 9235.16it/s] 43%|████▎     | 173647/400000 [00:19<00:24, 9179.77it/s] 44%|████▎     | 174566/400000 [00:19<00:24, 9110.67it/s] 44%|████▍     | 175478/400000 [00:19<00:25, 8947.43it/s] 44%|████▍     | 176374/400000 [00:19<00:25, 8907.93it/s] 44%|████▍     | 177268/400000 [00:19<00:24, 8914.90it/s] 45%|████▍     | 178169/400000 [00:19<00:24, 8940.42it/s] 45%|████▍     | 179080/400000 [00:19<00:24, 8990.26it/s] 45%|████▍     | 179980/400000 [00:20<00:24, 8922.13it/s] 45%|████▌     | 180891/400000 [00:20<00:24, 8976.83it/s] 45%|████▌     | 181819/400000 [00:20<00:24, 9065.61it/s] 46%|████▌     | 182732/400000 [00:20<00:23, 9084.21it/s] 46%|████▌     | 183649/400000 [00:20<00:23, 9107.48it/s] 46%|████▌     | 184560/400000 [00:20<00:23, 8999.01it/s] 46%|████▋     | 185461/400000 [00:20<00:23, 8943.40it/s] 47%|████▋     | 186366/400000 [00:20<00:23, 8974.74it/s] 47%|████▋     | 187275/400000 [00:20<00:23, 9008.41it/s] 47%|████▋     | 188177/400000 [00:21<00:23, 8993.25it/s] 47%|████▋     | 189077/400000 [00:21<00:23, 8949.73it/s] 47%|████▋     | 189975/400000 [00:21<00:23, 8958.15it/s] 48%|████▊     | 190871/400000 [00:21<00:23, 8938.89it/s] 48%|████▊     | 191774/400000 [00:21<00:23, 8963.96it/s] 48%|████▊     | 192671/400000 [00:21<00:23, 8946.09it/s] 48%|████▊     | 193566/400000 [00:21<00:23, 8929.57it/s] 49%|████▊     | 194469/400000 [00:21<00:22, 8959.02it/s] 49%|████▉     | 195374/400000 [00:21<00:22, 8985.36it/s] 49%|████▉     | 196286/400000 [00:21<00:22, 9023.93it/s] 49%|████▉     | 197190/400000 [00:22<00:22, 9027.30it/s] 50%|████▉     | 198093/400000 [00:22<00:22, 8861.01it/s] 50%|████▉     | 198980/400000 [00:22<00:22, 8806.18it/s] 50%|████▉     | 199871/400000 [00:22<00:22, 8836.44it/s] 50%|█████     | 200765/400000 [00:22<00:22, 8865.65it/s] 50%|█████     | 201661/400000 [00:22<00:22, 8891.40it/s] 51%|█████     | 202551/400000 [00:22<00:22, 8869.36it/s] 51%|█████     | 203439/400000 [00:22<00:22, 8812.36it/s] 51%|█████     | 204338/400000 [00:22<00:22, 8862.78it/s] 51%|█████▏    | 205225/400000 [00:22<00:21, 8864.79it/s] 52%|█████▏    | 206125/400000 [00:23<00:21, 8902.50it/s] 52%|█████▏    | 207017/400000 [00:23<00:21, 8905.96it/s] 52%|█████▏    | 207914/400000 [00:23<00:21, 8923.26it/s] 52%|█████▏    | 208810/400000 [00:23<00:21, 8931.69it/s] 52%|█████▏    | 209704/400000 [00:23<00:21, 8887.42it/s] 53%|█████▎    | 210593/400000 [00:23<00:21, 8860.61it/s] 53%|█████▎    | 211480/400000 [00:23<00:21, 8829.11it/s] 53%|█████▎    | 212379/400000 [00:23<00:21, 8874.42it/s] 53%|█████▎    | 213299/400000 [00:23<00:20, 8968.99it/s] 54%|█████▎    | 214197/400000 [00:23<00:20, 8969.54it/s] 54%|█████▍    | 215109/400000 [00:24<00:20, 9013.10it/s] 54%|█████▍    | 216011/400000 [00:24<00:20, 8964.52it/s] 54%|█████▍    | 216908/400000 [00:24<00:20, 8948.29it/s] 54%|█████▍    | 217819/400000 [00:24<00:20, 8995.82it/s] 55%|█████▍    | 218719/400000 [00:24<00:20, 8995.60it/s] 55%|█████▍    | 219619/400000 [00:24<00:20, 8975.11it/s] 55%|█████▌    | 220517/400000 [00:24<00:19, 8974.24it/s] 55%|█████▌    | 221415/400000 [00:24<00:20, 8834.61it/s] 56%|█████▌    | 222307/400000 [00:24<00:20, 8859.06it/s] 56%|█████▌    | 223206/400000 [00:24<00:19, 8896.45it/s] 56%|█████▌    | 224096/400000 [00:25<00:19, 8891.10it/s] 56%|█████▌    | 224999/400000 [00:25<00:19, 8929.51it/s] 56%|█████▋    | 225893/400000 [00:25<00:19, 8919.27it/s] 57%|█████▋    | 226813/400000 [00:25<00:19, 9001.21it/s] 57%|█████▋    | 227720/400000 [00:25<00:19, 9018.70it/s] 57%|█████▋    | 228624/400000 [00:25<00:18, 9023.69it/s] 57%|█████▋    | 229527/400000 [00:25<00:18, 8998.02it/s] 58%|█████▊    | 230427/400000 [00:25<00:19, 8881.39it/s] 58%|█████▊    | 231330/400000 [00:25<00:18, 8924.77it/s] 58%|█████▊    | 232233/400000 [00:25<00:18, 8953.76it/s] 58%|█████▊    | 233139/400000 [00:26<00:18, 8982.97it/s] 59%|█████▊    | 234038/400000 [00:26<00:18, 8974.97it/s] 59%|█████▊    | 234936/400000 [00:26<00:18, 8974.81it/s] 59%|█████▉    | 235834/400000 [00:26<00:18, 8736.13it/s] 59%|█████▉    | 236714/400000 [00:26<00:18, 8754.46it/s] 59%|█████▉    | 237625/400000 [00:26<00:18, 8855.77it/s] 60%|█████▉    | 238526/400000 [00:26<00:18, 8901.12it/s] 60%|█████▉    | 239417/400000 [00:26<00:18, 8896.71it/s] 60%|██████    | 240376/400000 [00:26<00:17, 9091.91it/s] 60%|██████    | 241307/400000 [00:26<00:17, 9154.55it/s] 61%|██████    | 242224/400000 [00:27<00:17, 9127.43it/s] 61%|██████    | 243155/400000 [00:27<00:17, 9180.50it/s] 61%|██████    | 244074/400000 [00:27<00:17, 9028.92it/s] 61%|██████    | 244991/400000 [00:27<00:17, 9067.78it/s] 61%|██████▏   | 245899/400000 [00:27<00:16, 9066.41it/s] 62%|██████▏   | 246813/400000 [00:27<00:16, 9086.00it/s] 62%|██████▏   | 247722/400000 [00:27<00:16, 9026.26it/s] 62%|██████▏   | 248625/400000 [00:27<00:17, 8818.11it/s] 62%|██████▏   | 249544/400000 [00:27<00:16, 8923.82it/s] 63%|██████▎   | 250461/400000 [00:27<00:16, 8994.12it/s] 63%|██████▎   | 251362/400000 [00:28<00:16, 8992.10it/s] 63%|██████▎   | 252262/400000 [00:28<00:16, 8919.18it/s] 63%|██████▎   | 253155/400000 [00:28<00:16, 8903.13it/s] 64%|██████▎   | 254046/400000 [00:28<00:16, 8723.48it/s] 64%|██████▎   | 254949/400000 [00:28<00:16, 8812.66it/s] 64%|██████▍   | 255845/400000 [00:28<00:16, 8854.63it/s] 64%|██████▍   | 256732/400000 [00:28<00:16, 8734.16it/s] 64%|██████▍   | 257607/400000 [00:28<00:16, 8638.72it/s] 65%|██████▍   | 258507/400000 [00:28<00:16, 8741.91it/s] 65%|██████▍   | 259405/400000 [00:28<00:15, 8810.72it/s] 65%|██████▌   | 260319/400000 [00:29<00:15, 8904.59it/s] 65%|██████▌   | 261258/400000 [00:29<00:15, 9043.49it/s] 66%|██████▌   | 262164/400000 [00:29<00:15, 8996.04it/s] 66%|██████▌   | 263082/400000 [00:29<00:15, 9049.92it/s] 66%|██████▌   | 263998/400000 [00:29<00:14, 9081.78it/s] 66%|██████▌   | 264907/400000 [00:29<00:14, 9073.54it/s] 66%|██████▋   | 265815/400000 [00:29<00:14, 9049.00it/s] 67%|██████▋   | 266721/400000 [00:29<00:14, 9009.29it/s] 67%|██████▋   | 267623/400000 [00:29<00:14, 8987.21it/s] 67%|██████▋   | 268544/400000 [00:29<00:14, 9050.47it/s] 67%|██████▋   | 269455/400000 [00:30<00:14, 9065.90it/s] 68%|██████▊   | 270362/400000 [00:30<00:14, 8992.67it/s] 68%|██████▊   | 271262/400000 [00:30<00:14, 8968.58it/s] 68%|██████▊   | 272160/400000 [00:30<00:14, 8932.25it/s] 68%|██████▊   | 273068/400000 [00:30<00:14, 8975.96it/s] 68%|██████▊   | 273971/400000 [00:30<00:14, 8989.48it/s] 69%|██████▊   | 274872/400000 [00:30<00:13, 8995.59it/s] 69%|██████▉   | 275772/400000 [00:30<00:14, 8755.47it/s] 69%|██████▉   | 276659/400000 [00:30<00:14, 8788.00it/s] 69%|██████▉   | 277576/400000 [00:31<00:13, 8897.95it/s] 70%|██████▉   | 278482/400000 [00:31<00:13, 8944.97it/s] 70%|██████▉   | 279380/400000 [00:31<00:13, 8953.81it/s] 70%|███████   | 280276/400000 [00:31<00:13, 8873.64it/s] 70%|███████   | 281188/400000 [00:31<00:13, 8943.78it/s] 71%|███████   | 282093/400000 [00:31<00:13, 8973.09it/s] 71%|███████   | 282995/400000 [00:31<00:13, 8985.19it/s] 71%|███████   | 283894/400000 [00:31<00:13, 8916.05it/s] 71%|███████   | 284786/400000 [00:31<00:13, 8782.08it/s] 71%|███████▏  | 285665/400000 [00:31<00:13, 8704.27it/s] 72%|███████▏  | 286564/400000 [00:32<00:12, 8785.37it/s] 72%|███████▏  | 287466/400000 [00:32<00:12, 8854.39it/s] 72%|███████▏  | 288410/400000 [00:32<00:12, 9020.76it/s] 72%|███████▏  | 289319/400000 [00:32<00:12, 9040.97it/s] 73%|███████▎  | 290224/400000 [00:32<00:12, 9000.46it/s] 73%|███████▎  | 291125/400000 [00:32<00:12, 8985.66it/s] 73%|███████▎  | 292043/400000 [00:32<00:11, 9041.22it/s] 73%|███████▎  | 292949/400000 [00:32<00:11, 9044.96it/s] 73%|███████▎  | 293871/400000 [00:32<00:11, 9095.46it/s] 74%|███████▎  | 294781/400000 [00:32<00:11, 9017.23it/s] 74%|███████▍  | 295701/400000 [00:33<00:11, 9068.76it/s] 74%|███████▍  | 296629/400000 [00:33<00:11, 9130.55it/s] 74%|███████▍  | 297559/400000 [00:33<00:11, 9178.19it/s] 75%|███████▍  | 298478/400000 [00:33<00:11, 9027.24it/s] 75%|███████▍  | 299382/400000 [00:33<00:11, 9006.20it/s] 75%|███████▌  | 300284/400000 [00:33<00:11, 8999.49it/s] 75%|███████▌  | 301191/400000 [00:33<00:10, 9019.63it/s] 76%|███████▌  | 302094/400000 [00:33<00:10, 9021.54it/s] 76%|███████▌  | 302999/400000 [00:33<00:10, 9027.43it/s] 76%|███████▌  | 303909/400000 [00:33<00:10, 9047.74it/s] 76%|███████▌  | 304820/400000 [00:34<00:10, 9066.22it/s] 76%|███████▋  | 305727/400000 [00:34<00:10, 9062.41it/s] 77%|███████▋  | 306652/400000 [00:34<00:10, 9116.95it/s] 77%|███████▋  | 307564/400000 [00:34<00:10, 9087.95it/s] 77%|███████▋  | 308473/400000 [00:34<00:10, 9079.82it/s] 77%|███████▋  | 309385/400000 [00:34<00:09, 9090.94it/s] 78%|███████▊  | 310295/400000 [00:34<00:09, 9069.34it/s] 78%|███████▊  | 311202/400000 [00:34<00:09, 9031.12it/s] 78%|███████▊  | 312106/400000 [00:34<00:09, 9029.93it/s] 78%|███████▊  | 313017/400000 [00:34<00:09, 9052.34it/s] 78%|███████▊  | 313937/400000 [00:35<00:09, 9094.68it/s] 79%|███████▊  | 314847/400000 [00:35<00:09, 8962.97it/s] 79%|███████▉  | 315792/400000 [00:35<00:09, 9102.89it/s] 79%|███████▉  | 316704/400000 [00:35<00:09, 9000.10it/s] 79%|███████▉  | 317620/400000 [00:35<00:09, 9046.70it/s] 80%|███████▉  | 318539/400000 [00:35<00:08, 9086.43it/s] 80%|███████▉  | 319451/400000 [00:35<00:08, 9094.30it/s] 80%|████████  | 320361/400000 [00:35<00:08, 8968.77it/s] 80%|████████  | 321259/400000 [00:35<00:08, 8895.07it/s] 81%|████████  | 322151/400000 [00:35<00:08, 8900.63it/s] 81%|████████  | 323047/400000 [00:36<00:08, 8918.20it/s] 81%|████████  | 323954/400000 [00:36<00:08, 8962.57it/s] 81%|████████  | 324858/400000 [00:36<00:08, 8985.28it/s] 81%|████████▏ | 325757/400000 [00:36<00:08, 8962.30it/s] 82%|████████▏ | 326676/400000 [00:36<00:08, 9028.02it/s] 82%|████████▏ | 327592/400000 [00:36<00:07, 9067.11it/s] 82%|████████▏ | 328499/400000 [00:36<00:07, 9067.79it/s] 82%|████████▏ | 329406/400000 [00:36<00:07, 9064.94it/s] 83%|████████▎ | 330313/400000 [00:36<00:07, 9062.19it/s] 83%|████████▎ | 331228/400000 [00:36<00:07, 9087.60it/s] 83%|████████▎ | 332146/400000 [00:37<00:07, 9112.64it/s] 83%|████████▎ | 333060/400000 [00:37<00:07, 9119.10it/s] 83%|████████▎ | 333972/400000 [00:37<00:07, 9096.83it/s] 84%|████████▎ | 334882/400000 [00:37<00:07, 9090.23it/s] 84%|████████▍ | 335792/400000 [00:37<00:07, 9083.65it/s] 84%|████████▍ | 336707/400000 [00:37<00:06, 9098.15it/s] 84%|████████▍ | 337617/400000 [00:37<00:06, 9072.15it/s] 85%|████████▍ | 338534/400000 [00:37<00:06, 9100.71it/s] 85%|████████▍ | 339445/400000 [00:37<00:06, 9086.49it/s] 85%|████████▌ | 340356/400000 [00:37<00:06, 9091.11it/s] 85%|████████▌ | 341266/400000 [00:38<00:06, 9090.93it/s] 86%|████████▌ | 342194/400000 [00:38<00:06, 9146.69it/s] 86%|████████▌ | 343138/400000 [00:38<00:06, 9230.67it/s] 86%|████████▌ | 344062/400000 [00:38<00:06, 9175.69it/s] 86%|████████▌ | 344980/400000 [00:38<00:05, 9176.36it/s] 86%|████████▋ | 345898/400000 [00:38<00:05, 9123.73it/s] 87%|████████▋ | 346811/400000 [00:38<00:05, 9017.92it/s] 87%|████████▋ | 347744/400000 [00:38<00:05, 9108.89it/s] 87%|████████▋ | 348656/400000 [00:38<00:05, 9105.67it/s] 87%|████████▋ | 349585/400000 [00:38<00:05, 9158.72it/s] 88%|████████▊ | 350515/400000 [00:39<00:05, 9199.38it/s] 88%|████████▊ | 351459/400000 [00:39<00:05, 9268.41it/s] 88%|████████▊ | 352388/400000 [00:39<00:05, 9273.36it/s] 88%|████████▊ | 353316/400000 [00:39<00:05, 9108.91it/s] 89%|████████▊ | 354228/400000 [00:39<00:05, 9089.16it/s] 89%|████████▉ | 355138/400000 [00:39<00:04, 9089.27it/s] 89%|████████▉ | 356053/400000 [00:39<00:04, 9104.69it/s] 89%|████████▉ | 356964/400000 [00:39<00:04, 9084.16it/s] 89%|████████▉ | 357873/400000 [00:39<00:04, 9011.84it/s] 90%|████████▉ | 358775/400000 [00:39<00:04, 8937.93it/s] 90%|████████▉ | 359677/400000 [00:40<00:04, 8961.76it/s] 90%|█████████ | 360576/400000 [00:40<00:04, 8968.11it/s] 90%|█████████ | 361475/400000 [00:40<00:04, 8973.25it/s] 91%|█████████ | 362373/400000 [00:40<00:04, 8939.81it/s] 91%|█████████ | 363279/400000 [00:40<00:04, 8974.58it/s] 91%|█████████ | 364178/400000 [00:40<00:03, 8976.86it/s] 91%|█████████▏| 365093/400000 [00:40<00:03, 9027.27it/s] 91%|█████████▏| 365996/400000 [00:40<00:03, 8900.60it/s] 92%|█████████▏| 366900/400000 [00:40<00:03, 8939.07it/s] 92%|█████████▏| 367809/400000 [00:40<00:03, 8981.32it/s] 92%|█████████▏| 368708/400000 [00:41<00:03, 8970.79it/s] 92%|█████████▏| 369606/400000 [00:41<00:03, 8939.93it/s] 93%|█████████▎| 370501/400000 [00:41<00:03, 8880.79it/s] 93%|█████████▎| 371395/400000 [00:41<00:03, 8898.25it/s] 93%|█████████▎| 372285/400000 [00:41<00:03, 8843.17it/s] 93%|█████████▎| 373170/400000 [00:41<00:03, 8826.85it/s] 94%|█████████▎| 374065/400000 [00:41<00:02, 8863.31it/s] 94%|█████████▎| 374952/400000 [00:41<00:02, 8822.99it/s] 94%|█████████▍| 375846/400000 [00:41<00:02, 8855.94it/s] 94%|█████████▍| 376732/400000 [00:42<00:02, 8796.88it/s] 94%|█████████▍| 377613/400000 [00:42<00:02, 8800.58it/s] 95%|█████████▍| 378520/400000 [00:42<00:02, 8879.12it/s] 95%|█████████▍| 379434/400000 [00:42<00:02, 8954.97it/s] 95%|█████████▌| 380330/400000 [00:42<00:02, 8934.74it/s] 95%|█████████▌| 381224/400000 [00:42<00:02, 8849.94it/s] 96%|█████████▌| 382115/400000 [00:42<00:02, 8867.25it/s] 96%|█████████▌| 383002/400000 [00:42<00:01, 8762.45it/s] 96%|█████████▌| 383905/400000 [00:42<00:01, 8839.72it/s] 96%|█████████▌| 384792/400000 [00:42<00:01, 8820.80it/s] 96%|█████████▋| 385675/400000 [00:43<00:01, 8750.42it/s] 97%|█████████▋| 386555/400000 [00:43<00:01, 8765.22it/s] 97%|█████████▋| 387478/400000 [00:43<00:01, 8898.72it/s] 97%|█████████▋| 388392/400000 [00:43<00:01, 8969.17it/s] 97%|█████████▋| 389290/400000 [00:43<00:01, 8893.07it/s] 98%|█████████▊| 390180/400000 [00:43<00:01, 8885.40it/s] 98%|█████████▊| 391088/400000 [00:43<00:00, 8942.12it/s] 98%|█████████▊| 391994/400000 [00:43<00:00, 8975.58it/s] 98%|█████████▊| 392897/400000 [00:43<00:00, 8991.51it/s] 98%|█████████▊| 393797/400000 [00:43<00:00, 8851.72it/s] 99%|█████████▊| 394687/400000 [00:44<00:00, 8864.48it/s] 99%|█████████▉| 395601/400000 [00:44<00:00, 8945.01it/s] 99%|█████████▉| 396503/400000 [00:44<00:00, 8967.28it/s] 99%|█████████▉| 397416/400000 [00:44<00:00, 9011.74it/s]100%|█████████▉| 398352/400000 [00:44<00:00, 9112.06it/s]100%|█████████▉| 399264/400000 [00:44<00:00, 8967.86it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8966.07it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5730dd20f0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011490363846179925 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010960566359618834 	 Accuracy: 67

  model saves at 67% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15780 out of table with 15686 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15780 out of table with 15686 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

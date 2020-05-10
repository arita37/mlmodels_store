
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff9c6fc5470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 00:16:46.215505
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 00:16:46.219714
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 00:16:46.222854
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 00:16:46.225841
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff9b3550b00> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355844.6562
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 253141.5000
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 146059.8125
Epoch 4/10

1/1 [==============================] - 0s 106ms/step - loss: 73469.2578
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 37069.9961
Epoch 6/10

1/1 [==============================] - 0s 110ms/step - loss: 20556.9043
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 12655.3936
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 8527.9160
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 6193.3882
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 4783.2319

  #### Inference Need return ypred, ytrue ######################### 
[[-6.22646213e-01 -2.48291358e-01 -9.29914057e-01 -7.68348813e-01
  -9.72158909e-01 -6.09475970e-02 -1.76725554e+00  8.25786352e-01
  -8.82213563e-02  7.24209964e-01  1.08107305e+00  3.52792203e-01
  -1.23978806e+00 -1.84709322e+00  8.66238117e-01 -9.45094645e-01
  -7.54261494e-01  7.28611469e-01 -1.23800349e+00  1.77438521e+00
   4.56670344e-01  1.25411630e-01 -1.16973543e+00 -2.45197132e-01
   1.11084366e+00  3.61681104e-01  1.20083606e+00 -6.59103870e-01
   5.29365778e-01  1.51027822e+00 -1.78240001e-01  6.33598089e-01
  -9.89690423e-02 -1.47684669e+00  1.55569649e+00 -1.35226119e+00
   9.87297118e-01  8.54826570e-02 -3.92259717e-01 -2.44484067e-01
   3.57782841e-02  2.35943508e+00  2.24418581e-01  1.22715378e+00
   8.80343080e-01  1.78193641e+00 -2.52108574e-01  7.91673660e-01
  -4.65228558e-02  1.02325308e+00  1.38108349e+00  7.29919970e-01
  -1.70644808e+00  5.59635341e-01  8.31781745e-01  4.58117425e-01
  -7.77974904e-01 -1.56788373e+00 -3.88799399e-01  6.40263200e-01
   3.02701771e-01  1.09474955e+01  8.33976364e+00  9.42153740e+00
   9.65867519e+00  9.51314831e+00  1.21493053e+01  1.19073639e+01
   1.01438932e+01  9.07885933e+00  1.06385431e+01  8.18051815e+00
   8.94823837e+00  9.73920822e+00  9.64546299e+00  7.13416004e+00
   9.06131649e+00  1.18987064e+01  8.25097942e+00  8.28981686e+00
   8.79241276e+00  1.03486614e+01  9.97261333e+00  9.67301655e+00
   1.09049921e+01  8.99606609e+00  1.08971796e+01  1.03989439e+01
   1.02222233e+01  1.00512228e+01  9.20376873e+00  1.01882019e+01
   1.07073069e+01  8.61606407e+00  9.59700871e+00  8.62448025e+00
   1.20681267e+01  9.70922661e+00  8.72317886e+00  1.12703848e+01
   9.75062466e+00  1.19573612e+01  8.68967247e+00  9.23902321e+00
   1.15227652e+01  9.23537731e+00  9.08183098e+00  1.00748005e+01
   1.11052876e+01  1.11014795e+01  1.00372152e+01  1.12367496e+01
   9.87459278e+00  9.13246536e+00  9.37832737e+00  1.09306316e+01
   1.22302532e+01  9.71543312e+00  1.09547873e+01  8.05575848e+00
   2.77052224e-01 -1.54366493e+00  1.43647420e+00 -1.35268581e+00
   5.68467975e-01 -3.19733024e-02 -7.34192491e-01  8.47070694e-01
   1.81105828e+00 -9.58636642e-01  7.43209958e-01 -1.31584275e+00
  -3.99717927e-01 -2.50095725e-01 -7.34127760e-01  1.61602497e-01
  -1.69418168e+00  9.48231697e-01 -5.09387255e-02 -7.91810155e-01
  -4.21637595e-01 -8.57448041e-01  9.91222441e-01 -3.72952133e-01
  -6.85365856e-01  7.98757255e-01 -8.40216637e-01 -6.20377064e-03
   2.91542864e+00  1.52566242e+00  5.84598482e-02 -2.79780746e-01
  -2.09721863e-01  1.74779177e+00 -1.91782594e+00  6.23359859e-01
   7.48365462e-01  2.66387463e-02 -4.43078578e-01 -9.84651983e-01
  -1.21673846e+00 -7.06854224e-01 -2.11932993e+00 -2.13619947e+00
   1.47615671e-01 -9.35690284e-01 -1.08109200e+00 -1.12333906e+00
  -4.88797069e-01 -9.49221194e-01  1.80862755e-01  1.19823098e+00
  -7.07811356e-01 -5.55153191e-01  1.11071849e+00 -1.58721650e+00
   4.86499548e-01 -4.32264566e-01 -1.04206061e+00 -7.35859931e-01
   8.84678185e-01  2.11435497e-01  1.51893830e+00  6.72715068e-01
   3.19304824e-01  1.75236678e+00  5.06086409e-01  1.51018012e+00
   3.52601647e-01  2.27120590e+00  4.67544198e-01  1.14044476e+00
   2.61690855e-01  2.65768957e+00  1.95646501e+00  2.11245775e+00
   9.50288534e-01  9.57692862e-02  1.69583559e-01  1.12509406e+00
   1.37395263e-01  2.67834187e-01  2.19774723e-01  9.18363929e-02
   3.80905032e-01  6.49748981e-01  5.69719672e-01  3.86858582e-01
   8.74194086e-01  1.28983307e+00  3.23475313e+00  9.10265148e-01
   1.71441293e+00  2.92911959e+00  1.97447181e+00  4.97013211e-01
   2.83252239e-01  3.13706458e-01  2.11762214e+00  6.41395271e-01
   1.01651168e+00  6.08175159e-01  5.92800915e-01  1.38704133e+00
   2.36740398e+00  3.88272762e-01  8.10592175e-01  7.39976406e-01
   1.00793231e+00  2.09787250e+00  4.03562546e-01  1.61796880e+00
   1.21547699e-01  1.25591278e-01  1.18654251e+00  1.46659541e+00
   8.02046120e-01  2.24471474e+00  6.12787008e-01  1.10204804e+00
   3.07071447e-01  9.39965820e+00  1.05610399e+01  9.94467163e+00
   1.05520515e+01  9.24834156e+00  9.71120834e+00  1.09836607e+01
   9.34049988e+00  9.63820553e+00  8.57203293e+00  9.10231018e+00
   1.18151274e+01  9.55573273e+00  1.00964060e+01  9.58974361e+00
   1.01881657e+01  1.01534901e+01  1.15886049e+01  9.70331097e+00
   1.00806370e+01  1.07898989e+01  1.01410942e+01  1.05144968e+01
   9.84997845e+00  8.59954262e+00  8.60959530e+00  8.83111000e+00
   1.14773426e+01  1.07550402e+01  7.98233891e+00  9.25864506e+00
   1.18263655e+01  1.12865601e+01  9.46936321e+00  9.71683598e+00
   9.20930576e+00  1.11566267e+01  9.04553986e+00  9.13332844e+00
   1.03575487e+01  1.00034895e+01  9.34325314e+00  1.12851591e+01
   1.11202869e+01  9.64153576e+00  9.10665512e+00  1.00266514e+01
   8.21424389e+00  1.06902704e+01  1.04706564e+01  1.06393251e+01
   1.06459093e+01  9.65829659e+00  1.18776722e+01  9.52206039e+00
   9.51579285e+00  9.91454887e+00  8.71938801e+00  9.45831299e+00
   1.48337722e-01  2.14456260e-01  9.16024029e-01  1.97983563e-01
   3.73168349e-01  2.13749528e-01  1.53342319e+00  1.97135031e+00
   2.08233714e-01  9.27456975e-01  3.17891836e+00  1.75780463e+00
   2.10899544e+00  3.30047488e-01  2.26911163e+00  3.64142835e-01
   1.24799299e+00  9.60266113e-01  7.34951615e-01  1.01218796e+00
   4.08638597e-01  3.54473114e+00  6.99079633e-02  1.36682105e+00
   9.14266109e-01  1.07476449e+00  1.75300634e+00  2.18867779e+00
   1.06905103e+00  2.39984655e+00  3.07968736e-01  2.15718102e+00
   9.68912840e-01  1.40909576e+00  1.51125860e+00  4.85267103e-01
   2.00581264e+00  5.52235425e-01  3.61368179e-01  7.19763160e-01
   1.60758734e+00  9.06299829e-01  3.43026495e+00  1.79705167e+00
   6.99105382e-01  1.64680171e+00  3.27686429e-01  5.24439156e-01
   9.05221939e-01  1.50694847e+00  4.25355911e-01  8.95480573e-01
   8.75520706e-01  5.62503278e-01  9.17746544e-01  9.05876040e-01
   4.18511033e-02  2.50032544e-01  5.23418307e-01  8.18859041e-01
  -8.30254650e+00  4.09941006e+00 -1.44052200e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 00:16:54.782849
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6034
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 00:16:54.786479
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8600.37
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 00:16:54.789439
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6624
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 00:16:54.792441
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.224
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140710090314360
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140709400130448
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140709400130952
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140709400131456
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140709400131960
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140709400132464

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff9b1778470> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.609857
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.574348
grad_step = 000002, loss = 0.552999
grad_step = 000003, loss = 0.530313
grad_step = 000004, loss = 0.506188
grad_step = 000005, loss = 0.479928
grad_step = 000006, loss = 0.453855
grad_step = 000007, loss = 0.434081
grad_step = 000008, loss = 0.421624
grad_step = 000009, loss = 0.414426
grad_step = 000010, loss = 0.400548
grad_step = 000011, loss = 0.384626
grad_step = 000012, loss = 0.372704
grad_step = 000013, loss = 0.362777
grad_step = 000014, loss = 0.352122
grad_step = 000015, loss = 0.340247
grad_step = 000016, loss = 0.327916
grad_step = 000017, loss = 0.316021
grad_step = 000018, loss = 0.305231
grad_step = 000019, loss = 0.295712
grad_step = 000020, loss = 0.285923
grad_step = 000021, loss = 0.275006
grad_step = 000022, loss = 0.264385
grad_step = 000023, loss = 0.254575
grad_step = 000024, loss = 0.244371
grad_step = 000025, loss = 0.233811
grad_step = 000026, loss = 0.223759
grad_step = 000027, loss = 0.214683
grad_step = 000028, loss = 0.206295
grad_step = 000029, loss = 0.198022
grad_step = 000030, loss = 0.189502
grad_step = 000031, loss = 0.180941
grad_step = 000032, loss = 0.172742
grad_step = 000033, loss = 0.164883
grad_step = 000034, loss = 0.157098
grad_step = 000035, loss = 0.149517
grad_step = 000036, loss = 0.142504
grad_step = 000037, loss = 0.135922
grad_step = 000038, loss = 0.129371
grad_step = 000039, loss = 0.123009
grad_step = 000040, loss = 0.117025
grad_step = 000041, loss = 0.111204
grad_step = 000042, loss = 0.105467
grad_step = 000043, loss = 0.100117
grad_step = 000044, loss = 0.095182
grad_step = 000045, loss = 0.090344
grad_step = 000046, loss = 0.085625
grad_step = 000047, loss = 0.081223
grad_step = 000048, loss = 0.077091
grad_step = 000049, loss = 0.073077
grad_step = 000050, loss = 0.069280
grad_step = 000051, loss = 0.065734
grad_step = 000052, loss = 0.062275
grad_step = 000053, loss = 0.058987
grad_step = 000054, loss = 0.055938
grad_step = 000055, loss = 0.052993
grad_step = 000056, loss = 0.050172
grad_step = 000057, loss = 0.047533
grad_step = 000058, loss = 0.045022
grad_step = 000059, loss = 0.042610
grad_step = 000060, loss = 0.040336
grad_step = 000061, loss = 0.038157
grad_step = 000062, loss = 0.036075
grad_step = 000063, loss = 0.034118
grad_step = 000064, loss = 0.032244
grad_step = 000065, loss = 0.030455
grad_step = 000066, loss = 0.028770
grad_step = 000067, loss = 0.027161
grad_step = 000068, loss = 0.025635
grad_step = 000069, loss = 0.024188
grad_step = 000070, loss = 0.022804
grad_step = 000071, loss = 0.021499
grad_step = 000072, loss = 0.020266
grad_step = 000073, loss = 0.019102
grad_step = 000074, loss = 0.018003
grad_step = 000075, loss = 0.016958
grad_step = 000076, loss = 0.015975
grad_step = 000077, loss = 0.015051
grad_step = 000078, loss = 0.014189
grad_step = 000079, loss = 0.013383
grad_step = 000080, loss = 0.012626
grad_step = 000081, loss = 0.011922
grad_step = 000082, loss = 0.011264
grad_step = 000083, loss = 0.010655
grad_step = 000084, loss = 0.010087
grad_step = 000085, loss = 0.009557
grad_step = 000086, loss = 0.009066
grad_step = 000087, loss = 0.008604
grad_step = 000088, loss = 0.008172
grad_step = 000089, loss = 0.007766
grad_step = 000090, loss = 0.007387
grad_step = 000091, loss = 0.007030
grad_step = 000092, loss = 0.006694
grad_step = 000093, loss = 0.006379
grad_step = 000094, loss = 0.006082
grad_step = 000095, loss = 0.005801
grad_step = 000096, loss = 0.005537
grad_step = 000097, loss = 0.005290
grad_step = 000098, loss = 0.005057
grad_step = 000099, loss = 0.004837
grad_step = 000100, loss = 0.004632
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004438
grad_step = 000102, loss = 0.004258
grad_step = 000103, loss = 0.004088
grad_step = 000104, loss = 0.003930
grad_step = 000105, loss = 0.003782
grad_step = 000106, loss = 0.003645
grad_step = 000107, loss = 0.003517
grad_step = 000108, loss = 0.003398
grad_step = 000109, loss = 0.003287
grad_step = 000110, loss = 0.003186
grad_step = 000111, loss = 0.003091
grad_step = 000112, loss = 0.003003
grad_step = 000113, loss = 0.002920
grad_step = 000114, loss = 0.002842
grad_step = 000115, loss = 0.002772
grad_step = 000116, loss = 0.002707
grad_step = 000117, loss = 0.002647
grad_step = 000118, loss = 0.002594
grad_step = 000119, loss = 0.002546
grad_step = 000120, loss = 0.002501
grad_step = 000121, loss = 0.002461
grad_step = 000122, loss = 0.002420
grad_step = 000123, loss = 0.002376
grad_step = 000124, loss = 0.002338
grad_step = 000125, loss = 0.002303
grad_step = 000126, loss = 0.002274
grad_step = 000127, loss = 0.002251
grad_step = 000128, loss = 0.002231
grad_step = 000129, loss = 0.002213
grad_step = 000130, loss = 0.002198
grad_step = 000131, loss = 0.002183
grad_step = 000132, loss = 0.002158
grad_step = 000133, loss = 0.002126
grad_step = 000134, loss = 0.002104
grad_step = 000135, loss = 0.002079
grad_step = 000136, loss = 0.002063
grad_step = 000137, loss = 0.002053
grad_step = 000138, loss = 0.002045
grad_step = 000139, loss = 0.002031
grad_step = 000140, loss = 0.002015
grad_step = 000141, loss = 0.002034
grad_step = 000142, loss = 0.002147
grad_step = 000143, loss = 0.002308
grad_step = 000144, loss = 0.002205
grad_step = 000145, loss = 0.002118
grad_step = 000146, loss = 0.002046
grad_step = 000147, loss = 0.002054
grad_step = 000148, loss = 0.002158
grad_step = 000149, loss = 0.001949
grad_step = 000150, loss = 0.001965
grad_step = 000151, loss = 0.002093
grad_step = 000152, loss = 0.001985
grad_step = 000153, loss = 0.001972
grad_step = 000154, loss = 0.001999
grad_step = 000155, loss = 0.001894
grad_step = 000156, loss = 0.001865
grad_step = 000157, loss = 0.001926
grad_step = 000158, loss = 0.001928
grad_step = 000159, loss = 0.002009
grad_step = 000160, loss = 0.002323
grad_step = 000161, loss = 0.001841
grad_step = 000162, loss = 0.002125
grad_step = 000163, loss = 0.002489
grad_step = 000164, loss = 0.002236
grad_step = 000165, loss = 0.002342
grad_step = 000166, loss = 0.002166
grad_step = 000167, loss = 0.002107
grad_step = 000168, loss = 0.002126
grad_step = 000169, loss = 0.002152
grad_step = 000170, loss = 0.002095
grad_step = 000171, loss = 0.002100
grad_step = 000172, loss = 0.002045
grad_step = 000173, loss = 0.002010
grad_step = 000174, loss = 0.002040
grad_step = 000175, loss = 0.001992
grad_step = 000176, loss = 0.002014
grad_step = 000177, loss = 0.002010
grad_step = 000178, loss = 0.001937
grad_step = 000179, loss = 0.001986
grad_step = 000180, loss = 0.001929
grad_step = 000181, loss = 0.001962
grad_step = 000182, loss = 0.001933
grad_step = 000183, loss = 0.001925
grad_step = 000184, loss = 0.001917
grad_step = 000185, loss = 0.001906
grad_step = 000186, loss = 0.001896
grad_step = 000187, loss = 0.001895
grad_step = 000188, loss = 0.001881
grad_step = 000189, loss = 0.001871
grad_step = 000190, loss = 0.001860
grad_step = 000191, loss = 0.001855
grad_step = 000192, loss = 0.001849
grad_step = 000193, loss = 0.001837
grad_step = 000194, loss = 0.001830
grad_step = 000195, loss = 0.001818
grad_step = 000196, loss = 0.001813
grad_step = 000197, loss = 0.001802
grad_step = 000198, loss = 0.001795
grad_step = 000199, loss = 0.001784
grad_step = 000200, loss = 0.001782
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001770
grad_step = 000202, loss = 0.001761
grad_step = 000203, loss = 0.001755
grad_step = 000204, loss = 0.001752
grad_step = 000205, loss = 0.001750
grad_step = 000206, loss = 0.001746
grad_step = 000207, loss = 0.001750
grad_step = 000208, loss = 0.001757
grad_step = 000209, loss = 0.001807
grad_step = 000210, loss = 0.001893
grad_step = 000211, loss = 0.002100
grad_step = 000212, loss = 0.002026
grad_step = 000213, loss = 0.001798
grad_step = 000214, loss = 0.001745
grad_step = 000215, loss = 0.001880
grad_step = 000216, loss = 0.001822
grad_step = 000217, loss = 0.001743
grad_step = 000218, loss = 0.001810
grad_step = 000219, loss = 0.001779
grad_step = 000220, loss = 0.001754
grad_step = 000221, loss = 0.001779
grad_step = 000222, loss = 0.001740
grad_step = 000223, loss = 0.001731
grad_step = 000224, loss = 0.001770
grad_step = 000225, loss = 0.001719
grad_step = 000226, loss = 0.001701
grad_step = 000227, loss = 0.001746
grad_step = 000228, loss = 0.001715
grad_step = 000229, loss = 0.001687
grad_step = 000230, loss = 0.001710
grad_step = 000231, loss = 0.001708
grad_step = 000232, loss = 0.001680
grad_step = 000233, loss = 0.001688
grad_step = 000234, loss = 0.001705
grad_step = 000235, loss = 0.001679
grad_step = 000236, loss = 0.001666
grad_step = 000237, loss = 0.001683
grad_step = 000238, loss = 0.001679
grad_step = 000239, loss = 0.001659
grad_step = 000240, loss = 0.001661
grad_step = 000241, loss = 0.001674
grad_step = 000242, loss = 0.001677
grad_step = 000243, loss = 0.001689
grad_step = 000244, loss = 0.001751
grad_step = 000245, loss = 0.001892
grad_step = 000246, loss = 0.001975
grad_step = 000247, loss = 0.001943
grad_step = 000248, loss = 0.001797
grad_step = 000249, loss = 0.001676
grad_step = 000250, loss = 0.001758
grad_step = 000251, loss = 0.001837
grad_step = 000252, loss = 0.001716
grad_step = 000253, loss = 0.001663
grad_step = 000254, loss = 0.001745
grad_step = 000255, loss = 0.001786
grad_step = 000256, loss = 0.001659
grad_step = 000257, loss = 0.001656
grad_step = 000258, loss = 0.001736
grad_step = 000259, loss = 0.001724
grad_step = 000260, loss = 0.001635
grad_step = 000261, loss = 0.001658
grad_step = 000262, loss = 0.001711
grad_step = 000263, loss = 0.001664
grad_step = 000264, loss = 0.001627
grad_step = 000265, loss = 0.001652
grad_step = 000266, loss = 0.001672
grad_step = 000267, loss = 0.001633
grad_step = 000268, loss = 0.001624
grad_step = 000269, loss = 0.001639
grad_step = 000270, loss = 0.001642
grad_step = 000271, loss = 0.001623
grad_step = 000272, loss = 0.001615
grad_step = 000273, loss = 0.001621
grad_step = 000274, loss = 0.001620
grad_step = 000275, loss = 0.001617
grad_step = 000276, loss = 0.001607
grad_step = 000277, loss = 0.001603
grad_step = 000278, loss = 0.001604
grad_step = 000279, loss = 0.001608
grad_step = 000280, loss = 0.001606
grad_step = 000281, loss = 0.001595
grad_step = 000282, loss = 0.001588
grad_step = 000283, loss = 0.001589
grad_step = 000284, loss = 0.001594
grad_step = 000285, loss = 0.001595
grad_step = 000286, loss = 0.001591
grad_step = 000287, loss = 0.001590
grad_step = 000288, loss = 0.001590
grad_step = 000289, loss = 0.001590
grad_step = 000290, loss = 0.001588
grad_step = 000291, loss = 0.001589
grad_step = 000292, loss = 0.001595
grad_step = 000293, loss = 0.001610
grad_step = 000294, loss = 0.001638
grad_step = 000295, loss = 0.001685
grad_step = 000296, loss = 0.001767
grad_step = 000297, loss = 0.001853
grad_step = 000298, loss = 0.001864
grad_step = 000299, loss = 0.001754
grad_step = 000300, loss = 0.001597
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001577
grad_step = 000302, loss = 0.001668
grad_step = 000303, loss = 0.001690
grad_step = 000304, loss = 0.001598
grad_step = 000305, loss = 0.001577
grad_step = 000306, loss = 0.001626
grad_step = 000307, loss = 0.001628
grad_step = 000308, loss = 0.001580
grad_step = 000309, loss = 0.001581
grad_step = 000310, loss = 0.001597
grad_step = 000311, loss = 0.001579
grad_step = 000312, loss = 0.001565
grad_step = 000313, loss = 0.001578
grad_step = 000314, loss = 0.001578
grad_step = 000315, loss = 0.001554
grad_step = 000316, loss = 0.001546
grad_step = 000317, loss = 0.001562
grad_step = 000318, loss = 0.001568
grad_step = 000319, loss = 0.001548
grad_step = 000320, loss = 0.001533
grad_step = 000321, loss = 0.001540
grad_step = 000322, loss = 0.001549
grad_step = 000323, loss = 0.001544
grad_step = 000324, loss = 0.001530
grad_step = 000325, loss = 0.001528
grad_step = 000326, loss = 0.001536
grad_step = 000327, loss = 0.001538
grad_step = 000328, loss = 0.001531
grad_step = 000329, loss = 0.001524
grad_step = 000330, loss = 0.001525
grad_step = 000331, loss = 0.001532
grad_step = 000332, loss = 0.001538
grad_step = 000333, loss = 0.001545
grad_step = 000334, loss = 0.001562
grad_step = 000335, loss = 0.001611
grad_step = 000336, loss = 0.001706
grad_step = 000337, loss = 0.001854
grad_step = 000338, loss = 0.001962
grad_step = 000339, loss = 0.001862
grad_step = 000340, loss = 0.001622
grad_step = 000341, loss = 0.001545
grad_step = 000342, loss = 0.001636
grad_step = 000343, loss = 0.001658
grad_step = 000344, loss = 0.001578
grad_step = 000345, loss = 0.001579
grad_step = 000346, loss = 0.001614
grad_step = 000347, loss = 0.001551
grad_step = 000348, loss = 0.001527
grad_step = 000349, loss = 0.001597
grad_step = 000350, loss = 0.001574
grad_step = 000351, loss = 0.001506
grad_step = 000352, loss = 0.001515
grad_step = 000353, loss = 0.001560
grad_step = 000354, loss = 0.001536
grad_step = 000355, loss = 0.001494
grad_step = 000356, loss = 0.001510
grad_step = 000357, loss = 0.001533
grad_step = 000358, loss = 0.001506
grad_step = 000359, loss = 0.001484
grad_step = 000360, loss = 0.001503
grad_step = 000361, loss = 0.001513
grad_step = 000362, loss = 0.001495
grad_step = 000363, loss = 0.001477
grad_step = 000364, loss = 0.001486
grad_step = 000365, loss = 0.001495
grad_step = 000366, loss = 0.001487
grad_step = 000367, loss = 0.001476
grad_step = 000368, loss = 0.001477
grad_step = 000369, loss = 0.001483
grad_step = 000370, loss = 0.001478
grad_step = 000371, loss = 0.001468
grad_step = 000372, loss = 0.001462
grad_step = 000373, loss = 0.001465
grad_step = 000374, loss = 0.001468
grad_step = 000375, loss = 0.001465
grad_step = 000376, loss = 0.001459
grad_step = 000377, loss = 0.001455
grad_step = 000378, loss = 0.001457
grad_step = 000379, loss = 0.001461
grad_step = 000380, loss = 0.001466
grad_step = 000381, loss = 0.001477
grad_step = 000382, loss = 0.001509
grad_step = 000383, loss = 0.001597
grad_step = 000384, loss = 0.001778
grad_step = 000385, loss = 0.002095
grad_step = 000386, loss = 0.002300
grad_step = 000387, loss = 0.001973
grad_step = 000388, loss = 0.001528
grad_step = 000389, loss = 0.001551
grad_step = 000390, loss = 0.001758
grad_step = 000391, loss = 0.001633
grad_step = 000392, loss = 0.001560
grad_step = 000393, loss = 0.001669
grad_step = 000394, loss = 0.001517
grad_step = 000395, loss = 0.001528
grad_step = 000396, loss = 0.001634
grad_step = 000397, loss = 0.001500
grad_step = 000398, loss = 0.001509
grad_step = 000399, loss = 0.001545
grad_step = 000400, loss = 0.001470
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001545
grad_step = 000402, loss = 0.001512
grad_step = 000403, loss = 0.001440
grad_step = 000404, loss = 0.001506
grad_step = 000405, loss = 0.001499
grad_step = 000406, loss = 0.001445
grad_step = 000407, loss = 0.001479
grad_step = 000408, loss = 0.001473
grad_step = 000409, loss = 0.001428
grad_step = 000410, loss = 0.001454
grad_step = 000411, loss = 0.001470
grad_step = 000412, loss = 0.001432
grad_step = 000413, loss = 0.001422
grad_step = 000414, loss = 0.001447
grad_step = 000415, loss = 0.001436
grad_step = 000416, loss = 0.001416
grad_step = 000417, loss = 0.001427
grad_step = 000418, loss = 0.001433
grad_step = 000419, loss = 0.001413
grad_step = 000420, loss = 0.001408
grad_step = 000421, loss = 0.001420
grad_step = 000422, loss = 0.001420
grad_step = 000423, loss = 0.001405
grad_step = 000424, loss = 0.001400
grad_step = 000425, loss = 0.001407
grad_step = 000426, loss = 0.001408
grad_step = 000427, loss = 0.001400
grad_step = 000428, loss = 0.001394
grad_step = 000429, loss = 0.001396
grad_step = 000430, loss = 0.001398
grad_step = 000431, loss = 0.001395
grad_step = 000432, loss = 0.001389
grad_step = 000433, loss = 0.001386
grad_step = 000434, loss = 0.001386
grad_step = 000435, loss = 0.001387
grad_step = 000436, loss = 0.001385
grad_step = 000437, loss = 0.001381
grad_step = 000438, loss = 0.001378
grad_step = 000439, loss = 0.001376
grad_step = 000440, loss = 0.001377
grad_step = 000441, loss = 0.001376
grad_step = 000442, loss = 0.001375
grad_step = 000443, loss = 0.001372
grad_step = 000444, loss = 0.001369
grad_step = 000445, loss = 0.001367
grad_step = 000446, loss = 0.001366
grad_step = 000447, loss = 0.001366
grad_step = 000448, loss = 0.001365
grad_step = 000449, loss = 0.001363
grad_step = 000450, loss = 0.001362
grad_step = 000451, loss = 0.001359
grad_step = 000452, loss = 0.001357
grad_step = 000453, loss = 0.001355
grad_step = 000454, loss = 0.001354
grad_step = 000455, loss = 0.001353
grad_step = 000456, loss = 0.001352
grad_step = 000457, loss = 0.001351
grad_step = 000458, loss = 0.001351
grad_step = 000459, loss = 0.001352
grad_step = 000460, loss = 0.001356
grad_step = 000461, loss = 0.001365
grad_step = 000462, loss = 0.001386
grad_step = 000463, loss = 0.001436
grad_step = 000464, loss = 0.001526
grad_step = 000465, loss = 0.001688
grad_step = 000466, loss = 0.001877
grad_step = 000467, loss = 0.001827
grad_step = 000468, loss = 0.001548
grad_step = 000469, loss = 0.001376
grad_step = 000470, loss = 0.001495
grad_step = 000471, loss = 0.001561
grad_step = 000472, loss = 0.001389
grad_step = 000473, loss = 0.001387
grad_step = 000474, loss = 0.001500
grad_step = 000475, loss = 0.001424
grad_step = 000476, loss = 0.001359
grad_step = 000477, loss = 0.001455
grad_step = 000478, loss = 0.001406
grad_step = 000479, loss = 0.001346
grad_step = 000480, loss = 0.001401
grad_step = 000481, loss = 0.001391
grad_step = 000482, loss = 0.001331
grad_step = 000483, loss = 0.001364
grad_step = 000484, loss = 0.001377
grad_step = 000485, loss = 0.001337
grad_step = 000486, loss = 0.001333
grad_step = 000487, loss = 0.001357
grad_step = 000488, loss = 0.001338
grad_step = 000489, loss = 0.001315
grad_step = 000490, loss = 0.001332
grad_step = 000491, loss = 0.001335
grad_step = 000492, loss = 0.001315
grad_step = 000493, loss = 0.001313
grad_step = 000494, loss = 0.001327
grad_step = 000495, loss = 0.001319
grad_step = 000496, loss = 0.001306
grad_step = 000497, loss = 0.001309
grad_step = 000498, loss = 0.001315
grad_step = 000499, loss = 0.001308
grad_step = 000500, loss = 0.001298
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001299
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

  date_run                              2020-05-10 00:17:12.584093
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.173548
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 00:17:12.590095
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0650993
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 00:17:12.597291
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.116604
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 00:17:12.601677
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0107931
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
0   2020-05-10 00:16:46.215505  ...    mean_absolute_error
1   2020-05-10 00:16:46.219714  ...     mean_squared_error
2   2020-05-10 00:16:46.222854  ...  median_absolute_error
3   2020-05-10 00:16:46.225841  ...               r2_score
4   2020-05-10 00:16:54.782849  ...    mean_absolute_error
5   2020-05-10 00:16:54.786479  ...     mean_squared_error
6   2020-05-10 00:16:54.789439  ...  median_absolute_error
7   2020-05-10 00:16:54.792441  ...               r2_score
8   2020-05-10 00:17:12.584093  ...    mean_absolute_error
9   2020-05-10 00:17:12.590095  ...     mean_squared_error
10  2020-05-10 00:17:12.597291  ...  median_absolute_error
11  2020-05-10 00:17:12.601677  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3178496/9912422 [00:00<00:00, 31718366.84it/s]9920512it [00:00, 35109563.84it/s]                             
0it [00:00, ?it/s]32768it [00:00, 682583.91it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466345.82it/s]1654784it [00:00, 11837132.53it/s]                         
0it [00:00, ?it/s]8192it [00:00, 228636.61it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04fc4b7780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0499bf9c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04fc46ee48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0499bf9da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04fc46ee48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04fc4b7e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04fc4b7780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04aee68cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0499bf9da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04aee68cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f04fc4b7f98> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0293d851d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2e02413be81d28c37802bae9ee326ed04e15ecb03c210aa6d6540721de9955e8
  Stored in directory: /tmp/pip-ephem-wheel-cache-gy003btj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f022b9690f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1761280/17464789 [==>...........................] - ETA: 0s
 6471680/17464789 [==========>...................] - ETA: 0s
12468224/17464789 [====================>.........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 00:18:37.581639: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 00:18:37.585919: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-10 00:18:37.586091: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556038b8a770 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 00:18:37.586104: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6027 - accuracy: 0.5042
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6119 - accuracy: 0.5036
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6015 - accuracy: 0.5042
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6104 - accuracy: 0.5037
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6176 - accuracy: 0.5032
11000/25000 [============>.................] - ETA: 3s - loss: 7.6150 - accuracy: 0.5034
12000/25000 [=============>................] - ETA: 3s - loss: 7.6334 - accuracy: 0.5022
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6217 - accuracy: 0.5029
15000/25000 [=================>............] - ETA: 2s - loss: 7.6032 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5909 - accuracy: 0.5049
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5629 - accuracy: 0.5068
18000/25000 [====================>.........] - ETA: 1s - loss: 7.5712 - accuracy: 0.5062
19000/25000 [=====================>........] - ETA: 1s - loss: 7.5948 - accuracy: 0.5047
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6260 - accuracy: 0.5027
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6396 - accuracy: 0.5018
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6304 - accuracy: 0.5024
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6393 - accuracy: 0.5018
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 7s 262us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 00:18:50.845853
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 00:18:50.845853  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 00:18:56.820356: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 00:18:56.826414: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-10 00:18:56.826614: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556fe20d3bd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 00:18:56.826628: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc2c9795b70> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3117 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.2673 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc2d0ef30b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4596 - accuracy: 0.5135 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5286 - accuracy: 0.5090
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4673 - accuracy: 0.5130
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5823 - accuracy: 0.5055
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5746 - accuracy: 0.5060
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6087 - accuracy: 0.5038
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6176 - accuracy: 0.5032
11000/25000 [============>.................] - ETA: 3s - loss: 7.6276 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6403 - accuracy: 0.5017
15000/25000 [=================>............] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6351 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6452 - accuracy: 0.5014
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6557 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6555 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc2ae8482b0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:33:35, 11.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:19:29, 15.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:46:52, 22.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:33:18, 31.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:16:30, 45.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.56M/862M [00:01<3:40:07, 64.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.1M/862M [00:01<2:33:25, 92.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.4M/862M [00:01<1:46:41, 131kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:01<1:14:17, 188kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<51:53, 267kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<36:10, 381kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.6M/862M [00:02<25:14, 543kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<17:43, 770kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.6M/862M [00:02<12:24, 1.09MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<09:31, 1.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<08:33, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<08:15, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<06:16, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:05<04:30, 2.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<1:22:15, 162kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<59:14, 225kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<41:50, 319kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<31:50, 417kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<23:47, 558kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<17:00, 780kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<14:42, 898kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<11:39, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<08:25, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<09:01, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<09:00, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<06:54, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:13<04:59, 2.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<1:51:09, 118kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<1:19:08, 165kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<55:37, 235kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<41:52, 311kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<30:45, 423kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<21:47, 596kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<18:04, 716kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<14:08, 915kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:19<10:14, 1.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<09:56, 1.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<09:50, 1.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<07:36, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:21<05:27, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<17:30, 731kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<13:41, 935kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<09:52, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<09:42, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<08:14, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:24<06:07, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:01, 1.80MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:45, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:07, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<04:27, 2.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<18:21, 686kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<14:16, 882kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:15, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:53, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<09:43, 1.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:31, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:25, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<18:51, 661kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<14:35, 854kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:29, 1.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<10:00, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:23, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<06:09, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:01, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:17, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<04:44, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:56, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:54, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:24, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:55, 3.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:10, 1.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:46, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:43, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:39, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:00, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:29, 2.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:45, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:44, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:23, 2.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:55, 3.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<17:12, 695kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:22, 894kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<09:40, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:20, 1.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:12, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:06, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<05:07, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<17:37, 671kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<13:41, 864kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<09:53, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:26, 1.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:13, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:09, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<05:07, 2.28MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<16:03, 729kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<12:32, 932kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<09:05, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:51, 1.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:55, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:53, 1.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<17:27, 662kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<13:30, 855kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<09:42, 1.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:16, 1.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:03, 1.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:53, 1.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<04:56, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<10:44, 1.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<08:46, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<06:26, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:59, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:24, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:49, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<04:11, 2.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<15:11, 742kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<11:51, 950kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<08:33, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:26, 1.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:25, 1.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:38, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:21, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:23, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:46, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:45, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:56, 2.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:09, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:03, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:51, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<03:31, 3.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<15:16, 716kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<11:54, 918kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<08:37, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:23, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<07:05, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:15, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:00, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:43, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:13, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<03:47, 2.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:52, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:42, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:56, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:46, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:25, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:59, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<03:39, 2.90MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:09, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:17, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:59, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<02:56, 3.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<10:13, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<08:19, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:04, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<06:31, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:53, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:24, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<03:54, 2.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<15:17, 679kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<11:51, 875kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<08:31, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:11, 1.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:03, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:06, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:28, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:55, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:14, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:55, 2.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:02, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:40, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:30, 2.89MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:38, 2.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:31, 1.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:20, 2.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:12, 3.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:13, 1.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:47, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:37, 2.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:41, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:24, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:18, 3.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:28, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:10, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:07, 3.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:27, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:17, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<03:06, 3.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<12:36, 773kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<09:54, 983kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<07:09, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<07:04, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<07:07, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:31, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:58, 2.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<13:01, 738kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<10:10, 944kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:20, 1.30MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:11, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:10, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:28, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:56, 2.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:46, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:48, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:16, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:01, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:31, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:17, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:08, 2.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:55, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:11, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:50, 2.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<02:48, 3.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<28:29, 325kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<21:59, 421kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<15:51, 583kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<11:09, 825kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<13:32, 679kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:27, 879kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:30, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<07:17, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<07:03, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:20, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:52, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:46, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:01, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:43, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:36, 1.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:10, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:05, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:58, 3.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<28:49, 310kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<21:09, 421kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<15:01, 592kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<12:23, 714kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<10:35, 836kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<07:47, 1.13MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:39, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:38, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:10, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:48, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:18, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:08, 2.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:59, 2.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<07:07, 1.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:57, 1.45MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:24, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:55, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:22, 1.60MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:13, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:03, 2.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<11:13, 758kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<08:48, 965kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:21, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:16, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:17, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:39, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:11, 1.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:07, 2.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:59, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:38, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:42, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:41, 3.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<09:33, 860kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<07:37, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:32, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:39, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:52, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:37, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:17, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:43, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:41, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:40, 3.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:11, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:11, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:48, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:26, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:53, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:51, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:47, 2.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<10:22, 760kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<08:05, 973kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<05:51, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:50, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:45, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:10, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<25:44, 301kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<18:49, 411kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<13:20, 578kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<11:00, 697kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<08:33, 896kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<06:11, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:57, 1.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:52, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:31, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<03:14, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<10:18, 731kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<08:03, 935kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<05:47, 1.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:40, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:38, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:21, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<03:08, 2.36MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<11:07, 665kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<08:35, 860kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<06:11, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:57, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:44, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:21, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<03:07, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<05:50, 1.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:54, 1.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:37, 1.99MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:05, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:39, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:43, 2.62MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:27, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:56, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:04, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:14, 3.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:20, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:49, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:52, 2.45MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:30, 1.99MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:58, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:08, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<02:16, 3.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<23:27, 295kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<17:03, 405kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<12:05, 569kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<09:53, 691kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<08:27, 809kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<06:17, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<04:28, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<11:19, 598kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<08:40, 780kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<06:14, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:48, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:33, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:11, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<03:01, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:40, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:00, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:58, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:30, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:44, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:55, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:06, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:25, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:31, 1.44MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<03:19, 1.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:43, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:02, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:07, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:16, 2.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:55, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:27, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:34, 2.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:08, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:48, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:03, 3.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:37, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:11, 1.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:22, 2.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:29, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:47, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:00, 3.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<07:53, 770kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:11, 980kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:28, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:25, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:46, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:14, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:36, 1.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:50, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:02, 2.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:42, 761kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<06:02, 970kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:21, 1.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:17, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:18, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:20, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:23, 2.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<08:36, 666kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:34, 871kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:46, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<04:31, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:46, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:46, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:12, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:30, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:45, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:49<01:59, 2.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<08:04, 684kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<06:16, 880kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<04:31, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:19, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:38, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:02, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:01, 2.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:34, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:21, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:46, 2.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:26, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:16, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:42, 3.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:21, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:12, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:36, 3.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:34, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:04, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:16, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:40, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:26, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:50, 2.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:21, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:45, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:12, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:35, 3.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<06:20, 773kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:58, 982kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:36, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:33, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:33, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:43, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:57, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:21, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:52, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:08, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:30, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:45, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:10, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:34, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<14:57, 309kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<10:57, 422kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<07:43, 594kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<06:23, 712kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<04:56, 919kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:33, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:30, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:27, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:38, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:52, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<05:15, 839kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<04:10, 1.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:00, 1.46MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:01, 1.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:05, 1.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:23, 1.80MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:42, 2.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<05:46, 741kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:30, 947kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:14, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:09, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:39, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:57, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:18, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:32, 2.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:59, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:18, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:47, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:18, 3.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:20, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:05, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:33, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:55, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:47, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:21, 2.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:46, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:06, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:41, 2.28MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:12, 3.13MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<04:53, 775kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:50, 985kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<02:46, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:44, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:18, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:42, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:01, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:49, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:21, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:43, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:00, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:35, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<01:08, 3.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<04:05, 860kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:13, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<02:20, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:23, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:28, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:54, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:21, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:45, 709kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:42, 909kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:39, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:33, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:30, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:54, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<01:20, 2.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<05:22, 603kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<04:06, 787kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<02:55, 1.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:43, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:37, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:00, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:17, 721kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:22, 917kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:24, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:42, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<07:55, 382kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<06:11, 488kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:28, 674kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<03:06, 951kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<09:15, 319kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<06:48, 434kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<04:48, 609kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:56, 732kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:24, 847kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:32, 1.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:46, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:15, 663kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:17, 856kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:21, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:13, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:48, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:20, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<00:57, 2.80MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:59, 897kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:41, 996kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:01, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:25, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:47, 688kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:56, 884kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<02:06, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:00, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:58, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:30, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:04, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:14, 763kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:32, 970kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:49, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:47, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:45, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:21, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<00:57, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<05:37, 415kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<04:11, 555kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:58, 776kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:32, 893kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:16, 993kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:42, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:12, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:11, 688kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:28, 886kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:45, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:40, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:38, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:15, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:53, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:49, 728kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:12, 931kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:34, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:30, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:29, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:07, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:47, 2.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<03:54, 492kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:55, 653kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<02:04, 911kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:50, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:40, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:14, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:53, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:15, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:04, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:46, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:54, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:01, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:48, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:34, 2.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:23, 689kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:50, 886kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:18, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:14, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:12, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:55, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:44<00:39, 2.31MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:16, 661kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:43, 868kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:13, 1.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:51, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:50, 780kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:36, 893kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:11, 1.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:49, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<02:06, 646kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:37, 837kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:08, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:03, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:01, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:45, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:31, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:26, 846kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:08, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:48, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:48, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:49, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:37, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:26, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:45, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:36, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:19, 2.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:22, 689kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:03, 885kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:45, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:41, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:34, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:24, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:29, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:22, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:15, 2.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:29, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:18, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:23, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:52, 690kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:39, 892kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:27, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:25, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:24, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:21, 1.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:17, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:13, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:07, 2.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:25, 762kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:19, 970kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:13, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:11, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:11, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:04, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:13, 815kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:10, 1.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.67MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.11MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 813/400000 [00:00<00:49, 8126.75it/s]  0%|          | 1728/400000 [00:00<00:47, 8407.95it/s]  1%|          | 2650/400000 [00:00<00:46, 8634.34it/s]  1%|          | 3548/400000 [00:00<00:45, 8734.71it/s]  1%|          | 4504/400000 [00:00<00:44, 8965.41it/s]  1%|▏         | 5439/400000 [00:00<00:43, 9075.11it/s]  2%|▏         | 6370/400000 [00:00<00:43, 9143.97it/s]  2%|▏         | 7303/400000 [00:00<00:42, 9196.85it/s]  2%|▏         | 8247/400000 [00:00<00:42, 9268.23it/s]  2%|▏         | 9143/400000 [00:01<00:42, 9120.04it/s]  3%|▎         | 10157/400000 [00:01<00:41, 9401.50it/s]  3%|▎         | 11090/400000 [00:01<00:41, 9377.54it/s]  3%|▎         | 12023/400000 [00:01<00:41, 9363.13it/s]  3%|▎         | 12953/400000 [00:01<00:41, 9326.90it/s]  3%|▎         | 13882/400000 [00:01<00:41, 9264.38it/s]  4%|▎         | 14823/400000 [00:01<00:41, 9307.33it/s]  4%|▍         | 15799/400000 [00:01<00:40, 9436.47it/s]  4%|▍         | 16742/400000 [00:01<00:40, 9358.82it/s]  4%|▍         | 17678/400000 [00:01<00:41, 9198.22it/s]  5%|▍         | 18599/400000 [00:02<00:42, 8962.40it/s]  5%|▍         | 19561/400000 [00:02<00:41, 9149.82it/s]  5%|▌         | 20519/400000 [00:02<00:40, 9271.93it/s]  5%|▌         | 21466/400000 [00:02<00:40, 9327.93it/s]  6%|▌         | 22401/400000 [00:02<00:41, 9009.41it/s]  6%|▌         | 23306/400000 [00:02<00:42, 8858.07it/s]  6%|▌         | 24304/400000 [00:02<00:40, 9165.55it/s]  6%|▋         | 25282/400000 [00:02<00:40, 9339.73it/s]  7%|▋         | 26278/400000 [00:02<00:39, 9516.38it/s]  7%|▋         | 27285/400000 [00:02<00:38, 9673.42it/s]  7%|▋         | 28256/400000 [00:03<00:38, 9604.39it/s]  7%|▋         | 29231/400000 [00:03<00:38, 9640.93it/s]  8%|▊         | 30241/400000 [00:03<00:37, 9772.79it/s]  8%|▊         | 31230/400000 [00:03<00:37, 9805.44it/s]  8%|▊         | 32220/400000 [00:03<00:37, 9831.36it/s]  8%|▊         | 33204/400000 [00:03<00:38, 9444.86it/s]  9%|▊         | 34153/400000 [00:03<00:38, 9429.93it/s]  9%|▉         | 35145/400000 [00:03<00:38, 9570.06it/s]  9%|▉         | 36105/400000 [00:03<00:38, 9537.80it/s]  9%|▉         | 37139/400000 [00:03<00:37, 9762.11it/s] 10%|▉         | 38118/400000 [00:04<00:37, 9723.66it/s] 10%|▉         | 39093/400000 [00:04<00:37, 9719.38it/s] 10%|█         | 40108/400000 [00:04<00:36, 9842.59it/s] 10%|█         | 41094/400000 [00:04<00:36, 9734.24it/s] 11%|█         | 42086/400000 [00:04<00:36, 9786.92it/s] 11%|█         | 43066/400000 [00:04<00:37, 9564.42it/s] 11%|█         | 44025/400000 [00:04<00:37, 9531.46it/s] 11%|█         | 44984/400000 [00:04<00:37, 9547.70it/s] 12%|█▏        | 46002/400000 [00:04<00:36, 9726.97it/s] 12%|█▏        | 46977/400000 [00:04<00:36, 9604.51it/s] 12%|█▏        | 47939/400000 [00:05<00:37, 9512.95it/s] 12%|█▏        | 48898/400000 [00:05<00:36, 9535.55it/s] 12%|█▏        | 49853/400000 [00:05<00:36, 9499.77it/s] 13%|█▎        | 50856/400000 [00:05<00:36, 9651.40it/s] 13%|█▎        | 51823/400000 [00:05<00:36, 9510.10it/s] 13%|█▎        | 52776/400000 [00:05<00:36, 9508.43it/s] 13%|█▎        | 53796/400000 [00:05<00:35, 9705.14it/s] 14%|█▎        | 54805/400000 [00:05<00:35, 9815.66it/s] 14%|█▍        | 55862/400000 [00:05<00:34, 10029.76it/s] 14%|█▍        | 56868/400000 [00:05<00:34, 9962.78it/s]  14%|█▍        | 57866/400000 [00:06<00:34, 9868.04it/s] 15%|█▍        | 58855/400000 [00:06<00:34, 9853.99it/s] 15%|█▍        | 59842/400000 [00:06<00:35, 9692.07it/s] 15%|█▌        | 60813/400000 [00:06<00:35, 9488.50it/s] 15%|█▌        | 61789/400000 [00:06<00:35, 9566.78it/s] 16%|█▌        | 62748/400000 [00:06<00:35, 9542.61it/s] 16%|█▌        | 63704/400000 [00:06<00:35, 9546.41it/s] 16%|█▌        | 64660/400000 [00:06<00:35, 9485.82it/s] 16%|█▋        | 65660/400000 [00:06<00:34, 9634.43it/s] 17%|█▋        | 66647/400000 [00:07<00:34, 9702.07it/s] 17%|█▋        | 67619/400000 [00:07<00:35, 9481.80it/s] 17%|█▋        | 68569/400000 [00:07<00:35, 9290.78it/s] 17%|█▋        | 69524/400000 [00:07<00:35, 9364.29it/s] 18%|█▊        | 70462/400000 [00:07<00:35, 9356.04it/s] 18%|█▊        | 71399/400000 [00:07<00:35, 9345.99it/s] 18%|█▊        | 72335/400000 [00:07<00:35, 9103.34it/s] 18%|█▊        | 73261/400000 [00:07<00:35, 9147.76it/s] 19%|█▊        | 74251/400000 [00:07<00:34, 9360.52it/s] 19%|█▉        | 75190/400000 [00:07<00:35, 9219.87it/s] 19%|█▉        | 76189/400000 [00:08<00:34, 9437.25it/s] 19%|█▉        | 77136/400000 [00:08<00:34, 9422.07it/s] 20%|█▉        | 78107/400000 [00:08<00:33, 9506.37it/s] 20%|█▉        | 79104/400000 [00:08<00:33, 9640.17it/s] 20%|██        | 80092/400000 [00:08<00:32, 9709.34it/s] 20%|██        | 81108/400000 [00:08<00:32, 9839.74it/s] 21%|██        | 82094/400000 [00:08<00:33, 9583.69it/s] 21%|██        | 83055/400000 [00:08<00:33, 9577.52it/s] 21%|██        | 84121/400000 [00:08<00:31, 9875.89it/s] 21%|██▏       | 85113/400000 [00:08<00:31, 9865.46it/s] 22%|██▏       | 86102/400000 [00:09<00:33, 9483.78it/s] 22%|██▏       | 87088/400000 [00:09<00:32, 9593.34it/s] 22%|██▏       | 88107/400000 [00:09<00:31, 9763.67it/s] 22%|██▏       | 89096/400000 [00:09<00:31, 9797.77it/s] 23%|██▎       | 90079/400000 [00:09<00:31, 9716.56it/s] 23%|██▎       | 91055/400000 [00:09<00:31, 9728.88it/s] 23%|██▎       | 92042/400000 [00:09<00:31, 9768.93it/s] 23%|██▎       | 93049/400000 [00:09<00:31, 9854.81it/s] 24%|██▎       | 94047/400000 [00:09<00:30, 9891.16it/s] 24%|██▍       | 95037/400000 [00:09<00:31, 9621.05it/s] 24%|██▍       | 96002/400000 [00:10<00:32, 9424.17it/s] 24%|██▍       | 96993/400000 [00:10<00:31, 9563.97it/s] 25%|██▍       | 98028/400000 [00:10<00:30, 9784.75it/s] 25%|██▍       | 99010/400000 [00:10<00:30, 9749.57it/s] 25%|██▍       | 99987/400000 [00:10<00:31, 9579.12it/s] 25%|██▌       | 100947/400000 [00:10<00:31, 9374.70it/s] 25%|██▌       | 101967/400000 [00:10<00:31, 9607.31it/s] 26%|██▌       | 102931/400000 [00:10<00:31, 9488.37it/s] 26%|██▌       | 103944/400000 [00:10<00:30, 9671.35it/s] 26%|██▌       | 104914/400000 [00:11<00:31, 9485.51it/s] 26%|██▋       | 105866/400000 [00:11<00:31, 9434.68it/s] 27%|██▋       | 106888/400000 [00:11<00:30, 9657.22it/s] 27%|██▋       | 107857/400000 [00:11<00:30, 9664.92it/s] 27%|██▋       | 108826/400000 [00:11<00:30, 9603.28it/s] 27%|██▋       | 109788/400000 [00:11<00:30, 9397.57it/s] 28%|██▊       | 110730/400000 [00:11<00:30, 9395.47it/s] 28%|██▊       | 111671/400000 [00:11<00:31, 9258.85it/s] 28%|██▊       | 112611/400000 [00:11<00:30, 9297.80it/s] 28%|██▊       | 113554/400000 [00:11<00:30, 9336.81it/s] 29%|██▊       | 114489/400000 [00:12<00:30, 9309.58it/s] 29%|██▉       | 115459/400000 [00:12<00:30, 9422.75it/s] 29%|██▉       | 116441/400000 [00:12<00:29, 9537.69it/s] 29%|██▉       | 117404/400000 [00:12<00:29, 9563.71it/s] 30%|██▉       | 118370/400000 [00:12<00:29, 9590.66it/s] 30%|██▉       | 119330/400000 [00:12<00:29, 9547.30it/s] 30%|███       | 120286/400000 [00:12<00:30, 9256.54it/s] 30%|███       | 121214/400000 [00:12<00:30, 9121.49it/s] 31%|███       | 122233/400000 [00:12<00:29, 9417.08it/s] 31%|███       | 123200/400000 [00:12<00:29, 9490.23it/s] 31%|███       | 124192/400000 [00:13<00:28, 9614.16it/s] 31%|███▏      | 125156/400000 [00:13<00:28, 9543.15it/s] 32%|███▏      | 126113/400000 [00:13<00:29, 9423.57it/s] 32%|███▏      | 127057/400000 [00:13<00:28, 9419.60it/s] 32%|███▏      | 128034/400000 [00:13<00:28, 9519.97it/s] 32%|███▏      | 129020/400000 [00:13<00:28, 9618.48it/s] 32%|███▏      | 129983/400000 [00:13<00:28, 9543.85it/s] 33%|███▎      | 130979/400000 [00:13<00:27, 9663.77it/s] 33%|███▎      | 131978/400000 [00:13<00:27, 9757.48it/s] 33%|███▎      | 132955/400000 [00:13<00:27, 9669.72it/s] 33%|███▎      | 133923/400000 [00:14<00:27, 9651.57it/s] 34%|███▎      | 134889/400000 [00:14<00:27, 9631.98it/s] 34%|███▍      | 135884/400000 [00:14<00:27, 9721.22it/s] 34%|███▍      | 136859/400000 [00:14<00:27, 9727.07it/s] 34%|███▍      | 137877/400000 [00:14<00:26, 9858.17it/s] 35%|███▍      | 138893/400000 [00:14<00:26, 9946.64it/s] 35%|███▍      | 139896/400000 [00:14<00:26, 9969.16it/s] 35%|███▌      | 140894/400000 [00:14<00:27, 9580.44it/s] 35%|███▌      | 141929/400000 [00:14<00:26, 9796.30it/s] 36%|███▌      | 142944/400000 [00:14<00:25, 9897.82it/s] 36%|███▌      | 143937/400000 [00:15<00:25, 9879.75it/s] 36%|███▌      | 144927/400000 [00:15<00:25, 9871.14it/s] 36%|███▋      | 145916/400000 [00:15<00:25, 9875.33it/s] 37%|███▋      | 146959/400000 [00:15<00:25, 10034.26it/s] 37%|███▋      | 147964/400000 [00:15<00:25, 9893.25it/s]  37%|███▋      | 148955/400000 [00:15<00:25, 9858.19it/s] 37%|███▋      | 149942/400000 [00:15<00:25, 9779.69it/s] 38%|███▊      | 150921/400000 [00:15<00:25, 9688.86it/s] 38%|███▊      | 151891/400000 [00:15<00:25, 9648.92it/s] 38%|███▊      | 152879/400000 [00:15<00:25, 9715.78it/s] 38%|███▊      | 153879/400000 [00:16<00:25, 9798.46it/s] 39%|███▊      | 154868/400000 [00:16<00:24, 9823.73it/s] 39%|███▉      | 155851/400000 [00:16<00:25, 9682.99it/s] 39%|███▉      | 156839/400000 [00:16<00:24, 9738.61it/s] 39%|███▉      | 157814/400000 [00:16<00:25, 9610.94it/s] 40%|███▉      | 158776/400000 [00:16<00:25, 9600.69it/s] 40%|███▉      | 159780/400000 [00:16<00:24, 9725.46it/s] 40%|████      | 160754/400000 [00:16<00:24, 9673.18it/s] 40%|████      | 161747/400000 [00:16<00:24, 9747.55it/s] 41%|████      | 162723/400000 [00:17<00:24, 9605.27it/s] 41%|████      | 163685/400000 [00:17<00:24, 9579.22it/s] 41%|████      | 164688/400000 [00:17<00:24, 9709.01it/s] 41%|████▏     | 165660/400000 [00:17<00:24, 9605.50it/s] 42%|████▏     | 166622/400000 [00:17<00:24, 9384.82it/s] 42%|████▏     | 167583/400000 [00:17<00:24, 9450.06it/s] 42%|████▏     | 168572/400000 [00:17<00:24, 9575.12it/s] 42%|████▏     | 169531/400000 [00:17<00:24, 9578.75it/s] 43%|████▎     | 170490/400000 [00:17<00:24, 9486.68it/s] 43%|████▎     | 171441/400000 [00:17<00:24, 9493.27it/s] 43%|████▎     | 172500/400000 [00:18<00:23, 9796.83it/s] 43%|████▎     | 173525/400000 [00:18<00:22, 9927.29it/s] 44%|████▎     | 174550/400000 [00:18<00:22, 10019.12it/s] 44%|████▍     | 175554/400000 [00:18<00:22, 9880.25it/s]  44%|████▍     | 176561/400000 [00:18<00:22, 9933.59it/s] 44%|████▍     | 177556/400000 [00:18<00:22, 9774.32it/s] 45%|████▍     | 178535/400000 [00:18<00:22, 9747.25it/s] 45%|████▍     | 179580/400000 [00:18<00:22, 9947.25it/s] 45%|████▌     | 180587/400000 [00:18<00:21, 9981.64it/s] 45%|████▌     | 181610/400000 [00:18<00:21, 10053.33it/s] 46%|████▌     | 182617/400000 [00:19<00:22, 9835.47it/s]  46%|████▌     | 183603/400000 [00:19<00:22, 9746.68it/s] 46%|████▌     | 184615/400000 [00:19<00:21, 9854.05it/s] 46%|████▋     | 185602/400000 [00:19<00:22, 9652.16it/s] 47%|████▋     | 186598/400000 [00:19<00:21, 9740.59it/s] 47%|████▋     | 187614/400000 [00:19<00:21, 9862.08it/s] 47%|████▋     | 188623/400000 [00:19<00:21, 9926.54it/s] 47%|████▋     | 189617/400000 [00:19<00:21, 9709.93it/s] 48%|████▊     | 190590/400000 [00:19<00:21, 9600.68it/s] 48%|████▊     | 191552/400000 [00:19<00:22, 9399.47it/s] 48%|████▊     | 192494/400000 [00:20<00:22, 9320.98it/s] 48%|████▊     | 193482/400000 [00:20<00:21, 9481.52it/s] 49%|████▊     | 194494/400000 [00:20<00:21, 9661.90it/s] 49%|████▉     | 195495/400000 [00:20<00:20, 9761.55it/s] 49%|████▉     | 196473/400000 [00:20<00:21, 9524.34it/s] 49%|████▉     | 197452/400000 [00:20<00:21, 9600.92it/s] 50%|████▉     | 198452/400000 [00:20<00:20, 9714.68it/s] 50%|████▉     | 199448/400000 [00:20<00:20, 9785.99it/s] 50%|█████     | 200428/400000 [00:20<00:20, 9521.26it/s] 50%|█████     | 201405/400000 [00:21<00:20, 9591.35it/s] 51%|█████     | 202366/400000 [00:21<00:20, 9553.71it/s] 51%|█████     | 203351/400000 [00:21<00:20, 9638.64it/s] 51%|█████     | 204347/400000 [00:21<00:20, 9730.23it/s] 51%|█████▏    | 205349/400000 [00:21<00:19, 9812.85it/s] 52%|█████▏    | 206359/400000 [00:21<00:19, 9895.03it/s] 52%|█████▏    | 207350/400000 [00:21<00:19, 9831.16it/s] 52%|█████▏    | 208354/400000 [00:21<00:19, 9891.81it/s] 52%|█████▏    | 209382/400000 [00:21<00:19, 10004.25it/s] 53%|█████▎    | 210384/400000 [00:21<00:19, 9975.15it/s]  53%|█████▎    | 211383/400000 [00:22<00:18, 9960.62it/s] 53%|█████▎    | 212410/400000 [00:22<00:18, 10051.30it/s] 53%|█████▎    | 213437/400000 [00:22<00:18, 10114.11it/s] 54%|█████▎    | 214449/400000 [00:22<00:18, 9927.88it/s]  54%|█████▍    | 215443/400000 [00:22<00:19, 9681.09it/s] 54%|█████▍    | 216449/400000 [00:22<00:18, 9789.47it/s] 54%|█████▍    | 217430/400000 [00:22<00:18, 9756.74it/s] 55%|█████▍    | 218408/400000 [00:22<00:18, 9760.85it/s] 55%|█████▍    | 219391/400000 [00:22<00:18, 9779.24it/s] 55%|█████▌    | 220370/400000 [00:22<00:18, 9723.53it/s] 55%|█████▌    | 221343/400000 [00:23<00:18, 9546.01it/s] 56%|█████▌    | 222299/400000 [00:23<00:18, 9529.80it/s] 56%|█████▌    | 223321/400000 [00:23<00:18, 9725.88it/s] 56%|█████▌    | 224296/400000 [00:23<00:18, 9665.97it/s] 56%|█████▋    | 225296/400000 [00:23<00:17, 9762.52it/s] 57%|█████▋    | 226285/400000 [00:23<00:17, 9798.91it/s] 57%|█████▋    | 227272/400000 [00:23<00:17, 9819.07it/s] 57%|█████▋    | 228255/400000 [00:23<00:17, 9807.32it/s] 57%|█████▋    | 229237/400000 [00:23<00:17, 9717.00it/s] 58%|█████▊    | 230210/400000 [00:23<00:18, 9413.30it/s] 58%|█████▊    | 231154/400000 [00:24<00:17, 9411.66it/s] 58%|█████▊    | 232142/400000 [00:24<00:17, 9546.55it/s] 58%|█████▊    | 233186/400000 [00:24<00:17, 9796.95it/s] 59%|█████▊    | 234169/400000 [00:24<00:17, 9743.04it/s] 59%|█████▉    | 235146/400000 [00:24<00:16, 9742.28it/s] 59%|█████▉    | 236126/400000 [00:24<00:16, 9756.73it/s] 59%|█████▉    | 237103/400000 [00:24<00:16, 9726.07it/s] 60%|█████▉    | 238152/400000 [00:24<00:16, 9940.39it/s] 60%|█████▉    | 239192/400000 [00:24<00:15, 10072.46it/s] 60%|██████    | 240201/400000 [00:24<00:16, 9951.50it/s]  60%|██████    | 241211/400000 [00:25<00:15, 9994.67it/s] 61%|██████    | 242212/400000 [00:25<00:15, 9996.59it/s] 61%|██████    | 243263/400000 [00:25<00:15, 10142.48it/s] 61%|██████    | 244304/400000 [00:25<00:15, 10220.79it/s] 61%|██████▏   | 245327/400000 [00:25<00:15, 10011.90it/s] 62%|██████▏   | 246330/400000 [00:25<00:15, 9891.45it/s]  62%|██████▏   | 247321/400000 [00:25<00:16, 9539.73it/s] 62%|██████▏   | 248279/400000 [00:25<00:16, 9479.65it/s] 62%|██████▏   | 249236/400000 [00:25<00:15, 9503.72it/s] 63%|██████▎   | 250189/400000 [00:25<00:16, 9169.65it/s] 63%|██████▎   | 251127/400000 [00:26<00:16, 9229.72it/s] 63%|██████▎   | 252068/400000 [00:26<00:15, 9281.98it/s] 63%|██████▎   | 252999/400000 [00:26<00:15, 9268.00it/s] 63%|██████▎   | 253969/400000 [00:26<00:15, 9391.20it/s] 64%|██████▎   | 254910/400000 [00:26<00:16, 9003.67it/s] 64%|██████▍   | 255857/400000 [00:26<00:15, 9137.63it/s] 64%|██████▍   | 256873/400000 [00:26<00:15, 9422.04it/s] 64%|██████▍   | 257824/400000 [00:26<00:15, 9448.20it/s] 65%|██████▍   | 258820/400000 [00:26<00:14, 9591.59it/s] 65%|██████▍   | 259783/400000 [00:27<00:14, 9387.46it/s] 65%|██████▌   | 260773/400000 [00:27<00:14, 9533.81it/s] 65%|██████▌   | 261766/400000 [00:27<00:14, 9649.24it/s] 66%|██████▌   | 262755/400000 [00:27<00:14, 9718.46it/s] 66%|██████▌   | 263729/400000 [00:27<00:14, 9660.80it/s] 66%|██████▌   | 264742/400000 [00:27<00:13, 9796.40it/s] 66%|██████▋   | 265725/400000 [00:27<00:13, 9803.39it/s] 67%|██████▋   | 266708/400000 [00:27<00:13, 9808.82it/s] 67%|██████▋   | 267708/400000 [00:27<00:13, 9862.28it/s] 67%|██████▋   | 268695/400000 [00:27<00:13, 9809.64it/s] 67%|██████▋   | 269696/400000 [00:28<00:13, 9868.39it/s] 68%|██████▊   | 270684/400000 [00:28<00:13, 9709.39it/s] 68%|██████▊   | 271721/400000 [00:28<00:12, 9896.06it/s] 68%|██████▊   | 272730/400000 [00:28<00:12, 9950.43it/s] 68%|██████▊   | 273727/400000 [00:28<00:12, 9834.74it/s] 69%|██████▊   | 274739/400000 [00:28<00:12, 9917.34it/s] 69%|██████▉   | 275749/400000 [00:28<00:12, 9970.77it/s] 69%|██████▉   | 276747/400000 [00:28<00:12, 9754.87it/s] 69%|██████▉   | 277726/400000 [00:28<00:12, 9762.87it/s] 70%|██████▉   | 278753/400000 [00:28<00:12, 9906.94it/s] 70%|██████▉   | 279745/400000 [00:29<00:12, 9545.04it/s] 70%|███████   | 280752/400000 [00:29<00:12, 9694.93it/s] 70%|███████   | 281725/400000 [00:29<00:12, 9686.16it/s] 71%|███████   | 282705/400000 [00:29<00:12, 9718.01it/s] 71%|███████   | 283679/400000 [00:29<00:12, 9686.38it/s] 71%|███████   | 284649/400000 [00:29<00:12, 9577.28it/s] 71%|███████▏  | 285608/400000 [00:29<00:12, 9507.53it/s] 72%|███████▏  | 286619/400000 [00:29<00:11, 9680.57it/s] 72%|███████▏  | 287651/400000 [00:29<00:11, 9863.47it/s] 72%|███████▏  | 288640/400000 [00:29<00:11, 9766.76it/s] 72%|███████▏  | 289619/400000 [00:30<00:11, 9684.49it/s] 73%|███████▎  | 290605/400000 [00:30<00:11, 9735.72it/s] 73%|███████▎  | 291611/400000 [00:30<00:11, 9827.98it/s] 73%|███████▎  | 292595/400000 [00:30<00:11, 9713.76it/s] 73%|███████▎  | 293568/400000 [00:30<00:11, 9660.41it/s] 74%|███████▎  | 294579/400000 [00:30<00:10, 9789.11it/s] 74%|███████▍  | 295559/400000 [00:30<00:10, 9591.34it/s] 74%|███████▍  | 296551/400000 [00:30<00:10, 9686.84it/s] 74%|███████▍  | 297560/400000 [00:30<00:10, 9804.10it/s] 75%|███████▍  | 298547/400000 [00:30<00:10, 9820.79it/s] 75%|███████▍  | 299535/400000 [00:31<00:10, 9836.24it/s] 75%|███████▌  | 300520/400000 [00:31<00:10, 9644.29it/s] 75%|███████▌  | 301486/400000 [00:31<00:10, 9464.97it/s] 76%|███████▌  | 302497/400000 [00:31<00:10, 9648.09it/s] 76%|███████▌  | 303464/400000 [00:31<00:10, 9611.80it/s] 76%|███████▌  | 304467/400000 [00:31<00:09, 9733.14it/s] 76%|███████▋  | 305480/400000 [00:31<00:09, 9847.95it/s] 77%|███████▋  | 306499/400000 [00:31<00:09, 9945.57it/s] 77%|███████▋  | 307495/400000 [00:31<00:09, 9923.08it/s] 77%|███████▋  | 308489/400000 [00:32<00:09, 9677.72it/s] 77%|███████▋  | 309459/400000 [00:32<00:09, 9534.98it/s] 78%|███████▊  | 310491/400000 [00:32<00:09, 9755.70it/s] 78%|███████▊  | 311470/400000 [00:32<00:09, 9615.97it/s] 78%|███████▊  | 312434/400000 [00:32<00:09, 9365.89it/s] 78%|███████▊  | 313374/400000 [00:32<00:09, 9295.05it/s] 79%|███████▊  | 314376/400000 [00:32<00:09, 9499.98it/s] 79%|███████▉  | 315329/400000 [00:32<00:08, 9413.82it/s] 79%|███████▉  | 316273/400000 [00:32<00:09, 9289.72it/s] 79%|███████▉  | 317309/400000 [00:32<00:08, 9586.37it/s] 80%|███████▉  | 318301/400000 [00:33<00:08, 9683.46it/s] 80%|███████▉  | 319273/400000 [00:33<00:08, 9683.38it/s] 80%|████████  | 320264/400000 [00:33<00:08, 9749.49it/s] 80%|████████  | 321322/400000 [00:33<00:07, 9981.60it/s] 81%|████████  | 322323/400000 [00:33<00:07, 9962.77it/s] 81%|████████  | 323344/400000 [00:33<00:07, 10035.45it/s] 81%|████████  | 324349/400000 [00:33<00:07, 9637.79it/s]  81%|████████▏ | 325336/400000 [00:33<00:07, 9704.51it/s] 82%|████████▏ | 326349/400000 [00:33<00:07, 9827.78it/s] 82%|████████▏ | 327368/400000 [00:33<00:07, 9933.35it/s] 82%|████████▏ | 328364/400000 [00:34<00:07, 9850.61it/s] 82%|████████▏ | 329356/400000 [00:34<00:07, 9869.31it/s] 83%|████████▎ | 330359/400000 [00:34<00:07, 9914.86it/s] 83%|████████▎ | 331423/400000 [00:34<00:06, 10120.78it/s] 83%|████████▎ | 332437/400000 [00:34<00:06, 10078.41it/s] 83%|████████▎ | 333447/400000 [00:34<00:06, 9746.88it/s]  84%|████████▎ | 334442/400000 [00:34<00:06, 9806.37it/s] 84%|████████▍ | 335438/400000 [00:34<00:06, 9849.36it/s] 84%|████████▍ | 336488/400000 [00:34<00:06, 10034.50it/s] 84%|████████▍ | 337494/400000 [00:34<00:06, 9961.82it/s]  85%|████████▍ | 338492/400000 [00:35<00:06, 9840.94it/s] 85%|████████▍ | 339478/400000 [00:35<00:06, 9798.79it/s] 85%|████████▌ | 340502/400000 [00:35<00:05, 9926.24it/s] 85%|████████▌ | 341506/400000 [00:35<00:05, 9959.90it/s] 86%|████████▌ | 342503/400000 [00:35<00:05, 9957.83it/s] 86%|████████▌ | 343500/400000 [00:35<00:05, 9731.73it/s] 86%|████████▌ | 344496/400000 [00:35<00:05, 9798.53it/s] 86%|████████▋ | 345485/400000 [00:35<00:05, 9823.37it/s] 87%|████████▋ | 346528/400000 [00:35<00:05, 9997.16it/s] 87%|████████▋ | 347530/400000 [00:35<00:05, 9864.11it/s] 87%|████████▋ | 348518/400000 [00:36<00:05, 9603.69it/s] 87%|████████▋ | 349512/400000 [00:36<00:05, 9699.99it/s] 88%|████████▊ | 350484/400000 [00:36<00:05, 9673.49it/s] 88%|████████▊ | 351453/400000 [00:36<00:05, 9424.10it/s] 88%|████████▊ | 352436/400000 [00:36<00:04, 9541.76it/s] 88%|████████▊ | 353393/400000 [00:36<00:04, 9499.34it/s] 89%|████████▊ | 354409/400000 [00:36<00:04, 9687.32it/s] 89%|████████▉ | 355380/400000 [00:36<00:04, 9541.89it/s] 89%|████████▉ | 356337/400000 [00:36<00:04, 9535.75it/s] 89%|████████▉ | 357372/400000 [00:37<00:04, 9764.01it/s] 90%|████████▉ | 358351/400000 [00:37<00:04, 9679.67it/s] 90%|████████▉ | 359352/400000 [00:37<00:04, 9774.26it/s] 90%|█████████ | 360334/400000 [00:37<00:04, 9787.29it/s] 90%|█████████ | 361314/400000 [00:37<00:03, 9717.89it/s] 91%|█████████ | 362287/400000 [00:37<00:03, 9634.15it/s] 91%|█████████ | 363252/400000 [00:37<00:03, 9365.51it/s] 91%|█████████ | 364191/400000 [00:37<00:03, 9283.13it/s] 91%|█████████▏| 365179/400000 [00:37<00:03, 9452.59it/s] 92%|█████████▏| 366167/400000 [00:37<00:03, 9574.42it/s] 92%|█████████▏| 367169/400000 [00:38<00:03, 9702.42it/s] 92%|█████████▏| 368141/400000 [00:38<00:03, 9506.35it/s] 92%|█████████▏| 369094/400000 [00:38<00:03, 9437.13it/s] 93%|█████████▎| 370040/400000 [00:38<00:03, 9221.99it/s] 93%|█████████▎| 370965/400000 [00:38<00:03, 9136.56it/s] 93%|█████████▎| 371953/400000 [00:38<00:03, 9346.58it/s] 93%|█████████▎| 372919/400000 [00:38<00:02, 9438.35it/s] 93%|█████████▎| 373911/400000 [00:38<00:02, 9577.46it/s] 94%|█████████▎| 374885/400000 [00:38<00:02, 9623.39it/s] 94%|█████████▍| 375849/400000 [00:38<00:02, 9588.83it/s] 94%|█████████▍| 376809/400000 [00:39<00:02, 9446.07it/s] 94%|█████████▍| 377755/400000 [00:39<00:02, 9072.14it/s] 95%|█████████▍| 378667/400000 [00:39<00:02, 8991.84it/s] 95%|█████████▍| 379612/400000 [00:39<00:02, 9121.89it/s] 95%|█████████▌| 380585/400000 [00:39<00:02, 9294.02it/s] 95%|█████████▌| 381524/400000 [00:39<00:01, 9322.07it/s] 96%|█████████▌| 382513/400000 [00:39<00:01, 9485.00it/s] 96%|█████████▌| 383473/400000 [00:39<00:01, 9518.52it/s] 96%|█████████▌| 384447/400000 [00:39<00:01, 9583.57it/s] 96%|█████████▋| 385414/400000 [00:39<00:01, 9606.39it/s] 97%|█████████▋| 386376/400000 [00:40<00:01, 9387.04it/s] 97%|█████████▋| 387365/400000 [00:40<00:01, 9532.23it/s] 97%|█████████▋| 388382/400000 [00:40<00:01, 9712.56it/s] 97%|█████████▋| 389356/400000 [00:40<00:01, 9561.22it/s] 98%|█████████▊| 390393/400000 [00:40<00:00, 9788.67it/s] 98%|█████████▊| 391398/400000 [00:40<00:00, 9864.48it/s] 98%|█████████▊| 392387/400000 [00:40<00:00, 9691.97it/s] 98%|█████████▊| 393434/400000 [00:40<00:00, 9912.55it/s] 99%|█████████▊| 394428/400000 [00:40<00:00, 9811.98it/s] 99%|█████████▉| 395429/400000 [00:41<00:00, 9870.18it/s] 99%|█████████▉| 396433/400000 [00:41<00:00, 9919.38it/s] 99%|█████████▉| 397427/400000 [00:41<00:00, 9682.59it/s]100%|█████████▉| 398424/400000 [00:41<00:00, 9766.90it/s]100%|█████████▉| 399417/400000 [00:41<00:00, 9814.94it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9642.82it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc2a5177d68> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011076394665735815 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011253581796601464 	 Accuracy: 48

  model saves at 48% accuracy 

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

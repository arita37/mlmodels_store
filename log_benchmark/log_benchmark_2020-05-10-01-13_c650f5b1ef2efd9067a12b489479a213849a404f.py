
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7feca8f53470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 01:13:23.895565
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 01:13:23.900356
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 01:13:23.904286
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 01:13:23.908071
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fec954deb38> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 350919.6250
Epoch 2/10

1/1 [==============================] - 0s 113ms/step - loss: 237614.0469
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 134511.4219
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 69068.0703
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 36296.2852
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 20936.9121
Epoch 7/10

1/1 [==============================] - 0s 96ms/step - loss: 13261.1045
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 9052.4619
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 6559.2358
Epoch 10/10

1/1 [==============================] - 0s 92ms/step - loss: 5010.4453

  #### Inference Need return ypred, ytrue ######################### 
[[ 4.44172859e-01  9.70356560e+00  7.83994436e+00  1.08294067e+01
   1.08454733e+01  9.87114143e+00  8.97019958e+00  9.24219227e+00
   1.11855888e+01  8.50727081e+00  9.79350662e+00  7.24039364e+00
   9.26980019e+00  9.61889362e+00  7.35383177e+00  7.80242729e+00
   9.62769318e+00  1.14861164e+01  1.03833294e+01  9.23067284e+00
   1.20238848e+01  7.71840811e+00  9.93381119e+00  1.10568237e+01
   9.44315338e+00  1.02559805e+01  1.07720318e+01  1.09794331e+01
   8.41015911e+00  9.76333332e+00  9.86473942e+00  8.36129475e+00
   7.74707890e+00  6.82399464e+00  6.95833015e+00  1.01336441e+01
   9.57945347e+00  1.13258066e+01  8.34196186e+00  7.50066662e+00
   8.97457409e+00  1.05284557e+01  8.91325569e+00  9.75792885e+00
   8.23831654e+00  8.66333961e+00  1.11611462e+01  1.06219454e+01
   9.13599110e+00  1.05233364e+01  9.15997028e+00  7.96765852e+00
   8.60323906e+00  9.80697441e+00  1.02354822e+01  8.77527618e+00
   9.53536892e+00  9.82017994e+00  8.84433746e+00  1.01753635e+01
   8.41306865e-01  8.58557045e-01  1.88389242e+00  1.46951586e-01
   1.79631853e+00 -3.05252850e-01 -1.65349245e+00 -2.64567226e-01
  -6.73105001e-01 -2.19121361e+00  5.16210854e-01  7.03145921e-01
  -8.38963032e-01 -3.98261011e-01  5.01406550e-01  1.56491864e+00
   7.25689411e-01 -4.03637469e-01 -3.16302955e-01  8.02606761e-01
  -2.06197596e+00  5.63732207e-01  5.78017950e-01  1.48664892e-01
   4.52517271e-01 -1.05940163e+00 -4.25642699e-01 -1.33100355e+00
  -9.25878227e-01  1.54437351e+00 -4.35479105e-01  5.10358691e-01
  -6.41030192e-01 -6.73654914e-01 -4.07420576e-01  2.35947013e-01
  -1.99840355e+00  1.03544319e+00  6.71115220e-01 -2.99005210e-01
   2.99646735e-01 -1.17582941e+00 -1.15866947e+00 -8.17676127e-01
   4.65015888e-01  1.39905918e+00 -3.08646321e-01  1.50418133e-01
  -1.85340691e+00  1.12327409e+00 -1.40273881e+00 -2.67236620e-01
  -1.12796402e+00 -1.76192677e+00 -6.60232604e-01  5.65258145e-01
   1.51893079e+00 -1.51921046e+00 -5.20017147e-01  9.41010237e-01
   1.68024921e+00 -2.87601995e+00  1.14845514e+00  9.55024064e-01
   1.76289296e+00 -2.23988581e+00 -1.33353114e-01 -9.89729166e-02
  -4.03800964e-01 -2.80046225e+00 -9.78465259e-01 -3.51321101e-02
   1.38483834e+00  3.85301948e-01  4.39805597e-01  2.58084059e-01
   9.25888479e-01  5.29344201e-01 -1.27912974e+00  5.61479092e-01
   1.36249852e+00 -1.01764703e+00  1.16871834e+00 -2.67962646e+00
  -6.87248111e-01 -9.49321508e-01 -2.57367343e-02 -1.44496298e+00
   6.28002286e-01 -8.91113818e-01 -4.16089594e-02  1.09353852e+00
  -1.45044374e+00 -1.47173500e+00 -1.10182226e+00  4.52419728e-01
  -2.37673521e-04 -1.45138502e+00  1.65289402e+00 -2.42830157e-01
   4.66068089e-01  5.46908379e-01  1.26390290e+00  7.33214676e-01
  -8.83428454e-01 -9.05545652e-01  1.67133272e-01 -7.84045100e-01
   2.77469605e-01  4.42283332e-01  1.51520312e-01 -1.70162928e+00
   3.36017728e-01  2.72782534e-01  1.29600215e+00 -2.06586868e-02
   6.01697028e-01 -4.76138860e-01 -1.05488276e+00 -5.64080358e-01
   3.28602195e-01  1.12989311e+01  9.76759815e+00  1.11062584e+01
   1.02805452e+01  8.74238205e+00  1.03546247e+01  1.05225840e+01
   8.61762238e+00  8.40383720e+00  1.01541948e+01  8.88802338e+00
   8.92561626e+00  9.72140217e+00  9.41794968e+00  1.01757574e+01
   9.86637592e+00  9.63934612e+00  1.05193892e+01  9.12091446e+00
   1.10487213e+01  1.04674835e+01  9.64415359e+00  1.06357784e+01
   9.48121643e+00  8.68103313e+00  9.17430210e+00  8.12078953e+00
   9.43727970e+00  9.62899590e+00  9.65940857e+00  8.37235832e+00
   1.00321770e+01  9.25019550e+00  9.96750736e+00  1.08663578e+01
   9.61648655e+00  1.01564388e+01  1.02196331e+01  9.74163723e+00
   9.40126896e+00  1.00065432e+01  1.06715536e+01  9.01322269e+00
   7.99356365e+00  9.15301704e+00  1.00600729e+01  1.09803305e+01
   9.36655712e+00  1.06148777e+01  9.52974987e+00  1.07396193e+01
   7.85322762e+00  1.05573339e+01  1.07081156e+01  9.13077354e+00
   9.95927620e+00  9.17292118e+00  8.75273323e+00  1.09644480e+01
   6.03728235e-01  9.56950009e-01  2.01665163e+00  2.45640564e+00
   2.89871097e-01  1.53182626e-01  4.23764110e-01  1.53380394e+00
   2.92406201e-01  9.40675855e-01  4.14843082e-01  1.69826198e+00
   1.71851850e+00  1.85452080e+00  4.42474246e-01  7.50056028e-01
   6.57023549e-01  1.42356467e+00  1.57691503e+00  1.06688118e+00
   2.08972907e+00  9.69785690e-01  1.12366259e-01  9.85853791e-01
   1.16758192e+00  6.53314710e-01  2.58229017e-01  6.06117249e-02
   2.31063890e+00  1.19474804e+00  9.46530282e-01  2.06185293e+00
   6.23739064e-01  2.18161964e+00  3.83594632e-02  1.44854581e+00
   3.44743252e-01  1.08118725e+00  3.66696715e-01  1.75251222e+00
   9.23590600e-01  1.48922086e-01  3.55679333e-01  1.64187264e+00
   2.06165743e+00  3.09115648e-01  3.16898763e-01  2.37410009e-01
   1.10295260e+00  1.41835976e+00  1.02258456e+00  5.53478301e-01
   3.00840974e-01  2.39652061e+00  1.66746783e+00  4.94567394e-01
   5.48571944e-01  7.39205420e-01  8.95718038e-01  1.09477353e+00
   1.23869777e+00  3.66208792e-01  1.29012132e+00  2.03058290e+00
   4.22894239e-01  3.43164563e-01  2.52203107e-01  2.63477325e-01
   1.10080063e-01  8.97640347e-01  2.48772836e+00  2.32196152e-01
   1.18099213e+00  2.84531355e-01  9.89477634e-02  7.50505149e-01
   9.14996803e-01  9.06145215e-01  8.10381413e-01  2.11652708e+00
   2.06688404e-01  1.05217218e+00  5.23708820e-01  1.71882987e-01
   2.58993387e+00  6.75544739e-02  1.04495800e+00  8.47862005e-01
   3.44883442e-01  1.14946973e+00  2.15898812e-01  1.08489156e-01
   8.01052451e-02  2.66794682e+00  4.30957556e-01  1.34662056e+00
   2.90874243e+00  2.88034868e+00  2.14415598e+00  2.08286190e+00
   7.22317100e-01  1.23907101e+00  7.86243796e-01  1.41855407e+00
   6.18505418e-01  1.33981395e+00  1.79414201e+00  2.57856369e+00
   1.25487137e+00  1.67821527e+00  1.78397417e+00  3.60330391e+00
   4.00923193e-01  2.46430159e+00  5.88904858e-01  4.07668769e-01
   2.18728638e+00  1.69676208e+00  2.34188795e+00  9.79084432e-01
   8.66083431e+00 -7.29800344e-01 -1.04926357e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 01:13:34.328083
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7093
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 01:13:34.332313
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8623.06
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 01:13:34.336521
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.9571
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 01:13:34.340203
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -771.256
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140653734773144
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140651222373208
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140651222373712
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140651222374216
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140651222374720
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140651222375224

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fec94a160f0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.576302
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.535368
grad_step = 000002, loss = 0.507582
grad_step = 000003, loss = 0.484266
grad_step = 000004, loss = 0.458970
grad_step = 000005, loss = 0.433459
grad_step = 000006, loss = 0.415171
grad_step = 000007, loss = 0.403960
grad_step = 000008, loss = 0.387401
grad_step = 000009, loss = 0.367420
grad_step = 000010, loss = 0.354235
grad_step = 000011, loss = 0.343879
grad_step = 000012, loss = 0.332996
grad_step = 000013, loss = 0.319852
grad_step = 000014, loss = 0.306151
grad_step = 000015, loss = 0.293048
grad_step = 000016, loss = 0.280709
grad_step = 000017, loss = 0.268516
grad_step = 000018, loss = 0.256570
grad_step = 000019, loss = 0.245689
grad_step = 000020, loss = 0.235466
grad_step = 000021, loss = 0.224257
grad_step = 000022, loss = 0.212363
grad_step = 000023, loss = 0.201402
grad_step = 000024, loss = 0.191594
grad_step = 000025, loss = 0.182027
grad_step = 000026, loss = 0.172392
grad_step = 000027, loss = 0.162991
grad_step = 000028, loss = 0.154057
grad_step = 000029, loss = 0.145345
grad_step = 000030, loss = 0.136755
grad_step = 000031, loss = 0.128521
grad_step = 000032, loss = 0.120716
grad_step = 000033, loss = 0.113307
grad_step = 000034, loss = 0.106426
grad_step = 000035, loss = 0.099881
grad_step = 000036, loss = 0.093268
grad_step = 000037, loss = 0.086811
grad_step = 000038, loss = 0.080928
grad_step = 000039, loss = 0.075538
grad_step = 000040, loss = 0.070247
grad_step = 000041, loss = 0.065237
grad_step = 000042, loss = 0.060571
grad_step = 000043, loss = 0.056187
grad_step = 000044, loss = 0.052074
grad_step = 000045, loss = 0.048165
grad_step = 000046, loss = 0.044469
grad_step = 000047, loss = 0.041106
grad_step = 000048, loss = 0.038000
grad_step = 000049, loss = 0.035031
grad_step = 000050, loss = 0.032226
grad_step = 000051, loss = 0.029632
grad_step = 000052, loss = 0.027293
grad_step = 000053, loss = 0.025119
grad_step = 000054, loss = 0.023054
grad_step = 000055, loss = 0.021196
grad_step = 000056, loss = 0.019468
grad_step = 000057, loss = 0.017843
grad_step = 000058, loss = 0.016361
grad_step = 000059, loss = 0.014990
grad_step = 000060, loss = 0.013755
grad_step = 000061, loss = 0.012621
grad_step = 000062, loss = 0.011567
grad_step = 000063, loss = 0.010599
grad_step = 000064, loss = 0.009719
grad_step = 000065, loss = 0.008926
grad_step = 000066, loss = 0.008192
grad_step = 000067, loss = 0.007531
grad_step = 000068, loss = 0.006919
grad_step = 000069, loss = 0.006380
grad_step = 000070, loss = 0.005893
grad_step = 000071, loss = 0.005442
grad_step = 000072, loss = 0.005044
grad_step = 000073, loss = 0.004692
grad_step = 000074, loss = 0.004377
grad_step = 000075, loss = 0.004088
grad_step = 000076, loss = 0.003835
grad_step = 000077, loss = 0.003613
grad_step = 000078, loss = 0.003416
grad_step = 000079, loss = 0.003237
grad_step = 000080, loss = 0.003088
grad_step = 000081, loss = 0.002956
grad_step = 000082, loss = 0.002845
grad_step = 000083, loss = 0.002755
grad_step = 000084, loss = 0.002697
grad_step = 000085, loss = 0.002672
grad_step = 000086, loss = 0.002683
grad_step = 000087, loss = 0.002658
grad_step = 000088, loss = 0.002553
grad_step = 000089, loss = 0.002384
grad_step = 000090, loss = 0.002309
grad_step = 000091, loss = 0.002344
grad_step = 000092, loss = 0.002380
grad_step = 000093, loss = 0.002332
grad_step = 000094, loss = 0.002231
grad_step = 000095, loss = 0.002195
grad_step = 000096, loss = 0.002233
grad_step = 000097, loss = 0.002251
grad_step = 000098, loss = 0.002208
grad_step = 000099, loss = 0.002152
grad_step = 000100, loss = 0.002149
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002181
grad_step = 000102, loss = 0.002187
grad_step = 000103, loss = 0.002151
grad_step = 000104, loss = 0.002116
grad_step = 000105, loss = 0.002118
grad_step = 000106, loss = 0.002139
grad_step = 000107, loss = 0.002139
grad_step = 000108, loss = 0.002115
grad_step = 000109, loss = 0.002092
grad_step = 000110, loss = 0.002091
grad_step = 000111, loss = 0.002102
grad_step = 000112, loss = 0.002105
grad_step = 000113, loss = 0.002092
grad_step = 000114, loss = 0.002074
grad_step = 000115, loss = 0.002065
grad_step = 000116, loss = 0.002067
grad_step = 000117, loss = 0.002071
grad_step = 000118, loss = 0.002068
grad_step = 000119, loss = 0.002059
grad_step = 000120, loss = 0.002047
grad_step = 000121, loss = 0.002038
grad_step = 000122, loss = 0.002035
grad_step = 000123, loss = 0.002035
grad_step = 000124, loss = 0.002035
grad_step = 000125, loss = 0.002032
grad_step = 000126, loss = 0.002026
grad_step = 000127, loss = 0.002018
grad_step = 000128, loss = 0.002011
grad_step = 000129, loss = 0.002005
grad_step = 000130, loss = 0.002000
grad_step = 000131, loss = 0.001996
grad_step = 000132, loss = 0.001994
grad_step = 000133, loss = 0.001992
grad_step = 000134, loss = 0.001990
grad_step = 000135, loss = 0.001989
grad_step = 000136, loss = 0.001988
grad_step = 000137, loss = 0.001990
grad_step = 000138, loss = 0.001993
grad_step = 000139, loss = 0.002001
grad_step = 000140, loss = 0.002017
grad_step = 000141, loss = 0.002042
grad_step = 000142, loss = 0.002084
grad_step = 000143, loss = 0.002133
grad_step = 000144, loss = 0.002188
grad_step = 000145, loss = 0.002194
grad_step = 000146, loss = 0.002147
grad_step = 000147, loss = 0.002040
grad_step = 000148, loss = 0.001952
grad_step = 000149, loss = 0.001931
grad_step = 000150, loss = 0.001972
grad_step = 000151, loss = 0.002028
grad_step = 000152, loss = 0.002047
grad_step = 000153, loss = 0.002016
grad_step = 000154, loss = 0.001953
grad_step = 000155, loss = 0.001909
grad_step = 000156, loss = 0.001907
grad_step = 000157, loss = 0.001935
grad_step = 000158, loss = 0.001963
grad_step = 000159, loss = 0.001964
grad_step = 000160, loss = 0.001939
grad_step = 000161, loss = 0.001902
grad_step = 000162, loss = 0.001878
grad_step = 000163, loss = 0.001873
grad_step = 000164, loss = 0.001881
grad_step = 000165, loss = 0.001894
grad_step = 000166, loss = 0.001904
grad_step = 000167, loss = 0.001906
grad_step = 000168, loss = 0.001898
grad_step = 000169, loss = 0.001884
grad_step = 000170, loss = 0.001867
grad_step = 000171, loss = 0.001850
grad_step = 000172, loss = 0.001836
grad_step = 000173, loss = 0.001826
grad_step = 000174, loss = 0.001818
grad_step = 000175, loss = 0.001813
grad_step = 000176, loss = 0.001809
grad_step = 000177, loss = 0.001807
grad_step = 000178, loss = 0.001810
grad_step = 000179, loss = 0.001827
grad_step = 000180, loss = 0.001879
grad_step = 000181, loss = 0.001984
grad_step = 000182, loss = 0.002204
grad_step = 000183, loss = 0.002442
grad_step = 000184, loss = 0.002623
grad_step = 000185, loss = 0.002330
grad_step = 000186, loss = 0.001892
grad_step = 000187, loss = 0.001778
grad_step = 000188, loss = 0.002039
grad_step = 000189, loss = 0.002228
grad_step = 000190, loss = 0.001998
grad_step = 000191, loss = 0.001761
grad_step = 000192, loss = 0.001835
grad_step = 000193, loss = 0.002007
grad_step = 000194, loss = 0.001964
grad_step = 000195, loss = 0.001761
grad_step = 000196, loss = 0.001760
grad_step = 000197, loss = 0.001902
grad_step = 000198, loss = 0.001880
grad_step = 000199, loss = 0.001752
grad_step = 000200, loss = 0.001720
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001808
grad_step = 000202, loss = 0.001842
grad_step = 000203, loss = 0.001750
grad_step = 000204, loss = 0.001698
grad_step = 000205, loss = 0.001744
grad_step = 000206, loss = 0.001784
grad_step = 000207, loss = 0.001756
grad_step = 000208, loss = 0.001696
grad_step = 000209, loss = 0.001692
grad_step = 000210, loss = 0.001727
grad_step = 000211, loss = 0.001736
grad_step = 000212, loss = 0.001722
grad_step = 000213, loss = 0.001687
grad_step = 000214, loss = 0.001679
grad_step = 000215, loss = 0.001681
grad_step = 000216, loss = 0.001688
grad_step = 000217, loss = 0.001701
grad_step = 000218, loss = 0.001694
grad_step = 000219, loss = 0.001682
grad_step = 000220, loss = 0.001657
grad_step = 000221, loss = 0.001647
grad_step = 000222, loss = 0.001654
grad_step = 000223, loss = 0.001662
grad_step = 000224, loss = 0.001671
grad_step = 000225, loss = 0.001672
grad_step = 000226, loss = 0.001685
grad_step = 000227, loss = 0.001685
grad_step = 000228, loss = 0.001687
grad_step = 000229, loss = 0.001669
grad_step = 000230, loss = 0.001655
grad_step = 000231, loss = 0.001639
grad_step = 000232, loss = 0.001629
grad_step = 000233, loss = 0.001623
grad_step = 000234, loss = 0.001622
grad_step = 000235, loss = 0.001627
grad_step = 000236, loss = 0.001634
grad_step = 000237, loss = 0.001644
grad_step = 000238, loss = 0.001645
grad_step = 000239, loss = 0.001650
grad_step = 000240, loss = 0.001641
grad_step = 000241, loss = 0.001634
grad_step = 000242, loss = 0.001620
grad_step = 000243, loss = 0.001609
grad_step = 000244, loss = 0.001602
grad_step = 000245, loss = 0.001600
grad_step = 000246, loss = 0.001603
grad_step = 000247, loss = 0.001609
grad_step = 000248, loss = 0.001623
grad_step = 000249, loss = 0.001643
grad_step = 000250, loss = 0.001702
grad_step = 000251, loss = 0.001745
grad_step = 000252, loss = 0.001845
grad_step = 000253, loss = 0.001771
grad_step = 000254, loss = 0.001689
grad_step = 000255, loss = 0.001626
grad_step = 000256, loss = 0.001631
grad_step = 000257, loss = 0.001648
grad_step = 000258, loss = 0.001630
grad_step = 000259, loss = 0.001638
grad_step = 000260, loss = 0.001656
grad_step = 000261, loss = 0.001628
grad_step = 000262, loss = 0.001584
grad_step = 000263, loss = 0.001578
grad_step = 000264, loss = 0.001607
grad_step = 000265, loss = 0.001622
grad_step = 000266, loss = 0.001607
grad_step = 000267, loss = 0.001591
grad_step = 000268, loss = 0.001598
grad_step = 000269, loss = 0.001619
grad_step = 000270, loss = 0.001640
grad_step = 000271, loss = 0.001621
grad_step = 000272, loss = 0.001611
grad_step = 000273, loss = 0.001617
grad_step = 000274, loss = 0.001627
grad_step = 000275, loss = 0.001617
grad_step = 000276, loss = 0.001592
grad_step = 000277, loss = 0.001573
grad_step = 000278, loss = 0.001567
grad_step = 000279, loss = 0.001565
grad_step = 000280, loss = 0.001560
grad_step = 000281, loss = 0.001555
grad_step = 000282, loss = 0.001559
grad_step = 000283, loss = 0.001570
grad_step = 000284, loss = 0.001577
grad_step = 000285, loss = 0.001579
grad_step = 000286, loss = 0.001580
grad_step = 000287, loss = 0.001587
grad_step = 000288, loss = 0.001595
grad_step = 000289, loss = 0.001604
grad_step = 000290, loss = 0.001610
grad_step = 000291, loss = 0.001619
grad_step = 000292, loss = 0.001628
grad_step = 000293, loss = 0.001645
grad_step = 000294, loss = 0.001648
grad_step = 000295, loss = 0.001651
grad_step = 000296, loss = 0.001636
grad_step = 000297, loss = 0.001620
grad_step = 000298, loss = 0.001596
grad_step = 000299, loss = 0.001572
grad_step = 000300, loss = 0.001549
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001531
grad_step = 000302, loss = 0.001522
grad_step = 000303, loss = 0.001522
grad_step = 000304, loss = 0.001528
grad_step = 000305, loss = 0.001538
grad_step = 000306, loss = 0.001548
grad_step = 000307, loss = 0.001558
grad_step = 000308, loss = 0.001571
grad_step = 000309, loss = 0.001587
grad_step = 000310, loss = 0.001616
grad_step = 000311, loss = 0.001646
grad_step = 000312, loss = 0.001701
grad_step = 000313, loss = 0.001726
grad_step = 000314, loss = 0.001753
grad_step = 000315, loss = 0.001723
grad_step = 000316, loss = 0.001675
grad_step = 000317, loss = 0.001612
grad_step = 000318, loss = 0.001563
grad_step = 000319, loss = 0.001538
grad_step = 000320, loss = 0.001526
grad_step = 000321, loss = 0.001525
grad_step = 000322, loss = 0.001536
grad_step = 000323, loss = 0.001559
grad_step = 000324, loss = 0.001583
grad_step = 000325, loss = 0.001585
grad_step = 000326, loss = 0.001566
grad_step = 000327, loss = 0.001535
grad_step = 000328, loss = 0.001511
grad_step = 000329, loss = 0.001504
grad_step = 000330, loss = 0.001503
grad_step = 000331, loss = 0.001502
grad_step = 000332, loss = 0.001494
grad_step = 000333, loss = 0.001486
grad_step = 000334, loss = 0.001485
grad_step = 000335, loss = 0.001492
grad_step = 000336, loss = 0.001504
grad_step = 000337, loss = 0.001513
grad_step = 000338, loss = 0.001526
grad_step = 000339, loss = 0.001540
grad_step = 000340, loss = 0.001568
grad_step = 000341, loss = 0.001606
grad_step = 000342, loss = 0.001657
grad_step = 000343, loss = 0.001701
grad_step = 000344, loss = 0.001733
grad_step = 000345, loss = 0.001722
grad_step = 000346, loss = 0.001679
grad_step = 000347, loss = 0.001602
grad_step = 000348, loss = 0.001526
grad_step = 000349, loss = 0.001475
grad_step = 000350, loss = 0.001462
grad_step = 000351, loss = 0.001483
grad_step = 000352, loss = 0.001519
grad_step = 000353, loss = 0.001556
grad_step = 000354, loss = 0.001577
grad_step = 000355, loss = 0.001582
grad_step = 000356, loss = 0.001563
grad_step = 000357, loss = 0.001532
grad_step = 000358, loss = 0.001495
grad_step = 000359, loss = 0.001468
grad_step = 000360, loss = 0.001454
grad_step = 000361, loss = 0.001453
grad_step = 000362, loss = 0.001462
grad_step = 000363, loss = 0.001474
grad_step = 000364, loss = 0.001485
grad_step = 000365, loss = 0.001492
grad_step = 000366, loss = 0.001495
grad_step = 000367, loss = 0.001495
grad_step = 000368, loss = 0.001495
grad_step = 000369, loss = 0.001494
grad_step = 000370, loss = 0.001498
grad_step = 000371, loss = 0.001504
grad_step = 000372, loss = 0.001522
grad_step = 000373, loss = 0.001531
grad_step = 000374, loss = 0.001545
grad_step = 000375, loss = 0.001520
grad_step = 000376, loss = 0.001491
grad_step = 000377, loss = 0.001456
grad_step = 000378, loss = 0.001446
grad_step = 000379, loss = 0.001458
grad_step = 000380, loss = 0.001468
grad_step = 000381, loss = 0.001462
grad_step = 000382, loss = 0.001439
grad_step = 000383, loss = 0.001427
grad_step = 000384, loss = 0.001431
grad_step = 000385, loss = 0.001442
grad_step = 000386, loss = 0.001452
grad_step = 000387, loss = 0.001450
grad_step = 000388, loss = 0.001447
grad_step = 000389, loss = 0.001445
grad_step = 000390, loss = 0.001458
grad_step = 000391, loss = 0.001490
grad_step = 000392, loss = 0.001547
grad_step = 000393, loss = 0.001637
grad_step = 000394, loss = 0.001766
grad_step = 000395, loss = 0.001908
grad_step = 000396, loss = 0.002051
grad_step = 000397, loss = 0.002032
grad_step = 000398, loss = 0.001873
grad_step = 000399, loss = 0.001602
grad_step = 000400, loss = 0.001428
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001451
grad_step = 000402, loss = 0.001594
grad_step = 000403, loss = 0.001700
grad_step = 000404, loss = 0.001657
grad_step = 000405, loss = 0.001514
grad_step = 000406, loss = 0.001421
grad_step = 000407, loss = 0.001447
grad_step = 000408, loss = 0.001523
grad_step = 000409, loss = 0.001549
grad_step = 000410, loss = 0.001509
grad_step = 000411, loss = 0.001445
grad_step = 000412, loss = 0.001412
grad_step = 000413, loss = 0.001424
grad_step = 000414, loss = 0.001461
grad_step = 000415, loss = 0.001500
grad_step = 000416, loss = 0.001489
grad_step = 000417, loss = 0.001447
grad_step = 000418, loss = 0.001400
grad_step = 000419, loss = 0.001389
grad_step = 000420, loss = 0.001411
grad_step = 000421, loss = 0.001432
grad_step = 000422, loss = 0.001431
grad_step = 000423, loss = 0.001413
grad_step = 000424, loss = 0.001394
grad_step = 000425, loss = 0.001384
grad_step = 000426, loss = 0.001386
grad_step = 000427, loss = 0.001397
grad_step = 000428, loss = 0.001405
grad_step = 000429, loss = 0.001401
grad_step = 000430, loss = 0.001388
grad_step = 000431, loss = 0.001377
grad_step = 000432, loss = 0.001375
grad_step = 000433, loss = 0.001379
grad_step = 000434, loss = 0.001382
grad_step = 000435, loss = 0.001383
grad_step = 000436, loss = 0.001384
grad_step = 000437, loss = 0.001383
grad_step = 000438, loss = 0.001379
grad_step = 000439, loss = 0.001374
grad_step = 000440, loss = 0.001370
grad_step = 000441, loss = 0.001369
grad_step = 000442, loss = 0.001370
grad_step = 000443, loss = 0.001374
grad_step = 000444, loss = 0.001377
grad_step = 000445, loss = 0.001383
grad_step = 000446, loss = 0.001388
grad_step = 000447, loss = 0.001396
grad_step = 000448, loss = 0.001399
grad_step = 000449, loss = 0.001403
grad_step = 000450, loss = 0.001401
grad_step = 000451, loss = 0.001398
grad_step = 000452, loss = 0.001389
grad_step = 000453, loss = 0.001380
grad_step = 000454, loss = 0.001369
grad_step = 000455, loss = 0.001360
grad_step = 000456, loss = 0.001353
grad_step = 000457, loss = 0.001348
grad_step = 000458, loss = 0.001346
grad_step = 000459, loss = 0.001346
grad_step = 000460, loss = 0.001347
grad_step = 000461, loss = 0.001350
grad_step = 000462, loss = 0.001357
grad_step = 000463, loss = 0.001367
grad_step = 000464, loss = 0.001388
grad_step = 000465, loss = 0.001405
grad_step = 000466, loss = 0.001440
grad_step = 000467, loss = 0.001441
grad_step = 000468, loss = 0.001444
grad_step = 000469, loss = 0.001409
grad_step = 000470, loss = 0.001396
grad_step = 000471, loss = 0.001433
grad_step = 000472, loss = 0.001524
grad_step = 000473, loss = 0.001675
grad_step = 000474, loss = 0.001851
grad_step = 000475, loss = 0.002025
grad_step = 000476, loss = 0.002147
grad_step = 000477, loss = 0.002023
grad_step = 000478, loss = 0.001735
grad_step = 000479, loss = 0.001424
grad_step = 000480, loss = 0.001340
grad_step = 000481, loss = 0.001494
grad_step = 000482, loss = 0.001680
grad_step = 000483, loss = 0.001690
grad_step = 000484, loss = 0.001505
grad_step = 000485, loss = 0.001338
grad_step = 000486, loss = 0.001347
grad_step = 000487, loss = 0.001467
grad_step = 000488, loss = 0.001532
grad_step = 000489, loss = 0.001466
grad_step = 000490, loss = 0.001361
grad_step = 000491, loss = 0.001324
grad_step = 000492, loss = 0.001370
grad_step = 000493, loss = 0.001425
grad_step = 000494, loss = 0.001423
grad_step = 000495, loss = 0.001376
grad_step = 000496, loss = 0.001323
grad_step = 000497, loss = 0.001317
grad_step = 000498, loss = 0.001351
grad_step = 000499, loss = 0.001375
grad_step = 000500, loss = 0.001357
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001315
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

  date_run                              2020-05-10 01:13:56.657915
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.266514
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 01:13:56.665119
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.182558
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 01:13:56.674155
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.136908
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 01:13:56.680951
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.77404
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
0   2020-05-10 01:13:23.895565  ...    mean_absolute_error
1   2020-05-10 01:13:23.900356  ...     mean_squared_error
2   2020-05-10 01:13:23.904286  ...  median_absolute_error
3   2020-05-10 01:13:23.908071  ...               r2_score
4   2020-05-10 01:13:34.328083  ...    mean_absolute_error
5   2020-05-10 01:13:34.332313  ...     mean_squared_error
6   2020-05-10 01:13:34.336521  ...  median_absolute_error
7   2020-05-10 01:13:34.340203  ...               r2_score
8   2020-05-10 01:13:56.657915  ...    mean_absolute_error
9   2020-05-10 01:13:56.665119  ...     mean_squared_error
10  2020-05-10 01:13:56.674155  ...  median_absolute_error
11  2020-05-10 01:13:56.680951  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 162302.38it/s]  1%|          | 98304/9912422 [00:00<00:46, 209192.08it/s]  4%|▍         | 434176/9912422 [00:00<00:32, 289007.48it/s] 18%|█▊        | 1753088/9912422 [00:00<00:20, 407964.09it/s] 44%|████▍     | 4341760/9912422 [00:00<00:09, 577921.04it/s] 73%|███████▎  | 7192576/9912422 [00:00<00:03, 816599.89it/s]9920512it [00:01, 9423474.36it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 146849.82it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 307462.82it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 396771.12it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 550229.26it/s]1654784it [00:00, 2774526.68it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53087.12it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4e3498f780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4dd20d2a90> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4e3498fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4e34946e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4dd20d5048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4ddb7f54a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4dd20d2a90> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4e3498ff98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4de7343c88> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4e3498ff98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4de7343c88> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9abf9ee208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b7b82f7b063b59969ec6b21e073569b3a8f237f99712d24f82544cc74fc83508
  Stored in directory: /tmp/pip-ephem-wheel-cache-2zkjg98r/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9a575d3048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 54s
   57344/17464789 [..............................] - ETA: 46s
  106496/17464789 [..............................] - ETA: 37s
  245760/17464789 [..............................] - ETA: 21s
  524288/17464789 [..............................] - ETA: 12s
 1081344/17464789 [>.............................] - ETA: 7s 
 2187264/17464789 [==>...........................] - ETA: 3s
 4382720/17464789 [======>.......................] - ETA: 1s
 7495680/17464789 [===========>..................] - ETA: 0s
10592256/17464789 [=================>............] - ETA: 0s
13672448/17464789 [======================>.......] - ETA: 0s
16752640/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 01:15:29.218016: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 01:15:29.222410: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-10 01:15:29.222558: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558410dec790 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 01:15:29.222576: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8813 - accuracy: 0.4860
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8711 - accuracy: 0.4867 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8353 - accuracy: 0.4890
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8261 - accuracy: 0.4896
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8583 - accuracy: 0.4875
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7827 - accuracy: 0.4924
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8046 - accuracy: 0.4910
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.8302 - accuracy: 0.4893
10000/25000 [===========>..................] - ETA: 4s - loss: 7.8230 - accuracy: 0.4898
11000/25000 [============>.................] - ETA: 4s - loss: 7.8297 - accuracy: 0.4894
12000/25000 [=============>................] - ETA: 4s - loss: 7.7880 - accuracy: 0.4921
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7574 - accuracy: 0.4941
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7488 - accuracy: 0.4946
15000/25000 [=================>............] - ETA: 3s - loss: 7.7188 - accuracy: 0.4966
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7193 - accuracy: 0.4966
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7198 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7015 - accuracy: 0.4977
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6690 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6732 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 9s 367us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 01:15:46.025248
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 01:15:46.025248  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 01:15:52.901401: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 01:15:52.907982: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-10 01:15:52.908239: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5622c43141b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 01:15:52.908508: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f31b5d95be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6063 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.4550 - val_crf_viterbi_accuracy: 0.1200

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3192d95630> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.2066 - accuracy: 0.5300
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7203 - accuracy: 0.4965
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8097 - accuracy: 0.4907 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8315 - accuracy: 0.4893
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8506 - accuracy: 0.4880
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7612 - accuracy: 0.4938
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6841 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6751 - accuracy: 0.4994
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
11000/25000 [============>.................] - ETA: 4s - loss: 7.7112 - accuracy: 0.4971
12000/25000 [=============>................] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6379 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6305 - accuracy: 0.5024
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6581 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6383 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6450 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 9s 371us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f314ca49208> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<15:00:50, 16.0kB/s].vector_cache/glove.6B.zip:   0%|          | 426k/862M [00:00<10:31:30, 22.7kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.47M/862M [00:00<7:19:33, 32.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:00<5:05:19, 46.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:00<3:31:40, 66.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.4M/862M [00:01<2:26:50, 94.6kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.8M/862M [00:01<1:41:32, 135kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 47.7M/862M [00:01<1:10:21, 193kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:01<49:31, 273kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:03<36:24, 369kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:03<26:33, 505kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:05<20:30, 651kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:05<15:06, 884kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:07<12:36, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:07<09:36, 1.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:09<08:46, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:09<06:56, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:11<06:53, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:11<05:37, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<04:23, 2.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<8:44:45, 25.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:13<6:06:09, 35.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:15<4:21:17, 49.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<3:03:54, 70.8kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:17<2:10:04, 99.6kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:17<1:31:43, 141kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:19<1:05:57, 195kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:19<46:49, 275kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:21<34:38, 370kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:21<24:59, 512kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:23<19:25, 657kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:23<14:19, 889kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<11:59, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<09:20, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<08:24, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<06:37, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:35, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<05:20, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:41, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:13, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:27, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<04:36, 2.69MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<05:05, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:16, 2.88MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:55, 2.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<04:10, 2.93MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:49, 2.53MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<04:04, 2.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:45, 2.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:00, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:40, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<03:58, 3.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<04:39, 2.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:57, 3.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:38, 2.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:57, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<03:11, 3.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<7:47:03, 25.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<5:25:50, 36.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<3:52:09, 50.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<2:43:22, 72.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<1:55:31, 102kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<1:21:30, 144kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<58:36, 199kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<41:39, 280kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<30:50, 376kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<22:16, 520kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<17:19, 666kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<12:48, 899kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<10:43, 1.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<08:10, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<07:29, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<05:55, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<05:54, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<04:46, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:06, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:13, 2.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<04:42, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:56, 2.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<04:30, 2.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:47, 2.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<04:23, 2.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:43, 2.97MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<04:19, 2.54MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:40, 2.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<04:16, 2.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:38, 2.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<04:14, 2.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:33, 3.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:09, 2.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<03:33, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<04:09, 2.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<03:33, 3.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<02:51, 3.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<7:00:39, 25.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:23<4:53:27, 36.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<3:28:26, 50.8kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<2:26:41, 72.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<1:43:39, 101kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<1:13:21, 143kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<52:36, 199kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<37:25, 279kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<27:40, 375kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<19:57, 520kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<15:31, 664kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<11:28, 897kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<09:35, 1.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<07:17, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:41, 1.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<05:17, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:16, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:18, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:34, 2.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:46, 2.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<04:12, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:34, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<04:01, 2.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:22, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:52, 2.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:22, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<03:49, 2.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:14, 3.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:47, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<03:13, 3.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:45, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<03:12, 3.00MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<03:43, 2.56MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<03:12, 2.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:42, 2.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<03:08, 3.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<02:32, 3.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<6:08:49, 25.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<4:16:57, 36.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<3:03:39, 51.0kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<2:09:13, 72.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<1:31:15, 102kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<1:04:16, 144kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<46:12, 200kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<32:50, 280kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<24:17, 377kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<17:29, 523kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<13:34, 669kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<10:02, 904kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<08:24, 1.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<06:25, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<05:51, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:37, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:37, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:40, 2.41MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:57, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:17, 2.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:39, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<03:03, 2.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:30, 2.47MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:58, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:24, 2.52MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:54, 2.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:21, 2.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:51, 2.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:19, 2.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<02:49, 2.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<03:17, 2.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<02:47, 3.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<03:15, 2.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:46, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<03:13, 2.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:44, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<02:12, 3.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<5:20:39, 25.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:33<3:43:05, 36.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<2:40:22, 50.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<1:52:50, 72.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<1:19:35, 101kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<55:57, 144kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<40:13, 199kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<28:35, 279kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<21:07, 375kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<15:13, 520kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<11:48, 665kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<08:43, 900kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<07:17, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<05:34, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<05:04, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<04:00, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:59, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<03:13, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:26, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:59, 2.54MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<03:12, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:40, 2.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<03:02, 2.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:33, 2.91MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<02:56, 2.51MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:28, 2.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:52, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:26, 3.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:49, 2.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:33, 2.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:50, 2.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:24, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:46, 2.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:37, 2.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:51, 2.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<02:42, 2.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:09, 3.23MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<4:18:29, 27.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:08<2:59:48, 38.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<2:08:21, 53.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<1:30:16, 76.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<1:03:39, 107kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<44:55, 152kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<32:12, 210kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<22:54, 295kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<16:57, 395kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<12:15, 546kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<09:44, 681kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<07:31, 873kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<05:25, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:58, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<04:00, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:44, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:58, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:03, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:27, 2.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:27<02:42, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:15, 2.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:32, 2.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:06, 2.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:25, 2.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:10, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:24, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:02, 2.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:21, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<02:00, 3.00MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:19, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<01:59, 2.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:17, 2.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:01, 2.90MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:16, 2.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<01:55, 3.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<01:33, 3.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<3:45:02, 25.6kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:43<2:36:10, 36.5kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<1:51:58, 50.8kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<1:18:44, 72.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<55:22, 101kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<39:01, 144kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<27:54, 199kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<19:49, 279kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<14:35, 376kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<10:32, 519kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<08:08, 665kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<06:00, 898kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<05:00, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:48, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<03:28, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:44, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:42, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:24, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:25, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:59, 2.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<02:10, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:44, 2.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:02, 2.45MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:43, 2.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:58, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:39, 2.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:55, 2.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:37, 3.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:52, 2.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:35, 3.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:50, 2.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:34, 3.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<01:49, 2.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<01:30, 3.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:45, 2.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<01:30, 3.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<01:14, 3.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<2:49:19, 26.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:18<1:57:22, 38.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:23:31, 53.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<58:43, 75.9kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<41:12, 107kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<29:02, 151kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<20:43, 209kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<14:43, 293kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<10:50, 393kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<08:00, 531kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<06:07, 683kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<04:39, 897kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:59, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:08, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:30<02:11, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<18:56, 214kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<13:30, 299kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<09:55, 401kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<07:09, 555kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<05:33, 705kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<04:06, 951kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<03:26, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:38, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<01:52, 2.03MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<03:11, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:31, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:18, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:56, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:52, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:36, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:38, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:26, 2.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:30, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:18, 2.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:25, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:16, 2.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:22, 2.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:15, 2.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:21, 2.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:12, 2.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:19, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:11, 2.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:17, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:11, 2.65MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:16, 2.43MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:08, 2.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:14, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:05, 2.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:11, 2.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:03, 2.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:09, 2.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:01, 2.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:07, 2.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:01, 2.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:06, 2.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<00:58, 2.80MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:04, 2.49MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:58, 2.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:03, 2.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:54, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:00, 2.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:53, 2.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:59, 2.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:53, 2.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:57, 2.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:53, 2.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<00:40, 3.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<2:08:55, 18.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<1:29:48, 25.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:01:44, 36.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<43:19, 52.1kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<29:47, 73.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<21:01, 104kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:25<14:18, 148kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<27:11, 78.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<19:07, 111kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<13:18, 155kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<09:29, 216kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<06:43, 295kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<04:51, 407kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<03:35, 535kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<02:42, 707kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:33<01:51, 1.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<05:09, 359kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<03:46, 488kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:35<02:35, 693kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<03:37, 491kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<02:45, 646kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:37<01:52, 915kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<03:26, 498kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<02:33, 666kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:58, 834kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:31, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:41<01:02, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<05:21, 293kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<03:50, 408kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<02:49, 534kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<02:08, 703kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<01:27, 994kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:42, 845kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:18, 1.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:04, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:51, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:49<00:35, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:36, 807kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:13, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:51<00:52, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<00:41, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<1:10:50, 17.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<49:01, 24.4kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<32:57, 34.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<23:00, 49.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<15:23, 69.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<10:46, 99.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<07:16, 138kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<05:07, 195kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<03:30, 267kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<02:29, 372kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:46, 491kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:18, 662kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:57, 828kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:44, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:35, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:27, 1.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:23, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:19, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:17, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:14, 2.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:14, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:12, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:10, 2.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 2.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:07, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.73MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 705/400000 [00:00<00:56, 7047.25it/s]  0%|          | 1436/400000 [00:00<00:55, 7123.32it/s]  1%|          | 2147/400000 [00:00<00:55, 7117.59it/s]  1%|          | 2883/400000 [00:00<00:55, 7188.61it/s]  1%|          | 3625/400000 [00:00<00:54, 7254.47it/s]  1%|          | 4361/400000 [00:00<00:54, 7285.41it/s]  1%|▏         | 5074/400000 [00:00<00:54, 7237.35it/s]  1%|▏         | 5807/400000 [00:00<00:54, 7264.20it/s]  2%|▏         | 6493/400000 [00:00<00:55, 7122.28it/s]  2%|▏         | 7178/400000 [00:01<00:56, 6964.06it/s]  2%|▏         | 7856/400000 [00:01<00:56, 6886.64it/s]  2%|▏         | 8532/400000 [00:01<00:57, 6841.40it/s]  2%|▏         | 9220/400000 [00:01<00:57, 6851.19it/s]  2%|▏         | 9954/400000 [00:01<00:55, 6990.47it/s]  3%|▎         | 10690/400000 [00:01<00:54, 7096.54it/s]  3%|▎         | 11440/400000 [00:01<00:53, 7211.56it/s]  3%|▎         | 12161/400000 [00:01<00:53, 7186.23it/s]  3%|▎         | 12896/400000 [00:01<00:53, 7232.42it/s]  3%|▎         | 13635/400000 [00:01<00:53, 7277.74it/s]  4%|▎         | 14371/400000 [00:02<00:52, 7300.25it/s]  4%|▍         | 15110/400000 [00:02<00:52, 7325.15it/s]  4%|▍         | 15843/400000 [00:02<00:53, 7139.09it/s]  4%|▍         | 16574/400000 [00:02<00:53, 7188.94it/s]  4%|▍         | 17318/400000 [00:02<00:52, 7262.09it/s]  5%|▍         | 18053/400000 [00:02<00:52, 7285.20it/s]  5%|▍         | 18783/400000 [00:02<00:52, 7269.56it/s]  5%|▍         | 19511/400000 [00:02<00:52, 7249.86it/s]  5%|▌         | 20237/400000 [00:02<00:52, 7213.59it/s]  5%|▌         | 20959/400000 [00:02<00:52, 7193.64it/s]  5%|▌         | 21679/400000 [00:03<00:53, 7097.15it/s]  6%|▌         | 22416/400000 [00:03<00:52, 7176.29it/s]  6%|▌         | 23155/400000 [00:03<00:52, 7236.81it/s]  6%|▌         | 23880/400000 [00:03<00:52, 7107.82it/s]  6%|▌         | 24620/400000 [00:03<00:52, 7192.03it/s]  6%|▋         | 25358/400000 [00:03<00:51, 7245.96it/s]  7%|▋         | 26097/400000 [00:03<00:51, 7285.79it/s]  7%|▋         | 26827/400000 [00:03<00:51, 7285.93it/s]  7%|▋         | 27566/400000 [00:03<00:50, 7315.97it/s]  7%|▋         | 28312/400000 [00:03<00:50, 7356.59it/s]  7%|▋         | 29056/400000 [00:04<00:50, 7380.71it/s]  7%|▋         | 29795/400000 [00:04<00:50, 7321.28it/s]  8%|▊         | 30528/400000 [00:04<00:51, 7197.43it/s]  8%|▊         | 31273/400000 [00:04<00:50, 7270.54it/s]  8%|▊         | 32001/400000 [00:04<00:50, 7254.99it/s]  8%|▊         | 32743/400000 [00:04<00:50, 7302.34it/s]  8%|▊         | 33478/400000 [00:04<00:50, 7315.12it/s]  9%|▊         | 34210/400000 [00:04<00:50, 7295.13it/s]  9%|▊         | 34940/400000 [00:04<00:50, 7295.72it/s]  9%|▉         | 35670/400000 [00:04<00:50, 7240.14it/s]  9%|▉         | 36419/400000 [00:05<00:49, 7312.16it/s]  9%|▉         | 37151/400000 [00:05<00:50, 7213.59it/s]  9%|▉         | 37884/400000 [00:05<00:49, 7247.94it/s] 10%|▉         | 38610/400000 [00:05<00:50, 7191.83it/s] 10%|▉         | 39350/400000 [00:05<00:49, 7250.59it/s] 10%|█         | 40083/400000 [00:05<00:49, 7272.55it/s] 10%|█         | 40820/400000 [00:05<00:49, 7299.60it/s] 10%|█         | 41556/400000 [00:05<00:48, 7316.04it/s] 11%|█         | 42288/400000 [00:05<00:49, 7274.36it/s] 11%|█         | 43026/400000 [00:05<00:48, 7304.97it/s] 11%|█         | 43757/400000 [00:06<00:48, 7298.83it/s] 11%|█         | 44487/400000 [00:06<00:49, 7227.12it/s] 11%|█▏        | 45219/400000 [00:06<00:48, 7252.27it/s] 11%|█▏        | 45953/400000 [00:06<00:48, 7277.12it/s] 12%|█▏        | 46690/400000 [00:06<00:48, 7304.50it/s] 12%|█▏        | 47421/400000 [00:06<00:48, 7301.70it/s] 12%|█▏        | 48153/400000 [00:06<00:48, 7306.97it/s] 12%|█▏        | 48884/400000 [00:06<00:48, 7273.72it/s] 12%|█▏        | 49612/400000 [00:06<00:49, 7033.42it/s] 13%|█▎        | 50318/400000 [00:06<00:50, 6880.93it/s] 13%|█▎        | 51033/400000 [00:07<00:50, 6957.63it/s] 13%|█▎        | 51773/400000 [00:07<00:49, 7083.54it/s] 13%|█▎        | 52514/400000 [00:07<00:48, 7177.68it/s] 13%|█▎        | 53257/400000 [00:07<00:47, 7250.18it/s] 14%|█▎        | 54005/400000 [00:07<00:47, 7317.41it/s] 14%|█▎        | 54756/400000 [00:07<00:46, 7371.59it/s] 14%|█▍        | 55495/400000 [00:07<00:46, 7375.57it/s] 14%|█▍        | 56244/400000 [00:07<00:46, 7407.90it/s] 14%|█▍        | 56986/400000 [00:07<00:46, 7400.41it/s] 14%|█▍        | 57727/400000 [00:07<00:47, 7152.62it/s] 15%|█▍        | 58472/400000 [00:08<00:47, 7238.51it/s] 15%|█▍        | 59198/400000 [00:08<00:47, 7177.28it/s] 15%|█▍        | 59941/400000 [00:08<00:46, 7251.23it/s] 15%|█▌        | 60680/400000 [00:08<00:46, 7290.38it/s] 15%|█▌        | 61421/400000 [00:08<00:46, 7323.85it/s] 16%|█▌        | 62171/400000 [00:08<00:45, 7373.21it/s] 16%|█▌        | 62913/400000 [00:08<00:45, 7385.95it/s] 16%|█▌        | 63656/400000 [00:08<00:45, 7398.58it/s] 16%|█▌        | 64397/400000 [00:08<00:45, 7397.36it/s] 16%|█▋        | 65148/400000 [00:08<00:45, 7430.01it/s] 16%|█▋        | 65892/400000 [00:09<00:45, 7399.32it/s] 17%|█▋        | 66640/400000 [00:09<00:44, 7422.56it/s] 17%|█▋        | 67383/400000 [00:09<00:44, 7413.10it/s] 17%|█▋        | 68127/400000 [00:09<00:44, 7420.39it/s] 17%|█▋        | 68873/400000 [00:09<00:44, 7430.46it/s] 17%|█▋        | 69617/400000 [00:09<00:44, 7406.17it/s] 18%|█▊        | 70358/400000 [00:09<00:44, 7369.18it/s] 18%|█▊        | 71095/400000 [00:09<00:44, 7365.10it/s] 18%|█▊        | 71832/400000 [00:09<00:45, 7224.45it/s] 18%|█▊        | 72556/400000 [00:10<00:47, 6948.83it/s] 18%|█▊        | 73254/400000 [00:10<00:48, 6788.59it/s] 18%|█▊        | 73945/400000 [00:10<00:47, 6822.54it/s] 19%|█▊        | 74673/400000 [00:10<00:46, 6953.24it/s] 19%|█▉        | 75415/400000 [00:10<00:45, 7086.83it/s] 19%|█▉        | 76142/400000 [00:10<00:45, 7138.79it/s] 19%|█▉        | 76880/400000 [00:10<00:44, 7208.46it/s] 19%|█▉        | 77626/400000 [00:10<00:44, 7280.69it/s] 20%|█▉        | 78366/400000 [00:10<00:43, 7315.81it/s] 20%|█▉        | 79099/400000 [00:10<00:44, 7269.30it/s] 20%|█▉        | 79827/400000 [00:11<00:46, 6938.26it/s] 20%|██        | 80573/400000 [00:11<00:45, 7086.81it/s] 20%|██        | 81320/400000 [00:11<00:44, 7197.46it/s] 21%|██        | 82053/400000 [00:11<00:43, 7234.72it/s] 21%|██        | 82779/400000 [00:11<00:45, 7046.57it/s] 21%|██        | 83487/400000 [00:11<00:46, 6867.87it/s] 21%|██        | 84205/400000 [00:11<00:45, 6954.37it/s] 21%|██        | 84903/400000 [00:11<00:46, 6747.16it/s] 21%|██▏       | 85612/400000 [00:11<00:45, 6845.96it/s] 22%|██▏       | 86346/400000 [00:11<00:44, 6985.37it/s] 22%|██▏       | 87093/400000 [00:12<00:43, 7123.75it/s] 22%|██▏       | 87840/400000 [00:12<00:43, 7222.90it/s] 22%|██▏       | 88583/400000 [00:12<00:42, 7282.78it/s] 22%|██▏       | 89330/400000 [00:12<00:42, 7337.00it/s] 23%|██▎       | 90074/400000 [00:12<00:42, 7364.91it/s] 23%|██▎       | 90813/400000 [00:12<00:41, 7370.62it/s] 23%|██▎       | 91551/400000 [00:12<00:41, 7372.68it/s] 23%|██▎       | 92289/400000 [00:12<00:41, 7355.82it/s] 23%|██▎       | 93025/400000 [00:12<00:44, 6928.20it/s] 23%|██▎       | 93724/400000 [00:13<00:44, 6816.73it/s] 24%|██▎       | 94410/400000 [00:13<00:45, 6698.46it/s] 24%|██▍       | 95145/400000 [00:13<00:44, 6880.67it/s] 24%|██▍       | 95881/400000 [00:13<00:43, 7017.46it/s] 24%|██▍       | 96618/400000 [00:13<00:42, 7118.24it/s] 24%|██▍       | 97333/400000 [00:13<00:42, 7062.61it/s] 25%|██▍       | 98042/400000 [00:13<00:43, 6881.99it/s] 25%|██▍       | 98733/400000 [00:13<00:43, 6863.82it/s] 25%|██▍       | 99462/400000 [00:13<00:43, 6985.98it/s] 25%|██▌       | 100163/400000 [00:13<00:43, 6892.51it/s] 25%|██▌       | 100904/400000 [00:14<00:42, 7039.13it/s] 25%|██▌       | 101647/400000 [00:14<00:41, 7150.96it/s] 26%|██▌       | 102364/400000 [00:14<00:41, 7152.60it/s] 26%|██▌       | 103081/400000 [00:14<00:41, 7105.78it/s] 26%|██▌       | 103820/400000 [00:14<00:41, 7186.83it/s] 26%|██▌       | 104555/400000 [00:14<00:40, 7233.00it/s] 26%|██▋       | 105301/400000 [00:14<00:40, 7299.07it/s] 27%|██▋       | 106032/400000 [00:14<00:40, 7176.76it/s] 27%|██▋       | 106751/400000 [00:14<00:41, 7002.92it/s] 27%|██▋       | 107453/400000 [00:14<00:41, 6990.16it/s] 27%|██▋       | 108190/400000 [00:15<00:41, 7099.71it/s] 27%|██▋       | 108939/400000 [00:15<00:40, 7211.51it/s] 27%|██▋       | 109700/400000 [00:15<00:39, 7324.15it/s] 28%|██▊       | 110454/400000 [00:15<00:39, 7385.86it/s] 28%|██▊       | 111199/400000 [00:15<00:39, 7403.14it/s] 28%|██▊       | 111949/400000 [00:15<00:38, 7430.18it/s] 28%|██▊       | 112698/400000 [00:15<00:38, 7445.19it/s] 28%|██▊       | 113443/400000 [00:15<00:39, 7170.74it/s] 29%|██▊       | 114163/400000 [00:15<00:40, 6978.04it/s] 29%|██▊       | 114864/400000 [00:15<00:42, 6707.87it/s] 29%|██▉       | 115615/400000 [00:16<00:41, 6927.86it/s] 29%|██▉       | 116359/400000 [00:16<00:40, 7072.60it/s] 29%|██▉       | 117108/400000 [00:16<00:39, 7190.90it/s] 29%|██▉       | 117854/400000 [00:16<00:38, 7267.55it/s] 30%|██▉       | 118584/400000 [00:16<00:40, 7024.61it/s] 30%|██▉       | 119290/400000 [00:16<00:40, 7009.14it/s] 30%|███       | 120031/400000 [00:16<00:39, 7122.83it/s] 30%|███       | 120768/400000 [00:16<00:38, 7193.79it/s] 30%|███       | 121500/400000 [00:16<00:38, 7228.97it/s] 31%|███       | 122243/400000 [00:17<00:38, 7287.80it/s] 31%|███       | 122973/400000 [00:17<00:39, 7062.28it/s] 31%|███       | 123682/400000 [00:17<00:39, 6920.98it/s] 31%|███       | 124377/400000 [00:17<00:40, 6885.14it/s] 31%|███▏      | 125121/400000 [00:17<00:39, 7040.96it/s] 31%|███▏      | 125828/400000 [00:17<00:39, 6948.48it/s] 32%|███▏      | 126558/400000 [00:17<00:38, 7049.31it/s] 32%|███▏      | 127284/400000 [00:17<00:38, 7110.40it/s] 32%|███▏      | 128011/400000 [00:17<00:38, 7155.55it/s] 32%|███▏      | 128737/400000 [00:17<00:37, 7183.72it/s] 32%|███▏      | 129466/400000 [00:18<00:37, 7212.50it/s] 33%|███▎      | 130193/400000 [00:18<00:37, 7229.52it/s] 33%|███▎      | 130918/400000 [00:18<00:37, 7233.06it/s] 33%|███▎      | 131670/400000 [00:18<00:36, 7315.37it/s] 33%|███▎      | 132402/400000 [00:18<00:36, 7314.74it/s] 33%|███▎      | 133134/400000 [00:18<00:36, 7277.16it/s] 33%|███▎      | 133862/400000 [00:18<00:36, 7229.51it/s] 34%|███▎      | 134607/400000 [00:18<00:36, 7293.52it/s] 34%|███▍      | 135343/400000 [00:18<00:36, 7312.12it/s] 34%|███▍      | 136075/400000 [00:18<00:36, 7239.66it/s] 34%|███▍      | 136813/400000 [00:19<00:36, 7280.76it/s] 34%|███▍      | 137542/400000 [00:19<00:36, 7281.07it/s] 35%|███▍      | 138295/400000 [00:19<00:35, 7352.99it/s] 35%|███▍      | 139045/400000 [00:19<00:35, 7395.53it/s] 35%|███▍      | 139792/400000 [00:19<00:35, 7417.14it/s] 35%|███▌      | 140536/400000 [00:19<00:34, 7423.35it/s] 35%|███▌      | 141294/400000 [00:19<00:34, 7468.33it/s] 36%|███▌      | 142048/400000 [00:19<00:34, 7487.03it/s] 36%|███▌      | 142797/400000 [00:19<00:34, 7487.64it/s] 36%|███▌      | 143546/400000 [00:19<00:34, 7409.64it/s] 36%|███▌      | 144288/400000 [00:20<00:34, 7363.30it/s] 36%|███▋      | 145045/400000 [00:20<00:34, 7423.26it/s] 36%|███▋      | 145794/400000 [00:20<00:34, 7442.93it/s] 37%|███▋      | 146553/400000 [00:20<00:33, 7485.05it/s] 37%|███▋      | 147302/400000 [00:20<00:33, 7477.81it/s] 37%|███▋      | 148051/400000 [00:20<00:33, 7479.47it/s] 37%|███▋      | 148800/400000 [00:20<00:33, 7477.24it/s] 37%|███▋      | 149550/400000 [00:20<00:33, 7482.99it/s] 38%|███▊      | 150306/400000 [00:20<00:33, 7505.90it/s] 38%|███▊      | 151057/400000 [00:20<00:33, 7486.16it/s] 38%|███▊      | 151806/400000 [00:21<00:34, 7211.05it/s] 38%|███▊      | 152530/400000 [00:21<00:35, 6966.83it/s] 38%|███▊      | 153231/400000 [00:21<00:35, 6878.31it/s] 38%|███▊      | 153922/400000 [00:21<00:36, 6808.12it/s] 39%|███▊      | 154605/400000 [00:21<00:36, 6747.33it/s] 39%|███▉      | 155336/400000 [00:21<00:35, 6905.95it/s] 39%|███▉      | 156074/400000 [00:21<00:34, 7041.36it/s] 39%|███▉      | 156781/400000 [00:21<00:34, 7047.65it/s] 39%|███▉      | 157499/400000 [00:21<00:34, 7084.41it/s] 40%|███▉      | 158248/400000 [00:21<00:33, 7200.05it/s] 40%|███▉      | 158993/400000 [00:22<00:33, 7271.70it/s] 40%|███▉      | 159722/400000 [00:22<00:33, 7106.24it/s] 40%|████      | 160463/400000 [00:22<00:33, 7192.64it/s] 40%|████      | 161208/400000 [00:22<00:32, 7267.41it/s] 40%|████      | 161949/400000 [00:22<00:32, 7309.06it/s] 41%|████      | 162701/400000 [00:22<00:32, 7368.55it/s] 41%|████      | 163445/400000 [00:22<00:32, 7387.97it/s] 41%|████      | 164185/400000 [00:22<00:32, 7283.28it/s] 41%|████      | 164915/400000 [00:22<00:33, 7075.50it/s] 41%|████▏     | 165625/400000 [00:23<00:34, 6891.33it/s] 42%|████▏     | 166317/400000 [00:23<00:34, 6791.10it/s] 42%|████▏     | 167063/400000 [00:23<00:33, 6977.21it/s] 42%|████▏     | 167815/400000 [00:23<00:32, 7130.09it/s] 42%|████▏     | 168568/400000 [00:23<00:31, 7243.46it/s] 42%|████▏     | 169295/400000 [00:23<00:32, 7191.16it/s] 43%|████▎     | 170016/400000 [00:23<00:32, 6985.46it/s] 43%|████▎     | 170718/400000 [00:23<00:33, 6847.30it/s] 43%|████▎     | 171406/400000 [00:23<00:33, 6800.26it/s] 43%|████▎     | 172112/400000 [00:23<00:33, 6874.11it/s] 43%|████▎     | 172833/400000 [00:24<00:32, 6970.91it/s] 43%|████▎     | 173547/400000 [00:24<00:32, 7019.55it/s] 44%|████▎     | 174263/400000 [00:24<00:31, 7059.18it/s] 44%|████▍     | 175021/400000 [00:24<00:31, 7206.41it/s] 44%|████▍     | 175766/400000 [00:24<00:30, 7275.91it/s] 44%|████▍     | 176512/400000 [00:24<00:30, 7330.17it/s] 44%|████▍     | 177259/400000 [00:24<00:30, 7369.77it/s] 44%|████▍     | 177997/400000 [00:24<00:30, 7262.65it/s] 45%|████▍     | 178740/400000 [00:24<00:30, 7307.41it/s] 45%|████▍     | 179486/400000 [00:24<00:29, 7351.69it/s] 45%|████▌     | 180223/400000 [00:25<00:29, 7354.45it/s] 45%|████▌     | 180959/400000 [00:25<00:30, 7296.42it/s] 45%|████▌     | 181709/400000 [00:25<00:29, 7355.15it/s] 46%|████▌     | 182453/400000 [00:25<00:29, 7379.75it/s] 46%|████▌     | 183205/400000 [00:25<00:29, 7419.96it/s] 46%|████▌     | 183948/400000 [00:25<00:29, 7412.69it/s] 46%|████▌     | 184692/400000 [00:25<00:29, 7419.02it/s] 46%|████▋     | 185442/400000 [00:25<00:28, 7443.01it/s] 47%|████▋     | 186187/400000 [00:25<00:28, 7435.76it/s] 47%|████▋     | 186936/400000 [00:25<00:28, 7450.77it/s] 47%|████▋     | 187682/400000 [00:26<00:28, 7417.97it/s] 47%|████▋     | 188424/400000 [00:26<00:28, 7393.04it/s] 47%|████▋     | 189172/400000 [00:26<00:28, 7413.58it/s] 47%|████▋     | 189921/400000 [00:26<00:28, 7435.28it/s] 48%|████▊     | 190665/400000 [00:26<00:29, 7179.46it/s] 48%|████▊     | 191385/400000 [00:26<00:29, 6985.32it/s] 48%|████▊     | 192087/400000 [00:26<00:30, 6908.31it/s] 48%|████▊     | 192833/400000 [00:26<00:29, 7063.96it/s] 48%|████▊     | 193577/400000 [00:26<00:28, 7170.42it/s] 49%|████▊     | 194316/400000 [00:26<00:28, 7234.11it/s] 49%|████▉     | 195053/400000 [00:27<00:28, 7269.51it/s] 49%|████▉     | 195782/400000 [00:27<00:28, 7223.39it/s] 49%|████▉     | 196526/400000 [00:27<00:27, 7286.03it/s] 49%|████▉     | 197265/400000 [00:27<00:27, 7314.32it/s] 50%|████▉     | 198009/400000 [00:27<00:27, 7349.72it/s] 50%|████▉     | 198748/400000 [00:27<00:27, 7360.39it/s] 50%|████▉     | 199485/400000 [00:27<00:27, 7350.76it/s] 50%|█████     | 200221/400000 [00:27<00:27, 7268.11it/s] 50%|█████     | 200949/400000 [00:27<00:28, 6879.18it/s] 50%|█████     | 201642/400000 [00:28<00:29, 6805.69it/s] 51%|█████     | 202326/400000 [00:28<00:29, 6738.86it/s] 51%|█████     | 203042/400000 [00:28<00:28, 6859.47it/s] 51%|█████     | 203782/400000 [00:28<00:27, 7010.53it/s] 51%|█████     | 204521/400000 [00:28<00:27, 7117.56it/s] 51%|█████▏    | 205257/400000 [00:28<00:27, 7187.62it/s] 51%|█████▏    | 205981/400000 [00:28<00:26, 7201.48it/s] 52%|█████▏    | 206721/400000 [00:28<00:26, 7258.89it/s] 52%|█████▏    | 207448/400000 [00:28<00:26, 7234.96it/s] 52%|█████▏    | 208173/400000 [00:28<00:26, 7179.96it/s] 52%|█████▏    | 208902/400000 [00:29<00:26, 7210.17it/s] 52%|█████▏    | 209632/400000 [00:29<00:26, 7235.92it/s] 53%|█████▎    | 210356/400000 [00:29<00:26, 7210.94it/s] 53%|█████▎    | 211085/400000 [00:29<00:26, 7232.14it/s] 53%|█████▎    | 211817/400000 [00:29<00:25, 7257.04it/s] 53%|█████▎    | 212555/400000 [00:29<00:25, 7291.86it/s] 53%|█████▎    | 213285/400000 [00:29<00:25, 7278.34it/s] 54%|█████▎    | 214018/400000 [00:29<00:25, 7291.58it/s] 54%|█████▎    | 214769/400000 [00:29<00:25, 7353.20it/s] 54%|█████▍    | 215505/400000 [00:29<00:26, 7040.34it/s] 54%|█████▍    | 216212/400000 [00:30<00:26, 6837.36it/s] 54%|█████▍    | 216900/400000 [00:30<00:27, 6767.28it/s] 54%|█████▍    | 217580/400000 [00:30<00:27, 6636.38it/s] 55%|█████▍    | 218326/400000 [00:30<00:26, 6861.86it/s] 55%|█████▍    | 219062/400000 [00:30<00:25, 7002.15it/s] 55%|█████▍    | 219797/400000 [00:30<00:25, 7101.65it/s] 55%|█████▌    | 220514/400000 [00:30<00:25, 7120.60it/s] 55%|█████▌    | 221257/400000 [00:30<00:24, 7209.50it/s] 55%|█████▌    | 221997/400000 [00:30<00:24, 7264.48it/s] 56%|█████▌    | 222729/400000 [00:30<00:24, 7279.30it/s] 56%|█████▌    | 223458/400000 [00:31<00:24, 7219.39it/s] 56%|█████▌    | 224195/400000 [00:31<00:24, 7263.04it/s] 56%|█████▌    | 224941/400000 [00:31<00:23, 7320.76it/s] 56%|█████▋    | 225693/400000 [00:31<00:23, 7377.44it/s] 57%|█████▋    | 226443/400000 [00:31<00:23, 7413.13it/s] 57%|█████▋    | 227195/400000 [00:31<00:23, 7442.89it/s] 57%|█████▋    | 227940/400000 [00:31<00:23, 7408.79it/s] 57%|█████▋    | 228689/400000 [00:31<00:23, 7431.63it/s] 57%|█████▋    | 229436/400000 [00:31<00:22, 7440.50it/s] 58%|█████▊    | 230181/400000 [00:31<00:22, 7422.99it/s] 58%|█████▊    | 230924/400000 [00:32<00:23, 7231.57it/s] 58%|█████▊    | 231649/400000 [00:32<00:23, 7153.98it/s] 58%|█████▊    | 232392/400000 [00:32<00:23, 7232.59it/s] 58%|█████▊    | 233144/400000 [00:32<00:22, 7316.15it/s] 58%|█████▊    | 233896/400000 [00:32<00:22, 7373.33it/s] 59%|█████▊    | 234645/400000 [00:32<00:22, 7407.67it/s] 59%|█████▉    | 235387/400000 [00:32<00:22, 7391.24it/s] 59%|█████▉    | 236127/400000 [00:32<00:22, 7361.94it/s] 59%|█████▉    | 236864/400000 [00:32<00:22, 7105.41it/s] 59%|█████▉    | 237577/400000 [00:32<00:22, 7066.10it/s] 60%|█████▉    | 238328/400000 [00:33<00:22, 7192.44it/s] 60%|█████▉    | 239049/400000 [00:33<00:22, 7037.17it/s] 60%|█████▉    | 239755/400000 [00:33<00:23, 6936.84it/s] 60%|██████    | 240498/400000 [00:33<00:22, 7076.12it/s] 60%|██████    | 241231/400000 [00:33<00:22, 7149.75it/s] 60%|██████    | 241958/400000 [00:33<00:22, 7183.72it/s] 61%|██████    | 242709/400000 [00:33<00:21, 7278.27it/s] 61%|██████    | 243440/400000 [00:33<00:21, 7287.30it/s] 61%|██████    | 244170/400000 [00:33<00:21, 7114.42it/s] 61%|██████    | 244911/400000 [00:34<00:21, 7191.32it/s] 61%|██████▏   | 245642/400000 [00:34<00:21, 7224.35it/s] 62%|██████▏   | 246366/400000 [00:34<00:21, 7196.59it/s] 62%|██████▏   | 247105/400000 [00:34<00:21, 7252.31it/s] 62%|██████▏   | 247856/400000 [00:34<00:20, 7325.07it/s] 62%|██████▏   | 248609/400000 [00:34<00:20, 7384.34it/s] 62%|██████▏   | 249363/400000 [00:34<00:20, 7428.74it/s] 63%|██████▎   | 250107/400000 [00:34<00:20, 7205.19it/s] 63%|██████▎   | 250830/400000 [00:34<00:21, 6975.95it/s] 63%|██████▎   | 251531/400000 [00:34<00:21, 6836.26it/s] 63%|██████▎   | 252218/400000 [00:35<00:21, 6764.79it/s] 63%|██████▎   | 252915/400000 [00:35<00:21, 6823.34it/s] 63%|██████▎   | 253635/400000 [00:35<00:21, 6930.40it/s] 64%|██████▎   | 254378/400000 [00:35<00:20, 7073.06it/s] 64%|██████▍   | 255120/400000 [00:35<00:20, 7171.66it/s] 64%|██████▍   | 255860/400000 [00:35<00:19, 7237.21it/s] 64%|██████▍   | 256592/400000 [00:35<00:19, 7260.57it/s] 64%|██████▍   | 257345/400000 [00:35<00:19, 7338.68it/s] 65%|██████▍   | 258096/400000 [00:35<00:19, 7387.64it/s] 65%|██████▍   | 258850/400000 [00:35<00:18, 7431.43it/s] 65%|██████▍   | 259597/400000 [00:36<00:18, 7440.80it/s] 65%|██████▌   | 260342/400000 [00:36<00:18, 7442.65it/s] 65%|██████▌   | 261088/400000 [00:36<00:18, 7446.16it/s] 65%|██████▌   | 261838/400000 [00:36<00:18, 7460.33it/s] 66%|██████▌   | 262585/400000 [00:36<00:18, 7458.35it/s] 66%|██████▌   | 263335/400000 [00:36<00:18, 7470.33it/s] 66%|██████▌   | 264083/400000 [00:36<00:18, 7446.98it/s] 66%|██████▌   | 264828/400000 [00:36<00:18, 7416.59it/s] 66%|██████▋   | 265580/400000 [00:36<00:18, 7446.37it/s] 67%|██████▋   | 266325/400000 [00:36<00:18, 7314.21it/s] 67%|██████▋   | 267058/400000 [00:37<00:18, 7116.32it/s] 67%|██████▋   | 267772/400000 [00:37<00:19, 6914.04it/s] 67%|██████▋   | 268499/400000 [00:37<00:18, 7015.14it/s] 67%|██████▋   | 269245/400000 [00:37<00:18, 7142.14it/s] 67%|██████▋   | 269998/400000 [00:37<00:17, 7251.52it/s] 68%|██████▊   | 270725/400000 [00:37<00:18, 6978.31it/s] 68%|██████▊   | 271427/400000 [00:37<00:18, 6797.82it/s] 68%|██████▊   | 272111/400000 [00:37<00:18, 6731.81it/s] 68%|██████▊   | 272797/400000 [00:37<00:18, 6768.29it/s] 68%|██████▊   | 273538/400000 [00:38<00:18, 6947.68it/s] 69%|██████▊   | 274284/400000 [00:38<00:17, 7092.68it/s] 69%|██████▉   | 275005/400000 [00:38<00:17, 7124.12it/s] 69%|██████▉   | 275727/400000 [00:38<00:17, 7151.65it/s] 69%|██████▉   | 276465/400000 [00:38<00:17, 7217.56it/s] 69%|██████▉   | 277206/400000 [00:38<00:16, 7273.39it/s] 69%|██████▉   | 277940/400000 [00:38<00:16, 7292.81it/s] 70%|██████▉   | 278670/400000 [00:38<00:16, 7283.18it/s] 70%|██████▉   | 279399/400000 [00:38<00:16, 7234.98it/s] 70%|███████   | 280123/400000 [00:38<00:16, 7129.62it/s] 70%|███████   | 280865/400000 [00:39<00:16, 7213.47it/s] 70%|███████   | 281600/400000 [00:39<00:16, 7252.27it/s] 71%|███████   | 282326/400000 [00:39<00:16, 7023.06it/s] 71%|███████   | 283031/400000 [00:39<00:16, 6903.70it/s] 71%|███████   | 283724/400000 [00:39<00:17, 6790.77it/s] 71%|███████   | 284405/400000 [00:39<00:17, 6556.38it/s] 71%|███████▏  | 285132/400000 [00:39<00:17, 6753.57it/s] 71%|███████▏  | 285859/400000 [00:39<00:16, 6898.04it/s] 72%|███████▏  | 286567/400000 [00:39<00:16, 6880.41it/s] 72%|███████▏  | 287258/400000 [00:39<00:16, 6744.26it/s] 72%|███████▏  | 287935/400000 [00:40<00:16, 6677.40it/s] 72%|███████▏  | 288605/400000 [00:40<00:16, 6600.95it/s] 72%|███████▏  | 289267/400000 [00:40<00:17, 6489.72it/s] 72%|███████▏  | 289921/400000 [00:40<00:16, 6502.21it/s] 73%|███████▎  | 290578/400000 [00:40<00:16, 6520.09it/s] 73%|███████▎  | 291319/400000 [00:40<00:16, 6761.96it/s] 73%|███████▎  | 292062/400000 [00:40<00:15, 6947.53it/s] 73%|███████▎  | 292768/400000 [00:40<00:15, 6979.55it/s] 73%|███████▎  | 293514/400000 [00:40<00:14, 7115.80it/s] 74%|███████▎  | 294250/400000 [00:40<00:14, 7185.97it/s] 74%|███████▎  | 294983/400000 [00:41<00:14, 7228.25it/s] 74%|███████▍  | 295708/400000 [00:41<00:14, 7002.11it/s] 74%|███████▍  | 296411/400000 [00:41<00:15, 6869.80it/s] 74%|███████▍  | 297101/400000 [00:41<00:15, 6744.54it/s] 74%|███████▍  | 297778/400000 [00:41<00:15, 6738.47it/s] 75%|███████▍  | 298515/400000 [00:41<00:14, 6914.62it/s] 75%|███████▍  | 299253/400000 [00:41<00:14, 7045.59it/s] 75%|███████▍  | 299969/400000 [00:41<00:14, 7078.20it/s] 75%|███████▌  | 300679/400000 [00:41<00:14, 6975.25it/s] 75%|███████▌  | 301378/400000 [00:42<00:14, 6861.37it/s] 76%|███████▌  | 302066/400000 [00:42<00:14, 6774.88it/s] 76%|███████▌  | 302745/400000 [00:42<00:14, 6751.16it/s] 76%|███████▌  | 303421/400000 [00:42<00:14, 6706.91it/s] 76%|███████▌  | 304126/400000 [00:42<00:14, 6805.70it/s] 76%|███████▌  | 304877/400000 [00:42<00:13, 7001.58it/s] 76%|███████▋  | 305607/400000 [00:42<00:13, 7088.48it/s] 77%|███████▋  | 306352/400000 [00:42<00:13, 7192.02it/s] 77%|███████▋  | 307085/400000 [00:42<00:12, 7230.60it/s] 77%|███████▋  | 307827/400000 [00:42<00:12, 7285.66it/s] 77%|███████▋  | 308573/400000 [00:43<00:12, 7335.08it/s] 77%|███████▋  | 309323/400000 [00:43<00:12, 7381.74it/s] 78%|███████▊  | 310062/400000 [00:43<00:12, 7365.24it/s] 78%|███████▊  | 310799/400000 [00:43<00:12, 7366.36it/s] 78%|███████▊  | 311536/400000 [00:43<00:12, 7320.06it/s] 78%|███████▊  | 312269/400000 [00:43<00:12, 7298.19it/s] 78%|███████▊  | 313020/400000 [00:43<00:11, 7357.99it/s] 78%|███████▊  | 313761/400000 [00:43<00:11, 7373.27it/s] 79%|███████▊  | 314509/400000 [00:43<00:11, 7402.11it/s] 79%|███████▉  | 315250/400000 [00:43<00:11, 7393.74it/s] 79%|███████▉  | 315996/400000 [00:44<00:11, 7412.46it/s] 79%|███████▉  | 316743/400000 [00:44<00:11, 7427.76it/s] 79%|███████▉  | 317488/400000 [00:44<00:11, 7433.71it/s] 80%|███████▉  | 318232/400000 [00:44<00:11, 7415.06it/s] 80%|███████▉  | 318974/400000 [00:44<00:11, 7341.84it/s] 80%|███████▉  | 319709/400000 [00:44<00:11, 7055.38it/s] 80%|████████  | 320418/400000 [00:44<00:11, 6908.07it/s] 80%|████████  | 321112/400000 [00:44<00:11, 6805.45it/s] 80%|████████  | 321849/400000 [00:44<00:11, 6963.56it/s] 81%|████████  | 322598/400000 [00:44<00:10, 7111.19it/s] 81%|████████  | 323312/400000 [00:45<00:10, 7039.07it/s] 81%|████████  | 324057/400000 [00:45<00:10, 7155.63it/s] 81%|████████  | 324797/400000 [00:45<00:10, 7225.43it/s] 81%|████████▏ | 325540/400000 [00:45<00:10, 7282.44it/s] 82%|████████▏ | 326285/400000 [00:45<00:10, 7329.81it/s] 82%|████████▏ | 327019/400000 [00:45<00:10, 7187.50it/s] 82%|████████▏ | 327739/400000 [00:45<00:10, 7176.88it/s] 82%|████████▏ | 328458/400000 [00:45<00:10, 7080.89it/s] 82%|████████▏ | 329175/400000 [00:45<00:09, 7105.30it/s] 82%|████████▏ | 329921/400000 [00:45<00:09, 7205.45it/s] 83%|████████▎ | 330669/400000 [00:46<00:09, 7283.29it/s] 83%|████████▎ | 331409/400000 [00:46<00:09, 7316.25it/s] 83%|████████▎ | 332161/400000 [00:46<00:09, 7373.96it/s] 83%|████████▎ | 332903/400000 [00:46<00:09, 7385.85it/s] 83%|████████▎ | 333647/400000 [00:46<00:08, 7400.60it/s] 84%|████████▎ | 334388/400000 [00:46<00:09, 7252.98it/s] 84%|████████▍ | 335132/400000 [00:46<00:08, 7305.20it/s] 84%|████████▍ | 335864/400000 [00:46<00:08, 7287.92it/s] 84%|████████▍ | 336605/400000 [00:46<00:08, 7323.94it/s] 84%|████████▍ | 337344/400000 [00:46<00:08, 7341.88it/s] 85%|████████▍ | 338088/400000 [00:47<00:08, 7369.22it/s] 85%|████████▍ | 338837/400000 [00:47<00:08, 7404.27it/s] 85%|████████▍ | 339578/400000 [00:47<00:08, 7378.87it/s] 85%|████████▌ | 340326/400000 [00:47<00:08, 7407.39it/s] 85%|████████▌ | 341067/400000 [00:47<00:07, 7377.80it/s] 85%|████████▌ | 341805/400000 [00:47<00:08, 7134.09it/s] 86%|████████▌ | 342553/400000 [00:47<00:07, 7233.15it/s] 86%|████████▌ | 343278/400000 [00:47<00:07, 7185.40it/s] 86%|████████▌ | 344022/400000 [00:47<00:07, 7257.45it/s] 86%|████████▌ | 344749/400000 [00:48<00:07, 7142.63it/s] 86%|████████▋ | 345465/400000 [00:48<00:07, 6984.79it/s] 87%|████████▋ | 346166/400000 [00:48<00:07, 6883.78it/s] 87%|████████▋ | 346856/400000 [00:48<00:07, 6802.11it/s] 87%|████████▋ | 347539/400000 [00:48<00:07, 6808.48it/s] 87%|████████▋ | 348221/400000 [00:48<00:07, 6664.83it/s] 87%|████████▋ | 348961/400000 [00:48<00:07, 6867.62it/s] 87%|████████▋ | 349651/400000 [00:48<00:07, 6814.31it/s] 88%|████████▊ | 350335/400000 [00:48<00:07, 6758.96it/s] 88%|████████▊ | 351053/400000 [00:48<00:07, 6877.45it/s] 88%|████████▊ | 351786/400000 [00:49<00:06, 7006.34it/s] 88%|████████▊ | 352533/400000 [00:49<00:06, 7137.61it/s] 88%|████████▊ | 353271/400000 [00:49<00:06, 7206.21it/s] 88%|████████▊ | 353997/400000 [00:49<00:06, 7219.85it/s] 89%|████████▊ | 354731/400000 [00:49<00:06, 7255.12it/s] 89%|████████▉ | 355471/400000 [00:49<00:06, 7296.64it/s] 89%|████████▉ | 356218/400000 [00:49<00:05, 7345.05it/s] 89%|████████▉ | 356961/400000 [00:49<00:05, 7364.99it/s] 89%|████████▉ | 357704/400000 [00:49<00:05, 7381.73it/s] 90%|████████▉ | 358443/400000 [00:49<00:05, 7347.53it/s] 90%|████████▉ | 359178/400000 [00:50<00:05, 7341.28it/s] 90%|████████▉ | 359913/400000 [00:50<00:05, 7341.70it/s] 90%|█████████ | 360648/400000 [00:50<00:05, 7341.63it/s] 90%|█████████ | 361390/400000 [00:50<00:05, 7363.82it/s] 91%|█████████ | 362127/400000 [00:50<00:05, 7291.33it/s] 91%|█████████ | 362857/400000 [00:50<00:05, 7264.88it/s] 91%|█████████ | 363600/400000 [00:50<00:04, 7310.93it/s] 91%|█████████ | 364347/400000 [00:50<00:04, 7357.61it/s] 91%|█████████▏| 365093/400000 [00:50<00:04, 7385.69it/s] 91%|█████████▏| 365832/400000 [00:50<00:04, 7184.44it/s] 92%|█████████▏| 366552/400000 [00:51<00:04, 7030.46it/s] 92%|█████████▏| 367257/400000 [00:51<00:04, 6927.37it/s] 92%|█████████▏| 368002/400000 [00:51<00:04, 7075.70it/s] 92%|█████████▏| 368751/400000 [00:51<00:04, 7193.70it/s] 92%|█████████▏| 369492/400000 [00:51<00:04, 7256.31it/s] 93%|█████████▎| 370240/400000 [00:51<00:04, 7320.31it/s] 93%|█████████▎| 370986/400000 [00:51<00:03, 7360.08it/s] 93%|█████████▎| 371728/400000 [00:51<00:03, 7377.47it/s] 93%|█████████▎| 372470/400000 [00:51<00:03, 7383.68it/s] 93%|█████████▎| 373209/400000 [00:51<00:03, 7196.27it/s] 93%|█████████▎| 373930/400000 [00:52<00:03, 6873.96it/s] 94%|█████████▎| 374658/400000 [00:52<00:03, 6989.08it/s] 94%|█████████▍| 375406/400000 [00:52<00:03, 7128.62it/s] 94%|█████████▍| 376150/400000 [00:52<00:03, 7219.18it/s] 94%|█████████▍| 376883/400000 [00:52<00:03, 7250.00it/s] 94%|█████████▍| 377637/400000 [00:52<00:03, 7332.77it/s] 95%|█████████▍| 378378/400000 [00:52<00:02, 7355.50it/s] 95%|█████████▍| 379128/400000 [00:52<00:02, 7395.63it/s] 95%|█████████▍| 379873/400000 [00:52<00:02, 7409.43it/s] 95%|█████████▌| 380615/400000 [00:53<00:02, 7265.61it/s] 95%|█████████▌| 381343/400000 [00:53<00:02, 7025.68it/s] 96%|█████████▌| 382060/400000 [00:53<00:02, 7068.13it/s] 96%|█████████▌| 382799/400000 [00:53<00:02, 7159.51it/s] 96%|█████████▌| 383548/400000 [00:53<00:02, 7254.43it/s] 96%|█████████▌| 384275/400000 [00:53<00:02, 7254.18it/s] 96%|█████████▋| 385005/400000 [00:53<00:02, 7265.88it/s] 96%|█████████▋| 385751/400000 [00:53<00:01, 7321.68it/s] 97%|█████████▋| 386500/400000 [00:53<00:01, 7370.89it/s] 97%|█████████▋| 387246/400000 [00:53<00:01, 7395.77it/s] 97%|█████████▋| 387986/400000 [00:54<00:01, 7340.83it/s] 97%|█████████▋| 388721/400000 [00:54<00:01, 7319.96it/s] 97%|█████████▋| 389457/400000 [00:54<00:01, 7331.83it/s] 98%|█████████▊| 390191/400000 [00:54<00:01, 7314.25it/s] 98%|█████████▊| 390923/400000 [00:54<00:01, 7285.64it/s] 98%|█████████▊| 391652/400000 [00:54<00:01, 7221.31it/s] 98%|█████████▊| 392390/400000 [00:54<00:01, 7266.77it/s] 98%|█████████▊| 393117/400000 [00:54<00:00, 7250.86it/s] 98%|█████████▊| 393859/400000 [00:54<00:00, 7299.52it/s] 99%|█████████▊| 394597/400000 [00:54<00:00, 7321.74it/s] 99%|█████████▉| 395330/400000 [00:55<00:00, 7285.97it/s] 99%|█████████▉| 396059/400000 [00:55<00:00, 7259.41it/s] 99%|█████████▉| 396786/400000 [00:55<00:00, 7212.73it/s] 99%|█████████▉| 397522/400000 [00:55<00:00, 7256.13it/s]100%|█████████▉| 398266/400000 [00:55<00:00, 7308.99it/s]100%|█████████▉| 399010/400000 [00:55<00:00, 7346.56it/s]100%|█████████▉| 399753/400000 [00:55<00:00, 7368.04it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7186.01it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3155a5b4a8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010953020128569378 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011470482700245834 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15667 out of table with 15527 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15667 out of table with 15527 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

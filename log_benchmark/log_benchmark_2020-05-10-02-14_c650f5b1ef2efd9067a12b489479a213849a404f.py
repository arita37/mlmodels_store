
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f690a297f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 02:14:15.542952
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 02:14:15.546668
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 02:14:15.549775
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 02:14:15.552932
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f691176e080> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355241.0000
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 266245.4688
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 167936.6094
Epoch 4/10

1/1 [==============================] - 0s 89ms/step - loss: 93800.9531
Epoch 5/10

1/1 [==============================] - 0s 91ms/step - loss: 50973.6367
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 29829.4883
Epoch 7/10

1/1 [==============================] - 0s 94ms/step - loss: 19085.9824
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 13111.8457
Epoch 9/10

1/1 [==============================] - 0s 90ms/step - loss: 9547.3525
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 7282.1792

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.16121048  7.5805774   8.929966    6.246202    8.457623    6.664016
   7.7306437   9.618616    7.469315    7.881277    7.367315    6.964972
   7.918744    8.648971    9.104103    7.6938753   7.3659387   7.258792
   5.7382207   8.425872    7.8944473   8.637125    7.0622745   5.1321964
   6.796814    7.274242    6.9921684   6.3041587   7.4103856   8.104833
   8.467991    7.313403    7.43754     7.9750752   7.2138433   6.8963003
   8.557608    8.693869    6.6574316   9.122929    6.014683    6.3548713
   6.2207794   5.9059153   7.619307    7.487407    8.514314    8.343933
   8.101953    8.205903    6.432581    9.072542    8.290042    8.31073
   7.4257474   6.455926    8.2191725   6.929896    7.3044796   9.318038
   0.15845045 -0.49933392  1.6127608  -0.16795844  1.4892986  -0.22894502
   0.03067066  2.4886909   1.9141774   1.7834159   0.62251866  1.1139243
  -0.08819714 -0.03124112  2.1290092   1.5543013  -1.0303198  -0.77936864
  -1.718705    0.53200626  1.1559765   1.333941    2.474833    0.7569438
  -0.6498264  -0.5648259   0.16488546  0.15571958  0.03784335 -0.4612944
  -0.3956739  -0.6543927   0.074697   -1.2120117   0.40067226 -0.2597447
   1.3768159   0.903291    0.318147    1.2953061  -0.6879243   0.36727017
   1.4844851  -0.96801484 -2.538293   -1.4533391  -0.19907054 -0.04338616
  -0.69380295  0.09896481  0.13770369  0.16723806  0.90022975 -1.0517614
  -1.804436   -0.28632915 -0.5384566   1.5263809   0.9151204   0.41327623
  -0.06750011 -0.6928065   1.3259333  -0.9176793   0.88709104 -1.3659022
   0.3799258  -1.2652777   0.53243184  1.2647331   1.4792498   0.9604161
   1.0626725  -0.15428689  1.6015121   1.1189072  -1.1229253  -2.1959202
  -1.2720389  -1.1698703   1.3193839  -0.3734249  -0.16200382  0.9638665
   1.1776825   0.73338956  0.958189   -1.4372003  -1.1985668  -0.38892448
  -0.3942046   0.44933084 -0.80848414 -0.64250934 -1.7040068   1.2457608
  -0.7431692   0.16461673  0.751698   -2.0961268  -0.7364861   0.6009623
  -0.6867773  -2.194802   -0.88235027 -0.8436344  -0.8797207  -0.8742373
   0.82706535 -0.39625368  1.1893941  -0.26584625  0.53186584  1.6184899
   0.6154271   0.52626896  1.5552309  -1.0329862  -0.06032488 -1.3323026
   0.12027961  8.766821    8.187086   10.015677    8.7853      7.6879053
   6.6560497   9.302856    7.747421    8.292765    8.253683    7.6374955
   8.67728     8.286888    8.066209    8.5506115   6.7650585   8.5703
   9.245517    7.9520764   8.42212     7.600042   10.100586    7.4444165
   8.897595    9.045943    8.36        7.395807    8.078005    7.919046
   7.7610884   8.426002    7.7530675   7.298215    8.128127    9.170567
   7.8011146   6.6377187   8.15197     7.6832323   6.9843235   7.550214
   6.902079    6.7710185   8.663939   10.1742      8.723057    6.8352323
   6.42539     7.678629    8.381842    6.5212226   7.9717717   6.8832045
   7.390433    7.2912145   7.842643    7.4269233   8.027477    7.354676
   0.58112085  0.89560163  0.276636    1.8482027   0.44239873  0.94025904
   0.2876233   0.38759053  2.9794044   0.82659096  1.1183187   0.5275165
   2.3557136   0.7329916   0.24048638  0.24520195  3.7668839   0.15054774
   1.6151876   2.201877    1.4055831   0.5517254   1.9597875   0.12236589
   0.6684269   0.80552226  0.47672093  1.4930018   2.3185935   0.33863115
   1.9133108   1.3606349   1.3266222   0.23491263  2.0154657   0.40667307
   1.344758    0.80118513  3.0613914   0.76831436  0.5783288   1.9947624
   1.2877262   0.35589647  2.6026125   2.9193487   1.7070069   0.7210144
   0.5336739   1.9111531   1.918977    0.31880683  1.7520615   1.1234169
   0.29747784  0.2583369   1.7058253   0.9324273   0.08809596  2.320343
   2.0839324   0.79976934  0.29378045  1.1728095   0.76811624  1.5754399
   1.3637881   0.13089556  2.1342282   0.22978866  0.25042927  1.4682363
   1.1199517   0.12111974  0.5798715   1.6458781   0.51556665  1.6089418
   0.8865241   0.48732907  0.68838495  0.8487767   2.907814    3.087616
   2.8632689   1.5773078   0.52923775  0.71801984  0.43955624  2.2882223
   1.4770014   0.45806456  1.5951662   1.5803478   0.33025014  1.0638639
   1.2709408   1.1093664   0.2770915   0.20127177  1.5000131   0.24709356
   0.24670613  1.5317676   1.0761365   1.0511489   0.27136862  2.3998368
   0.13786197  0.36521268  0.5139308   0.5044401   2.6464424   0.9129271
   0.65248036  0.4315502   1.5139532   1.4080466   1.4063175   0.5720912
   5.9124866  -8.19817    -0.38738716]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 02:14:26.029700
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.1996
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 02:14:26.033426
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8893.21
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 02:14:26.036702
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6337
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 02:14:26.039780
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -795.449
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140088776819768
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140088086442784
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140088086443288
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140088086443792
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140088086444296
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140088086444800

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f690a297a90> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.595493
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.554888
grad_step = 000002, loss = 0.522187
grad_step = 000003, loss = 0.485128
grad_step = 000004, loss = 0.441257
grad_step = 000005, loss = 0.392639
grad_step = 000006, loss = 0.350076
grad_step = 000007, loss = 0.336087
grad_step = 000008, loss = 0.321330
grad_step = 000009, loss = 0.301158
grad_step = 000010, loss = 0.279397
grad_step = 000011, loss = 0.255978
grad_step = 000012, loss = 0.238562
grad_step = 000013, loss = 0.228009
grad_step = 000014, loss = 0.219741
grad_step = 000015, loss = 0.209519
grad_step = 000016, loss = 0.196534
grad_step = 000017, loss = 0.183153
grad_step = 000018, loss = 0.172261
grad_step = 000019, loss = 0.164322
grad_step = 000020, loss = 0.154900
grad_step = 000021, loss = 0.144410
grad_step = 000022, loss = 0.135304
grad_step = 000023, loss = 0.126358
grad_step = 000024, loss = 0.117973
grad_step = 000025, loss = 0.110598
grad_step = 000026, loss = 0.103262
grad_step = 000027, loss = 0.095537
grad_step = 000028, loss = 0.088157
grad_step = 000029, loss = 0.082038
grad_step = 000030, loss = 0.076588
grad_step = 000031, loss = 0.070522
grad_step = 000032, loss = 0.064390
grad_step = 000033, loss = 0.059282
grad_step = 000034, loss = 0.054935
grad_step = 000035, loss = 0.050712
grad_step = 000036, loss = 0.046575
grad_step = 000037, loss = 0.042600
grad_step = 000038, loss = 0.038745
grad_step = 000039, loss = 0.035293
grad_step = 000040, loss = 0.032369
grad_step = 000041, loss = 0.029536
grad_step = 000042, loss = 0.026798
grad_step = 000043, loss = 0.024465
grad_step = 000044, loss = 0.022322
grad_step = 000045, loss = 0.020217
grad_step = 000046, loss = 0.018334
grad_step = 000047, loss = 0.016753
grad_step = 000048, loss = 0.015236
grad_step = 000049, loss = 0.013796
grad_step = 000050, loss = 0.012554
grad_step = 000051, loss = 0.011410
grad_step = 000052, loss = 0.010356
grad_step = 000053, loss = 0.009381
grad_step = 000054, loss = 0.008470
grad_step = 000055, loss = 0.007732
grad_step = 000056, loss = 0.007124
grad_step = 000057, loss = 0.006502
grad_step = 000058, loss = 0.005930
grad_step = 000059, loss = 0.005489
grad_step = 000060, loss = 0.005083
grad_step = 000061, loss = 0.004691
grad_step = 000062, loss = 0.004357
grad_step = 000063, loss = 0.004073
grad_step = 000064, loss = 0.003839
grad_step = 000065, loss = 0.003613
grad_step = 000066, loss = 0.003391
grad_step = 000067, loss = 0.003238
grad_step = 000068, loss = 0.003118
grad_step = 000069, loss = 0.002988
grad_step = 000070, loss = 0.002877
grad_step = 000071, loss = 0.002785
grad_step = 000072, loss = 0.002707
grad_step = 000073, loss = 0.002643
grad_step = 000074, loss = 0.002583
grad_step = 000075, loss = 0.002533
grad_step = 000076, loss = 0.002494
grad_step = 000077, loss = 0.002453
grad_step = 000078, loss = 0.002423
grad_step = 000079, loss = 0.002401
grad_step = 000080, loss = 0.002375
grad_step = 000081, loss = 0.002358
grad_step = 000082, loss = 0.002343
grad_step = 000083, loss = 0.002329
grad_step = 000084, loss = 0.002319
grad_step = 000085, loss = 0.002308
grad_step = 000086, loss = 0.002299
grad_step = 000087, loss = 0.002292
grad_step = 000088, loss = 0.002285
grad_step = 000089, loss = 0.002282
grad_step = 000090, loss = 0.002275
grad_step = 000091, loss = 0.002267
grad_step = 000092, loss = 0.002262
grad_step = 000093, loss = 0.002258
grad_step = 000094, loss = 0.002253
grad_step = 000095, loss = 0.002247
grad_step = 000096, loss = 0.002241
grad_step = 000097, loss = 0.002237
grad_step = 000098, loss = 0.002231
grad_step = 000099, loss = 0.002226
grad_step = 000100, loss = 0.002222
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002217
grad_step = 000102, loss = 0.002212
grad_step = 000103, loss = 0.002207
grad_step = 000104, loss = 0.002203
grad_step = 000105, loss = 0.002199
grad_step = 000106, loss = 0.002195
grad_step = 000107, loss = 0.002192
grad_step = 000108, loss = 0.002188
grad_step = 000109, loss = 0.002185
grad_step = 000110, loss = 0.002182
grad_step = 000111, loss = 0.002179
grad_step = 000112, loss = 0.002176
grad_step = 000113, loss = 0.002173
grad_step = 000114, loss = 0.002170
grad_step = 000115, loss = 0.002168
grad_step = 000116, loss = 0.002165
grad_step = 000117, loss = 0.002163
grad_step = 000118, loss = 0.002160
grad_step = 000119, loss = 0.002158
grad_step = 000120, loss = 0.002156
grad_step = 000121, loss = 0.002154
grad_step = 000122, loss = 0.002152
grad_step = 000123, loss = 0.002150
grad_step = 000124, loss = 0.002148
grad_step = 000125, loss = 0.002145
grad_step = 000126, loss = 0.002144
grad_step = 000127, loss = 0.002142
grad_step = 000128, loss = 0.002140
grad_step = 000129, loss = 0.002138
grad_step = 000130, loss = 0.002136
grad_step = 000131, loss = 0.002134
grad_step = 000132, loss = 0.002132
grad_step = 000133, loss = 0.002130
grad_step = 000134, loss = 0.002129
grad_step = 000135, loss = 0.002127
grad_step = 000136, loss = 0.002125
grad_step = 000137, loss = 0.002123
grad_step = 000138, loss = 0.002121
grad_step = 000139, loss = 0.002119
grad_step = 000140, loss = 0.002118
grad_step = 000141, loss = 0.002116
grad_step = 000142, loss = 0.002114
grad_step = 000143, loss = 0.002112
grad_step = 000144, loss = 0.002110
grad_step = 000145, loss = 0.002108
grad_step = 000146, loss = 0.002107
grad_step = 000147, loss = 0.002105
grad_step = 000148, loss = 0.002103
grad_step = 000149, loss = 0.002101
grad_step = 000150, loss = 0.002099
grad_step = 000151, loss = 0.002097
grad_step = 000152, loss = 0.002095
grad_step = 000153, loss = 0.002093
grad_step = 000154, loss = 0.002091
grad_step = 000155, loss = 0.002089
grad_step = 000156, loss = 0.002087
grad_step = 000157, loss = 0.002085
grad_step = 000158, loss = 0.002083
grad_step = 000159, loss = 0.002081
grad_step = 000160, loss = 0.002079
grad_step = 000161, loss = 0.002077
grad_step = 000162, loss = 0.002075
grad_step = 000163, loss = 0.002073
grad_step = 000164, loss = 0.002071
grad_step = 000165, loss = 0.002069
grad_step = 000166, loss = 0.002066
grad_step = 000167, loss = 0.002064
grad_step = 000168, loss = 0.002062
grad_step = 000169, loss = 0.002060
grad_step = 000170, loss = 0.002057
grad_step = 000171, loss = 0.002055
grad_step = 000172, loss = 0.002053
grad_step = 000173, loss = 0.002050
grad_step = 000174, loss = 0.002047
grad_step = 000175, loss = 0.002045
grad_step = 000176, loss = 0.002042
grad_step = 000177, loss = 0.002039
grad_step = 000178, loss = 0.002036
grad_step = 000179, loss = 0.002033
grad_step = 000180, loss = 0.002030
grad_step = 000181, loss = 0.002027
grad_step = 000182, loss = 0.002024
grad_step = 000183, loss = 0.002020
grad_step = 000184, loss = 0.002017
grad_step = 000185, loss = 0.002015
grad_step = 000186, loss = 0.002010
grad_step = 000187, loss = 0.002007
grad_step = 000188, loss = 0.002005
grad_step = 000189, loss = 0.001999
grad_step = 000190, loss = 0.001993
grad_step = 000191, loss = 0.001990
grad_step = 000192, loss = 0.001982
grad_step = 000193, loss = 0.001979
grad_step = 000194, loss = 0.001970
grad_step = 000195, loss = 0.001966
grad_step = 000196, loss = 0.001960
grad_step = 000197, loss = 0.001953
grad_step = 000198, loss = 0.001938
grad_step = 000199, loss = 0.001933
grad_step = 000200, loss = 0.001924
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001951
grad_step = 000202, loss = 0.002024
grad_step = 000203, loss = 0.001965
grad_step = 000204, loss = 0.001920
grad_step = 000205, loss = 0.001875
grad_step = 000206, loss = 0.001967
grad_step = 000207, loss = 0.002062
grad_step = 000208, loss = 0.002060
grad_step = 000209, loss = 0.001865
grad_step = 000210, loss = 0.001844
grad_step = 000211, loss = 0.002009
grad_step = 000212, loss = 0.002177
grad_step = 000213, loss = 0.001909
grad_step = 000214, loss = 0.001999
grad_step = 000215, loss = 0.002245
grad_step = 000216, loss = 0.001890
grad_step = 000217, loss = 0.001851
grad_step = 000218, loss = 0.001794
grad_step = 000219, loss = 0.002004
grad_step = 000220, loss = 0.001942
grad_step = 000221, loss = 0.001793
grad_step = 000222, loss = 0.001835
grad_step = 000223, loss = 0.001783
grad_step = 000224, loss = 0.002095
grad_step = 000225, loss = 0.002368
grad_step = 000226, loss = 0.002063
grad_step = 000227, loss = 0.001861
grad_step = 000228, loss = 0.001986
grad_step = 000229, loss = 0.002055
grad_step = 000230, loss = 0.001918
grad_step = 000231, loss = 0.001850
grad_step = 000232, loss = 0.001910
grad_step = 000233, loss = 0.001897
grad_step = 000234, loss = 0.001796
grad_step = 000235, loss = 0.001675
grad_step = 000236, loss = 0.001781
grad_step = 000237, loss = 0.001768
grad_step = 000238, loss = 0.001799
grad_step = 000239, loss = 0.001827
grad_step = 000240, loss = 0.001700
grad_step = 000241, loss = 0.001695
grad_step = 000242, loss = 0.001690
grad_step = 000243, loss = 0.001749
grad_step = 000244, loss = 0.001936
grad_step = 000245, loss = 0.002016
grad_step = 000246, loss = 0.001850
grad_step = 000247, loss = 0.001708
grad_step = 000248, loss = 0.001677
grad_step = 000249, loss = 0.001827
grad_step = 000250, loss = 0.001851
grad_step = 000251, loss = 0.001698
grad_step = 000252, loss = 0.001750
grad_step = 000253, loss = 0.001844
grad_step = 000254, loss = 0.001707
grad_step = 000255, loss = 0.001654
grad_step = 000256, loss = 0.001765
grad_step = 000257, loss = 0.001737
grad_step = 000258, loss = 0.001647
grad_step = 000259, loss = 0.001652
grad_step = 000260, loss = 0.001699
grad_step = 000261, loss = 0.001705
grad_step = 000262, loss = 0.001651
grad_step = 000263, loss = 0.001630
grad_step = 000264, loss = 0.001679
grad_step = 000265, loss = 0.001677
grad_step = 000266, loss = 0.001644
grad_step = 000267, loss = 0.001626
grad_step = 000268, loss = 0.001635
grad_step = 000269, loss = 0.001659
grad_step = 000270, loss = 0.001647
grad_step = 000271, loss = 0.001625
grad_step = 000272, loss = 0.001613
grad_step = 000273, loss = 0.001612
grad_step = 000274, loss = 0.001628
grad_step = 000275, loss = 0.001639
grad_step = 000276, loss = 0.001647
grad_step = 000277, loss = 0.001648
grad_step = 000278, loss = 0.001640
grad_step = 000279, loss = 0.001638
grad_step = 000280, loss = 0.001630
grad_step = 000281, loss = 0.001626
grad_step = 000282, loss = 0.001624
grad_step = 000283, loss = 0.001623
grad_step = 000284, loss = 0.001625
grad_step = 000285, loss = 0.001631
grad_step = 000286, loss = 0.001638
grad_step = 000287, loss = 0.001650
grad_step = 000288, loss = 0.001655
grad_step = 000289, loss = 0.001662
grad_step = 000290, loss = 0.001645
grad_step = 000291, loss = 0.001628
grad_step = 000292, loss = 0.001601
grad_step = 000293, loss = 0.001583
grad_step = 000294, loss = 0.001583
grad_step = 000295, loss = 0.001594
grad_step = 000296, loss = 0.001613
grad_step = 000297, loss = 0.001629
grad_step = 000298, loss = 0.001640
grad_step = 000299, loss = 0.001642
grad_step = 000300, loss = 0.001631
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001616
grad_step = 000302, loss = 0.001597
grad_step = 000303, loss = 0.001583
grad_step = 000304, loss = 0.001573
grad_step = 000305, loss = 0.001566
grad_step = 000306, loss = 0.001564
grad_step = 000307, loss = 0.001565
grad_step = 000308, loss = 0.001568
grad_step = 000309, loss = 0.001572
grad_step = 000310, loss = 0.001579
grad_step = 000311, loss = 0.001591
grad_step = 000312, loss = 0.001606
grad_step = 000313, loss = 0.001628
grad_step = 000314, loss = 0.001655
grad_step = 000315, loss = 0.001676
grad_step = 000316, loss = 0.001681
grad_step = 000317, loss = 0.001658
grad_step = 000318, loss = 0.001612
grad_step = 000319, loss = 0.001569
grad_step = 000320, loss = 0.001546
grad_step = 000321, loss = 0.001552
grad_step = 000322, loss = 0.001570
grad_step = 000323, loss = 0.001580
grad_step = 000324, loss = 0.001573
grad_step = 000325, loss = 0.001553
grad_step = 000326, loss = 0.001538
grad_step = 000327, loss = 0.001539
grad_step = 000328, loss = 0.001549
grad_step = 000329, loss = 0.001555
grad_step = 000330, loss = 0.001551
grad_step = 000331, loss = 0.001539
grad_step = 000332, loss = 0.001528
grad_step = 000333, loss = 0.001524
grad_step = 000334, loss = 0.001525
grad_step = 000335, loss = 0.001531
grad_step = 000336, loss = 0.001538
grad_step = 000337, loss = 0.001551
grad_step = 000338, loss = 0.001573
grad_step = 000339, loss = 0.001612
grad_step = 000340, loss = 0.001666
grad_step = 000341, loss = 0.001734
grad_step = 000342, loss = 0.001758
grad_step = 000343, loss = 0.001730
grad_step = 000344, loss = 0.001617
grad_step = 000345, loss = 0.001519
grad_step = 000346, loss = 0.001516
grad_step = 000347, loss = 0.001578
grad_step = 000348, loss = 0.001600
grad_step = 000349, loss = 0.001545
grad_step = 000350, loss = 0.001499
grad_step = 000351, loss = 0.001521
grad_step = 000352, loss = 0.001551
grad_step = 000353, loss = 0.001526
grad_step = 000354, loss = 0.001490
grad_step = 000355, loss = 0.001499
grad_step = 000356, loss = 0.001523
grad_step = 000357, loss = 0.001513
grad_step = 000358, loss = 0.001485
grad_step = 000359, loss = 0.001474
grad_step = 000360, loss = 0.001488
grad_step = 000361, loss = 0.001502
grad_step = 000362, loss = 0.001497
grad_step = 000363, loss = 0.001479
grad_step = 000364, loss = 0.001462
grad_step = 000365, loss = 0.001455
grad_step = 000366, loss = 0.001458
grad_step = 000367, loss = 0.001465
grad_step = 000368, loss = 0.001472
grad_step = 000369, loss = 0.001480
grad_step = 000370, loss = 0.001493
grad_step = 000371, loss = 0.001512
grad_step = 000372, loss = 0.001539
grad_step = 000373, loss = 0.001567
grad_step = 000374, loss = 0.001596
grad_step = 000375, loss = 0.001591
grad_step = 000376, loss = 0.001553
grad_step = 000377, loss = 0.001485
grad_step = 000378, loss = 0.001430
grad_step = 000379, loss = 0.001421
grad_step = 000380, loss = 0.001449
grad_step = 000381, loss = 0.001465
grad_step = 000382, loss = 0.001448
grad_step = 000383, loss = 0.001414
grad_step = 000384, loss = 0.001406
grad_step = 000385, loss = 0.001424
grad_step = 000386, loss = 0.001433
grad_step = 000387, loss = 0.001420
grad_step = 000388, loss = 0.001397
grad_step = 000389, loss = 0.001388
grad_step = 000390, loss = 0.001393
grad_step = 000391, loss = 0.001401
grad_step = 000392, loss = 0.001399
grad_step = 000393, loss = 0.001391
grad_step = 000394, loss = 0.001381
grad_step = 000395, loss = 0.001374
grad_step = 000396, loss = 0.001369
grad_step = 000397, loss = 0.001367
grad_step = 000398, loss = 0.001368
grad_step = 000399, loss = 0.001369
grad_step = 000400, loss = 0.001373
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001384
grad_step = 000402, loss = 0.001417
grad_step = 000403, loss = 0.001493
grad_step = 000404, loss = 0.001649
grad_step = 000405, loss = 0.001827
grad_step = 000406, loss = 0.001981
grad_step = 000407, loss = 0.001693
grad_step = 000408, loss = 0.001397
grad_step = 000409, loss = 0.001409
grad_step = 000410, loss = 0.001576
grad_step = 000411, loss = 0.001518
grad_step = 000412, loss = 0.001370
grad_step = 000413, loss = 0.001468
grad_step = 000414, loss = 0.001510
grad_step = 000415, loss = 0.001380
grad_step = 000416, loss = 0.001422
grad_step = 000417, loss = 0.001477
grad_step = 000418, loss = 0.001373
grad_step = 000419, loss = 0.001390
grad_step = 000420, loss = 0.001445
grad_step = 000421, loss = 0.001380
grad_step = 000422, loss = 0.001349
grad_step = 000423, loss = 0.001400
grad_step = 000424, loss = 0.001394
grad_step = 000425, loss = 0.001346
grad_step = 000426, loss = 0.001341
grad_step = 000427, loss = 0.001374
grad_step = 000428, loss = 0.001384
grad_step = 000429, loss = 0.001355
grad_step = 000430, loss = 0.001330
grad_step = 000431, loss = 0.001335
grad_step = 000432, loss = 0.001354
grad_step = 000433, loss = 0.001363
grad_step = 000434, loss = 0.001351
grad_step = 000435, loss = 0.001333
grad_step = 000436, loss = 0.001322
grad_step = 000437, loss = 0.001325
grad_step = 000438, loss = 0.001334
grad_step = 000439, loss = 0.001338
grad_step = 000440, loss = 0.001334
grad_step = 000441, loss = 0.001322
grad_step = 000442, loss = 0.001315
grad_step = 000443, loss = 0.001314
grad_step = 000444, loss = 0.001319
grad_step = 000445, loss = 0.001321
grad_step = 000446, loss = 0.001319
grad_step = 000447, loss = 0.001314
grad_step = 000448, loss = 0.001309
grad_step = 000449, loss = 0.001306
grad_step = 000450, loss = 0.001306
grad_step = 000451, loss = 0.001309
grad_step = 000452, loss = 0.001310
grad_step = 000453, loss = 0.001310
grad_step = 000454, loss = 0.001308
grad_step = 000455, loss = 0.001305
grad_step = 000456, loss = 0.001301
grad_step = 000457, loss = 0.001299
grad_step = 000458, loss = 0.001297
grad_step = 000459, loss = 0.001295
grad_step = 000460, loss = 0.001294
grad_step = 000461, loss = 0.001293
grad_step = 000462, loss = 0.001292
grad_step = 000463, loss = 0.001291
grad_step = 000464, loss = 0.001290
grad_step = 000465, loss = 0.001289
grad_step = 000466, loss = 0.001288
grad_step = 000467, loss = 0.001287
grad_step = 000468, loss = 0.001287
grad_step = 000469, loss = 0.001288
grad_step = 000470, loss = 0.001292
grad_step = 000471, loss = 0.001300
grad_step = 000472, loss = 0.001319
grad_step = 000473, loss = 0.001358
grad_step = 000474, loss = 0.001430
grad_step = 000475, loss = 0.001541
grad_step = 000476, loss = 0.001642
grad_step = 000477, loss = 0.001692
grad_step = 000478, loss = 0.001547
grad_step = 000479, loss = 0.001352
grad_step = 000480, loss = 0.001286
grad_step = 000481, loss = 0.001393
grad_step = 000482, loss = 0.001452
grad_step = 000483, loss = 0.001350
grad_step = 000484, loss = 0.001286
grad_step = 000485, loss = 0.001357
grad_step = 000486, loss = 0.001378
grad_step = 000487, loss = 0.001300
grad_step = 000488, loss = 0.001291
grad_step = 000489, loss = 0.001344
grad_step = 000490, loss = 0.001329
grad_step = 000491, loss = 0.001274
grad_step = 000492, loss = 0.001284
grad_step = 000493, loss = 0.001320
grad_step = 000494, loss = 0.001308
grad_step = 000495, loss = 0.001271
grad_step = 000496, loss = 0.001263
grad_step = 000497, loss = 0.001285
grad_step = 000498, loss = 0.001303
grad_step = 000499, loss = 0.001294
grad_step = 000500, loss = 0.001272
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001257
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

  date_run                              2020-05-10 02:14:44.573588
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.167122
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 02:14:44.579968
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0626295
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 02:14:44.588122
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.113981
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 02:14:44.593078
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0483228
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
0   2020-05-10 02:14:15.542952  ...    mean_absolute_error
1   2020-05-10 02:14:15.546668  ...     mean_squared_error
2   2020-05-10 02:14:15.549775  ...  median_absolute_error
3   2020-05-10 02:14:15.552932  ...               r2_score
4   2020-05-10 02:14:26.029700  ...    mean_absolute_error
5   2020-05-10 02:14:26.033426  ...     mean_squared_error
6   2020-05-10 02:14:26.036702  ...  median_absolute_error
7   2020-05-10 02:14:26.039780  ...               r2_score
8   2020-05-10 02:14:44.573588  ...    mean_absolute_error
9   2020-05-10 02:14:44.579968  ...     mean_squared_error
10  2020-05-10 02:14:44.588122  ...  median_absolute_error
11  2020-05-10 02:14:44.593078  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 146972.16it/s] 81%|████████  | 7987200/9912422 [00:00<00:09, 209791.48it/s]9920512it [00:00, 43746529.91it/s]                           
0it [00:00, ?it/s]32768it [00:00, 579380.71it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477352.29it/s]1654784it [00:00, 12012501.41it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202767.35it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcf24f83780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcec26c79b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcf24f3ae48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcec26c7da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcf24f3ae48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcf24f83e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcf24f83780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fced7935cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcec26c7da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fced7935cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcf24f3ae48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc815d7a1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d151e2abc95b00a9586dfa94b3c94905389f91e87e400c61c51ed2ef17554fee
  Stored in directory: /tmp/pip-ephem-wheel-cache-ep7a96sq/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc80dbfbfd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3670016/17464789 [=====>........................] - ETA: 0s
 9650176/17464789 [===============>..............] - ETA: 0s
16375808/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 02:16:10.637436: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 02:16:10.641465: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 02:16:10.641594: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5583db7fc920 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 02:16:10.641607: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.1606 - accuracy: 0.5330
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5210 - accuracy: 0.5095 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5184 - accuracy: 0.5097
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6145 - accuracy: 0.5034
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6078 - accuracy: 0.5038
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6321 - accuracy: 0.5023
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6394 - accuracy: 0.5018
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
11000/25000 [============>.................] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6383 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6677 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6915 - accuracy: 0.4984
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6901 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6835 - accuracy: 0.4989
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 274us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 02:16:24.064713
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 02:16:24.064713  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 02:16:30.004708: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 02:16:30.009998: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 02:16:30.010579: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5559c62473a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 02:16:30.010605: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6f4f934be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4869 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.3720 - val_crf_viterbi_accuracy: 0.0667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6f2a879f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5210 - accuracy: 0.5095 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5031 - accuracy: 0.5107
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6250 - accuracy: 0.5027
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6321 - accuracy: 0.5023
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6121 - accuracy: 0.5036
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
12000/25000 [=============>................] - ETA: 3s - loss: 7.6781 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6713 - accuracy: 0.4997
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6699 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
25000/25000 [==============================] - 7s 279us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f6ee66097f0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:35:19, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:38:31, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:18:11, 23.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:13:14, 33.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:02:30, 47.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.87M/862M [00:01<3:30:19, 67.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:01<2:26:20, 96.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:41:51, 138kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:01<1:10:58, 196kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<49:35, 280kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.9M/862M [00:01<34:37, 399kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<24:13, 567kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.9M/862M [00:02<16:56, 806kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<11:54, 1.14MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<08:31, 1.58MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<06:22, 2.11MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<11:25:31, 19.6kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<8:00:26, 28.0kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:04<5:35:32, 39.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<4:00:18, 55.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<2:51:01, 78.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<2:00:11, 111kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:06<1:23:59, 159kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:08<1:13:53, 180kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<52:52, 252kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<37:17, 356kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:10<29:03, 456kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<21:42, 610kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:10<15:30, 852kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<13:53, 949kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<12:23, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<09:19, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<08:38, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<07:11, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<05:22, 2.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<06:46, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<06:03, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<04:30, 2.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:18<06:12, 2.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<05:39, 2.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<04:16, 3.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:20<06:02, 2.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<05:32, 2.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<04:11, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<05:58, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<06:46, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<05:16, 2.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<03:51, 3.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<09:09, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<07:43, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<05:41, 2.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<06:57, 1.82MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:28, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:51, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:14, 2.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<12:08:54, 17.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<8:31:15, 24.7kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<5:57:26, 35.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<4:12:25, 49.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<2:57:52, 70.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<2:04:31, 101kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<1:29:53, 139kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<1:05:06, 192kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<46:12, 270kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<32:36, 382kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<26:29, 469kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<19:46, 624kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<15:11, 812kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<10:57, 1.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<10:19, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<09:42, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:24, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<05:18, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<11:51:49, 17.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<8:19:17, 24.4kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<5:49:04, 34.9kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<4:06:28, 49.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<2:54:57, 69.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<2:02:57, 98.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<1:26:21, 140kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<8:29:44, 23.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<5:56:39, 33.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<4:10:52, 47.9kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<2:58:38, 67.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<2:05:33, 95.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<1:27:59, 136kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<1:04:20, 186kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<46:13, 258kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<32:35, 366kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<25:33, 465kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<19:07, 621kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<13:39, 867kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<12:19, 958kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<09:48, 1.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<07:09, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<07:47, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<06:37, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<04:53, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<05:30, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:08, 2.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<05:38, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<05:07, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<03:52, 2.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<05:26, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<06:03, 1.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<04:43, 2.43MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:27, 3.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<08:52, 1.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:24, 1.55MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:27, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:29, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:51, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:22, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<05:34, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:03, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<03:46, 2.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:17, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:59, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:42, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:25, 3.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<09:07, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<07:31, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:32, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:28, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:46, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:12, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:46, 2.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<08:46, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:16, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:21, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:19, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:39, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:12, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:45, 2.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<1:21:36, 134kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<58:11, 187kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<40:53, 266kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<31:04, 349kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<23:56, 453kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<17:12, 629kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<12:07, 889kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<1:14:36, 145kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<7:20:08, 24.5kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<5:07:58, 35.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<3:36:30, 49.5kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<2:33:36, 69.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<1:47:54, 99.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<1:16:55, 138kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<54:45, 194kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<38:32, 276kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<29:15, 362kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<22:32, 469kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<16:17, 649kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<13:03, 805kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<11:22, 924kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<08:30, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<06:03, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<35:39, 293kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<26:01, 401kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<18:26, 565kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<15:18, 678kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<12:54, 804kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<09:33, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<06:48, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<10:10, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:27, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:11, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<06:20, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:12, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:41, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:08, 2.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:07, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:58, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:09, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:36, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:12, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:57, 2.54MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:44, 2.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:02, 1.66MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<10:20, 970kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:18, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:27, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:39, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<08:14, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<08:10, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:20, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<04:33, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:26, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:10, 1.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:20, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:37, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:13, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:54, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:39, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:29, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:27, 2.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:18, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:28, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:27, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:15, 2.94MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:10, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:14, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:37, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:06, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:00, 1.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:48, 1.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:30, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:19, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:20, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:40, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:06, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:58, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:42, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:24, 2.73MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:22, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<05:41, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:13, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:46, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:41, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:30, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:15, 2.81MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:14, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<05:34, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:42, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:36, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:25, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:18, 2.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:21, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:14, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:14, 2.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:01, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:57, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:00, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:12, 4.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<20:24, 434kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<16:32, 535kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<12:02, 734kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:33, 1.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<08:38, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:11, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<05:17, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:24, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:54, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:42, 2.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:17, 2.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:08, 2.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:03, 2.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:59, 2.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<4:55:10, 29.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<3:26:23, 41.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<2:25:17, 58.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<1:43:52, 82.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<1:13:06, 116kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<51:07, 166kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<38:26, 220kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<27:58, 302kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<19:46, 426kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<13:53, 604kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<24:00, 349kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<17:54, 468kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<12:46, 654kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<10:31, 789kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<08:26, 983kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:07, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:54, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:16, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:51, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:31, 2.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:54, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:30, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:24, 2.39MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:58, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:48, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:53, 2.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:36, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:32, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:42, 2.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:27, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:29, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:36, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:38, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:03, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:47, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:10, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:46, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:44, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<02:43, 2.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:26, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:11, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:34, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:42, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:00, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:56, 1.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:50, 2.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<09:37, 791kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:41, 990kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:34, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<05:22, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:34, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:29, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:32, 2.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<05:13, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:34, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:19, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:06, 2.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:12, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:34, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:50, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:34, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:40, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:39, 2.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:53, 1.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:22, 1.66MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:16, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:42, 1.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:34, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:39, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:40, 2.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:31, 1.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:47, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:32, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:52, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:32, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:34, 1.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:34, 2.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<05:03, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:18, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:20, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:26, 2.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:21, 1.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:49, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:52, 2.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:28, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:12, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:19, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:24, 2.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:52, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:16, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:12, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:33, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:12, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:22, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:27, 2.71MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:08, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<04:26, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:17, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:35, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:22, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:32, 2.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:03, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:48, 1.71MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:04, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:13, 2.89MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:45, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:09, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:06, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:25, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:07, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:13, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:20, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:37, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:20, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:31, 2.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:59, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:43, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:11, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:41, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:02, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:59, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:18, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:39, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:52, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:04, 2.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:14, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:14, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:06, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:16, 2.63MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<06:39, 893kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<06:12, 958kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<04:40, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:24, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:41, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:20, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:29, 2.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:53, 2.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:31, 1.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:46, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:05, 2.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:43, 2.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:38, 2.17MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:00, 2.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:45, 3.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<3:32:56, 26.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<2:28:42, 38.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<1:44:16, 53.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<1:14:11, 75.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<52:04, 108kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<36:19, 153kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<27:27, 202kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<19:54, 278kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<14:04, 393kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<10:50, 505kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<08:51, 618kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:27, 846kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<04:33, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:47, 935kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:44, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:27, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:27, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:47, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:00, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:09, 2.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:41, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:15, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:26, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:43, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:14, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:36, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:53, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:44, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:16, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:26, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:41, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:11, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:30, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:49, 2.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:50, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:36, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:57, 2.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:20, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:10, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:38, 2.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:11, 2.22MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:04, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:34, 3.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:07, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:47, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:12, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:36, 2.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:38, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:27, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:51, 2.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:12, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:48, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:12, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:35, 2.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:58, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:39, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:00, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:16, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:49, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:15, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:38, 2.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:25, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:57, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:12, 2.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:23, 1.82MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:48, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:14, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:36, 2.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:16, 1.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:50, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:07, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:18, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:10, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:38, 2.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:11, 3.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<06:33, 635kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<05:08, 808kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:43, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<03:22, 1.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:48, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:15, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:37, 2.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:41, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:24, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:46, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:01, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:28, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:59, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:26, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:48, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:27, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:50, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:01, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:23, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:53, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:21, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:41, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:21, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:45, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:57, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:18, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:51, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:20, 2.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:47, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:24, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:46, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:16, 2.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<06:19, 560kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<04:48, 734kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:25, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:07, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:05, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:22, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:42, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:04, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:53, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:24, 2.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:01, 3.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<05:01, 662kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<03:57, 841kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:51, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:36, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:15, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:39, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:10, 2.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<1:36:20, 33.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<1:08:11, 46.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<47:46, 66.5kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<33:04, 94.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<24:33, 127kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<17:34, 177kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<12:19, 251kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<09:06, 335kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<07:08, 427kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<05:09, 590kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:36, 832kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:35, 831kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:54, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<02:06, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:01, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:05, 1.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:36, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<01:08, 2.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:48, 1.58MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<1:47:38, 26.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<1:14:53, 37.9kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<51:40, 54.0kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<40:03, 69.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<28:43, 96.9kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<20:11, 137kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<14:01, 196kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<10:35, 256kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<07:43, 351kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<05:26, 494kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:18, 614kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<03:41, 716kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:44, 959kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:56, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:37, 980kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:10, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:34, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:35, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:25, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:03, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:12, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:07, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:50, 2.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:36, 3.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:42, 876kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:26, 970kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:50, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<03:01, 759kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:22, 966kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:42, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:39, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:46, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:22, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:58, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:44, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:30, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:06, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:10, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:05, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:49, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:57, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:12, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:01, 1.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:45, 2.61MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:52, 2.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:39, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:49, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:05, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:52, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:38, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:22, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:11, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:52, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:37, 2.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:40, 654kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:05, 832kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:29, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<01:02, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<08:22, 201kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<06:18, 266kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<04:29, 371kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<03:05, 525kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<02:56, 547kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<02:14, 717kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:34, 999kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:24, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:22, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:02, 1.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:44, 2.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:53, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:47, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:34, 2.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:41, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:50, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:40, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:28, 2.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:59, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:51, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:37, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:40, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:47, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:26, 2.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:50, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:43, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:32, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:23, 2.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:51, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:53, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:41, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:29, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:51, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:43, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:31, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:33, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:38, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:30, 1.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:21, 2.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:42, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:36, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:18, 2.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:38, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:39, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:30, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:21, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:37, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:31, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:23, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:16, 2.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:33, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:28, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:20, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:24, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:13, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:12, 2.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:17, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:09, 2.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:14, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:10, 2.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:07, 2.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 2.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:09, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:08, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:05, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 3.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<03:33, 26.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<01:56, 37.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:28, 53.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:18, 75.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:07, 107kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 816/400000 [00:00<00:48, 8157.63it/s]  0%|          | 1679/400000 [00:00<00:48, 8292.02it/s]  1%|          | 2550/400000 [00:00<00:47, 8410.53it/s]  1%|          | 3412/400000 [00:00<00:46, 8472.29it/s]  1%|          | 4276/400000 [00:00<00:46, 8520.29it/s]  1%|▏         | 5150/400000 [00:00<00:46, 8582.52it/s]  2%|▏         | 6024/400000 [00:00<00:45, 8628.67it/s]  2%|▏         | 6894/400000 [00:00<00:45, 8649.71it/s]  2%|▏         | 7767/400000 [00:00<00:45, 8671.27it/s]  2%|▏         | 8643/400000 [00:01<00:45, 8696.26it/s]  2%|▏         | 9515/400000 [00:01<00:44, 8701.79it/s]  3%|▎         | 10380/400000 [00:01<00:44, 8685.55it/s]  3%|▎         | 11238/400000 [00:01<00:45, 8548.12it/s]  3%|▎         | 12086/400000 [00:01<00:46, 8406.89it/s]  3%|▎         | 12957/400000 [00:01<00:45, 8494.65it/s]  3%|▎         | 13827/400000 [00:01<00:45, 8553.83it/s]  4%|▎         | 14696/400000 [00:01<00:44, 8593.80it/s]  4%|▍         | 15576/400000 [00:01<00:44, 8653.43it/s]  4%|▍         | 16454/400000 [00:01<00:44, 8688.62it/s]  4%|▍         | 17331/400000 [00:02<00:43, 8712.61it/s]  5%|▍         | 18206/400000 [00:02<00:43, 8721.04it/s]  5%|▍         | 19078/400000 [00:02<00:43, 8715.91it/s]  5%|▍         | 19952/400000 [00:02<00:43, 8722.30it/s]  5%|▌         | 20826/400000 [00:02<00:43, 8727.32it/s]  5%|▌         | 21699/400000 [00:02<00:43, 8708.66it/s]  6%|▌         | 22574/400000 [00:02<00:43, 8720.91it/s]  6%|▌         | 23447/400000 [00:02<00:43, 8720.69it/s]  6%|▌         | 24320/400000 [00:02<00:43, 8694.62it/s]  6%|▋         | 25192/400000 [00:02<00:43, 8701.92it/s]  7%|▋         | 26063/400000 [00:03<00:43, 8679.11it/s]  7%|▋         | 26931/400000 [00:03<00:43, 8481.53it/s]  7%|▋         | 27794/400000 [00:03<00:43, 8524.68it/s]  7%|▋         | 28648/400000 [00:03<00:43, 8510.02it/s]  7%|▋         | 29518/400000 [00:03<00:43, 8564.72it/s]  8%|▊         | 30389/400000 [00:03<00:42, 8606.23it/s]  8%|▊         | 31263/400000 [00:03<00:42, 8645.13it/s]  8%|▊         | 32132/400000 [00:03<00:42, 8656.56it/s]  8%|▊         | 33002/400000 [00:03<00:42, 8667.71it/s]  8%|▊         | 33869/400000 [00:03<00:43, 8411.85it/s]  9%|▊         | 34735/400000 [00:04<00:43, 8482.09it/s]  9%|▉         | 35602/400000 [00:04<00:42, 8536.08it/s]  9%|▉         | 36475/400000 [00:04<00:42, 8590.90it/s]  9%|▉         | 37340/400000 [00:04<00:42, 8606.65it/s] 10%|▉         | 38207/400000 [00:04<00:41, 8624.44it/s] 10%|▉         | 39070/400000 [00:04<00:41, 8609.76it/s] 10%|▉         | 39933/400000 [00:04<00:41, 8615.62it/s] 10%|█         | 40801/400000 [00:04<00:41, 8634.06it/s] 10%|█         | 41682/400000 [00:04<00:41, 8685.01it/s] 11%|█         | 42556/400000 [00:04<00:41, 8699.32it/s] 11%|█         | 43427/400000 [00:05<00:41, 8693.18it/s] 11%|█         | 44298/400000 [00:05<00:40, 8696.44it/s] 11%|█▏        | 45168/400000 [00:05<00:40, 8688.92it/s] 12%|█▏        | 46041/400000 [00:05<00:40, 8700.06it/s] 12%|█▏        | 46915/400000 [00:05<00:40, 8709.37it/s] 12%|█▏        | 47786/400000 [00:05<00:40, 8673.45it/s] 12%|█▏        | 48654/400000 [00:05<00:40, 8675.35it/s] 12%|█▏        | 49527/400000 [00:05<00:40, 8691.18it/s] 13%|█▎        | 50398/400000 [00:05<00:40, 8694.72it/s] 13%|█▎        | 51270/400000 [00:05<00:40, 8700.39it/s] 13%|█▎        | 52141/400000 [00:06<00:40, 8693.72it/s] 13%|█▎        | 53014/400000 [00:06<00:39, 8704.35it/s] 13%|█▎        | 53886/400000 [00:06<00:39, 8703.41it/s] 14%|█▎        | 54757/400000 [00:06<00:39, 8704.71it/s] 14%|█▍        | 55631/400000 [00:06<00:39, 8713.00it/s] 14%|█▍        | 56503/400000 [00:06<00:39, 8708.25it/s] 14%|█▍        | 57374/400000 [00:06<00:39, 8698.74it/s] 15%|█▍        | 58246/400000 [00:06<00:39, 8703.40it/s] 15%|█▍        | 59117/400000 [00:06<00:39, 8649.24it/s] 15%|█▍        | 59987/400000 [00:06<00:39, 8661.94it/s] 15%|█▌        | 60854/400000 [00:07<00:39, 8653.82it/s] 15%|█▌        | 61727/400000 [00:07<00:38, 8675.14it/s] 16%|█▌        | 62601/400000 [00:07<00:38, 8693.30it/s] 16%|█▌        | 63476/400000 [00:07<00:38, 8707.70it/s] 16%|█▌        | 64347/400000 [00:07<00:38, 8661.18it/s] 16%|█▋        | 65219/400000 [00:07<00:38, 8677.01it/s] 17%|█▋        | 66091/400000 [00:07<00:38, 8687.16it/s] 17%|█▋        | 66960/400000 [00:07<00:38, 8620.92it/s] 17%|█▋        | 67829/400000 [00:07<00:38, 8640.02it/s] 17%|█▋        | 68701/400000 [00:07<00:38, 8661.89it/s] 17%|█▋        | 69572/400000 [00:08<00:38, 8674.83it/s] 18%|█▊        | 70447/400000 [00:08<00:37, 8696.98it/s] 18%|█▊        | 71317/400000 [00:08<00:37, 8684.42it/s] 18%|█▊        | 72186/400000 [00:08<00:37, 8681.54it/s] 18%|█▊        | 73061/400000 [00:08<00:37, 8701.08it/s] 18%|█▊        | 73937/400000 [00:08<00:37, 8716.84it/s] 19%|█▊        | 74813/400000 [00:08<00:37, 8728.62it/s] 19%|█▉        | 75687/400000 [00:08<00:37, 8731.77it/s] 19%|█▉        | 76561/400000 [00:08<00:37, 8728.00it/s] 19%|█▉        | 77434/400000 [00:08<00:36, 8720.80it/s] 20%|█▉        | 78307/400000 [00:09<00:36, 8718.63it/s] 20%|█▉        | 79179/400000 [00:09<00:36, 8711.24it/s] 20%|██        | 80051/400000 [00:09<00:38, 8417.63it/s] 20%|██        | 80916/400000 [00:09<00:37, 8484.54it/s] 20%|██        | 81800/400000 [00:09<00:37, 8585.79it/s] 21%|██        | 82664/400000 [00:09<00:36, 8601.98it/s] 21%|██        | 83526/400000 [00:09<00:36, 8585.56it/s] 21%|██        | 84386/400000 [00:09<00:36, 8541.73it/s] 21%|██▏       | 85260/400000 [00:09<00:36, 8597.71it/s] 22%|██▏       | 86130/400000 [00:09<00:36, 8625.48it/s] 22%|██▏       | 86997/400000 [00:10<00:36, 8638.43it/s] 22%|██▏       | 87870/400000 [00:10<00:36, 8665.61it/s] 22%|██▏       | 88743/400000 [00:10<00:35, 8683.06it/s] 22%|██▏       | 89612/400000 [00:10<00:35, 8683.23it/s] 23%|██▎       | 90481/400000 [00:10<00:35, 8676.93it/s] 23%|██▎       | 91352/400000 [00:10<00:35, 8683.99it/s] 23%|██▎       | 92221/400000 [00:10<00:35, 8675.93it/s] 23%|██▎       | 93089/400000 [00:10<00:35, 8554.29it/s] 23%|██▎       | 93963/400000 [00:10<00:35, 8606.58it/s] 24%|██▎       | 94833/400000 [00:10<00:35, 8633.22it/s] 24%|██▍       | 95698/400000 [00:11<00:35, 8637.12it/s] 24%|██▍       | 96562/400000 [00:11<00:35, 8553.46it/s] 24%|██▍       | 97429/400000 [00:11<00:35, 8585.61it/s] 25%|██▍       | 98288/400000 [00:11<00:35, 8563.32it/s] 25%|██▍       | 99154/400000 [00:11<00:35, 8589.92it/s] 25%|██▌       | 100014/400000 [00:11<00:34, 8590.40it/s] 25%|██▌       | 100887/400000 [00:11<00:34, 8631.74it/s] 25%|██▌       | 101755/400000 [00:11<00:34, 8646.03it/s] 26%|██▌       | 102630/400000 [00:11<00:34, 8675.08it/s] 26%|██▌       | 103501/400000 [00:11<00:34, 8684.04it/s] 26%|██▌       | 104370/400000 [00:12<00:34, 8672.57it/s] 26%|██▋       | 105238/400000 [00:12<00:34, 8649.81it/s] 27%|██▋       | 106104/400000 [00:12<00:34, 8626.94it/s] 27%|██▋       | 106978/400000 [00:12<00:33, 8657.75it/s] 27%|██▋       | 107851/400000 [00:12<00:33, 8676.77it/s] 27%|██▋       | 108727/400000 [00:12<00:33, 8699.30it/s] 27%|██▋       | 109597/400000 [00:12<00:33, 8680.07it/s] 28%|██▊       | 110466/400000 [00:12<00:33, 8606.34it/s] 28%|██▊       | 111327/400000 [00:12<00:33, 8583.66it/s] 28%|██▊       | 112204/400000 [00:12<00:33, 8637.48it/s] 28%|██▊       | 113078/400000 [00:13<00:33, 8666.49it/s] 28%|██▊       | 113945/400000 [00:13<00:33, 8665.42it/s] 29%|██▊       | 114812/400000 [00:13<00:32, 8665.20it/s] 29%|██▉       | 115682/400000 [00:13<00:32, 8674.70it/s] 29%|██▉       | 116555/400000 [00:13<00:32, 8690.27it/s] 29%|██▉       | 117426/400000 [00:13<00:32, 8696.06it/s] 30%|██▉       | 118296/400000 [00:13<00:32, 8679.25it/s] 30%|██▉       | 119167/400000 [00:13<00:32, 8686.70it/s] 30%|███       | 120036/400000 [00:13<00:32, 8565.94it/s] 30%|███       | 120908/400000 [00:13<00:32, 8610.34it/s] 30%|███       | 121775/400000 [00:14<00:32, 8626.61it/s] 31%|███       | 122644/400000 [00:14<00:32, 8643.11it/s] 31%|███       | 123512/400000 [00:14<00:31, 8650.83it/s] 31%|███       | 124387/400000 [00:14<00:31, 8680.00it/s] 31%|███▏      | 125257/400000 [00:14<00:31, 8683.12it/s] 32%|███▏      | 126127/400000 [00:14<00:31, 8686.39it/s] 32%|███▏      | 126997/400000 [00:14<00:31, 8687.96it/s] 32%|███▏      | 127866/400000 [00:14<00:31, 8659.69it/s] 32%|███▏      | 128733/400000 [00:14<00:31, 8593.99it/s] 32%|███▏      | 129605/400000 [00:14<00:31, 8631.23it/s] 33%|███▎      | 130469/400000 [00:15<00:31, 8616.58it/s] 33%|███▎      | 131331/400000 [00:15<00:32, 8388.60it/s] 33%|███▎      | 132187/400000 [00:15<00:31, 8439.21it/s] 33%|███▎      | 133044/400000 [00:15<00:31, 8475.82it/s] 33%|███▎      | 133915/400000 [00:15<00:31, 8542.67it/s] 34%|███▎      | 134796/400000 [00:15<00:30, 8618.59it/s] 34%|███▍      | 135668/400000 [00:15<00:30, 8645.98it/s] 34%|███▍      | 136540/400000 [00:15<00:30, 8665.31it/s] 34%|███▍      | 137415/400000 [00:15<00:30, 8687.72it/s] 35%|███▍      | 138287/400000 [00:16<00:30, 8694.69it/s] 35%|███▍      | 139157/400000 [00:16<00:30, 8686.91it/s] 35%|███▌      | 140026/400000 [00:16<00:29, 8668.47it/s] 35%|███▌      | 140896/400000 [00:16<00:29, 8676.75it/s] 35%|███▌      | 141765/400000 [00:16<00:29, 8680.20it/s] 36%|███▌      | 142639/400000 [00:16<00:29, 8694.80it/s] 36%|███▌      | 143509/400000 [00:16<00:29, 8675.32it/s] 36%|███▌      | 144377/400000 [00:16<00:29, 8603.40it/s] 36%|███▋      | 145241/400000 [00:16<00:29, 8612.86it/s] 37%|███▋      | 146113/400000 [00:16<00:29, 8643.47it/s] 37%|███▋      | 146978/400000 [00:17<00:29, 8558.11it/s] 37%|███▋      | 147855/400000 [00:17<00:29, 8619.54it/s] 37%|███▋      | 148725/400000 [00:17<00:29, 8641.92it/s] 37%|███▋      | 149592/400000 [00:17<00:28, 8648.43it/s] 38%|███▊      | 150466/400000 [00:17<00:28, 8674.13it/s] 38%|███▊      | 151334/400000 [00:17<00:28, 8667.65it/s] 38%|███▊      | 152213/400000 [00:17<00:28, 8701.29it/s] 38%|███▊      | 153084/400000 [00:17<00:28, 8701.85it/s] 38%|███▊      | 153955/400000 [00:17<00:28, 8642.11it/s] 39%|███▊      | 154826/400000 [00:17<00:28, 8660.72it/s] 39%|███▉      | 155696/400000 [00:18<00:28, 8670.99it/s] 39%|███▉      | 156564/400000 [00:18<00:28, 8659.11it/s] 39%|███▉      | 157432/400000 [00:18<00:27, 8664.41it/s] 40%|███▉      | 158299/400000 [00:18<00:28, 8577.61it/s] 40%|███▉      | 159174/400000 [00:18<00:27, 8626.49it/s] 40%|████      | 160044/400000 [00:18<00:27, 8646.15it/s] 40%|████      | 160919/400000 [00:18<00:27, 8675.20it/s] 40%|████      | 161798/400000 [00:18<00:27, 8706.46it/s] 41%|████      | 162669/400000 [00:18<00:27, 8694.93it/s] 41%|████      | 163543/400000 [00:18<00:27, 8708.25it/s] 41%|████      | 164418/400000 [00:19<00:27, 8717.94it/s] 41%|████▏     | 165294/400000 [00:19<00:26, 8727.99it/s] 42%|████▏     | 166167/400000 [00:19<00:27, 8635.38it/s] 42%|████▏     | 167031/400000 [00:19<00:27, 8479.22it/s] 42%|████▏     | 167900/400000 [00:19<00:27, 8538.80it/s] 42%|████▏     | 168774/400000 [00:19<00:26, 8596.19it/s] 42%|████▏     | 169646/400000 [00:19<00:26, 8632.41it/s] 43%|████▎     | 170520/400000 [00:19<00:26, 8662.36it/s] 43%|████▎     | 171387/400000 [00:19<00:26, 8657.74it/s] 43%|████▎     | 172259/400000 [00:19<00:26, 8674.33it/s] 43%|████▎     | 173127/400000 [00:20<00:26, 8568.41it/s] 43%|████▎     | 173985/400000 [00:20<00:26, 8568.91it/s] 44%|████▎     | 174852/400000 [00:20<00:26, 8596.35it/s] 44%|████▍     | 175719/400000 [00:20<00:26, 8616.55it/s] 44%|████▍     | 176589/400000 [00:20<00:25, 8640.62it/s] 44%|████▍     | 177454/400000 [00:20<00:25, 8640.50it/s] 45%|████▍     | 178328/400000 [00:20<00:25, 8669.33it/s] 45%|████▍     | 179202/400000 [00:20<00:25, 8690.31it/s] 45%|████▌     | 180072/400000 [00:20<00:25, 8681.77it/s] 45%|████▌     | 180948/400000 [00:20<00:25, 8702.83it/s] 45%|████▌     | 181819/400000 [00:21<00:25, 8628.97it/s] 46%|████▌     | 182691/400000 [00:21<00:25, 8655.96it/s] 46%|████▌     | 183565/400000 [00:21<00:24, 8678.74it/s] 46%|████▌     | 184433/400000 [00:21<00:25, 8551.12it/s] 46%|████▋     | 185297/400000 [00:21<00:25, 8575.77it/s] 47%|████▋     | 186175/400000 [00:21<00:24, 8633.88it/s] 47%|████▋     | 187049/400000 [00:21<00:24, 8664.62it/s] 47%|████▋     | 187923/400000 [00:21<00:24, 8685.90it/s] 47%|████▋     | 188794/400000 [00:21<00:24, 8690.34it/s] 47%|████▋     | 189669/400000 [00:21<00:24, 8705.37it/s] 48%|████▊     | 190542/400000 [00:22<00:24, 8710.91it/s] 48%|████▊     | 191417/400000 [00:22<00:23, 8721.26it/s] 48%|████▊     | 192290/400000 [00:22<00:23, 8716.25it/s] 48%|████▊     | 193162/400000 [00:22<00:23, 8712.53it/s] 49%|████▊     | 194034/400000 [00:22<00:23, 8712.16it/s] 49%|████▊     | 194908/400000 [00:22<00:23, 8719.45it/s] 49%|████▉     | 195780/400000 [00:22<00:23, 8605.25it/s] 49%|████▉     | 196641/400000 [00:22<00:23, 8605.54it/s] 49%|████▉     | 197502/400000 [00:22<00:23, 8572.76it/s] 50%|████▉     | 198363/400000 [00:22<00:23, 8582.33it/s] 50%|████▉     | 199229/400000 [00:23<00:23, 8603.86it/s] 50%|█████     | 200090/400000 [00:23<00:23, 8596.01it/s] 50%|█████     | 200963/400000 [00:23<00:23, 8633.13it/s] 50%|█████     | 201827/400000 [00:23<00:23, 8581.19it/s] 51%|█████     | 202697/400000 [00:23<00:22, 8614.61it/s] 51%|█████     | 203568/400000 [00:23<00:22, 8642.86it/s] 51%|█████     | 204441/400000 [00:23<00:22, 8668.35it/s] 51%|█████▏    | 205316/400000 [00:23<00:22, 8689.86it/s] 52%|█████▏    | 206186/400000 [00:23<00:22, 8667.28it/s] 52%|█████▏    | 207057/400000 [00:23<00:22, 8679.59it/s] 52%|█████▏    | 207939/400000 [00:24<00:22, 8718.55it/s] 52%|█████▏    | 208817/400000 [00:24<00:21, 8736.14it/s] 52%|█████▏    | 209691/400000 [00:24<00:21, 8719.87it/s] 53%|█████▎    | 210564/400000 [00:24<00:21, 8706.63it/s] 53%|█████▎    | 211435/400000 [00:24<00:22, 8308.92it/s] 53%|█████▎    | 212288/400000 [00:24<00:22, 8372.13it/s] 53%|█████▎    | 213161/400000 [00:24<00:22, 8474.09it/s] 54%|█████▎    | 214032/400000 [00:24<00:21, 8543.23it/s] 54%|█████▎    | 214909/400000 [00:24<00:21, 8607.52it/s] 54%|█████▍    | 215783/400000 [00:24<00:21, 8645.00it/s] 54%|█████▍    | 216657/400000 [00:25<00:21, 8672.14it/s] 54%|█████▍    | 217525/400000 [00:25<00:21, 8665.55it/s] 55%|█████▍    | 218393/400000 [00:25<00:21, 8526.26it/s] 55%|█████▍    | 219252/400000 [00:25<00:21, 8542.57it/s] 55%|█████▌    | 220107/400000 [00:25<00:21, 8531.96it/s] 55%|█████▌    | 220982/400000 [00:25<00:20, 8595.65it/s] 55%|█████▌    | 221858/400000 [00:25<00:20, 8641.74it/s] 56%|█████▌    | 222723/400000 [00:25<00:21, 8301.35it/s] 56%|█████▌    | 223557/400000 [00:25<00:21, 8250.67it/s] 56%|█████▌    | 224428/400000 [00:25<00:20, 8381.51it/s] 56%|█████▋    | 225302/400000 [00:26<00:20, 8483.22it/s] 57%|█████▋    | 226170/400000 [00:26<00:20, 8538.54it/s] 57%|█████▋    | 227041/400000 [00:26<00:20, 8587.13it/s] 57%|█████▋    | 227911/400000 [00:26<00:19, 8618.14it/s] 57%|█████▋    | 228782/400000 [00:26<00:19, 8644.46it/s] 57%|█████▋    | 229656/400000 [00:26<00:19, 8670.99it/s] 58%|█████▊    | 230524/400000 [00:26<00:19, 8631.48it/s] 58%|█████▊    | 231388/400000 [00:26<00:20, 8408.93it/s] 58%|█████▊    | 232231/400000 [00:26<00:20, 8346.11it/s] 58%|█████▊    | 233082/400000 [00:27<00:19, 8392.24it/s] 58%|█████▊    | 233956/400000 [00:27<00:19, 8492.05it/s] 59%|█████▊    | 234830/400000 [00:27<00:19, 8562.70it/s] 59%|█████▉    | 235688/400000 [00:27<00:19, 8464.99it/s] 59%|█████▉    | 236536/400000 [00:27<00:19, 8313.83it/s] 59%|█████▉    | 237406/400000 [00:27<00:19, 8424.18it/s] 60%|█████▉    | 238278/400000 [00:27<00:19, 8507.19it/s] 60%|█████▉    | 239147/400000 [00:27<00:18, 8559.10it/s] 60%|██████    | 240020/400000 [00:27<00:18, 8609.43it/s] 60%|██████    | 240882/400000 [00:27<00:18, 8528.22it/s] 60%|██████    | 241743/400000 [00:28<00:18, 8550.78it/s] 61%|██████    | 242599/400000 [00:28<00:18, 8466.10it/s] 61%|██████    | 243447/400000 [00:28<00:19, 8142.26it/s] 61%|██████    | 244265/400000 [00:28<00:19, 8069.43it/s] 61%|██████▏   | 245132/400000 [00:28<00:18, 8239.82it/s] 62%|██████▏   | 246006/400000 [00:28<00:18, 8381.36it/s] 62%|██████▏   | 246877/400000 [00:28<00:18, 8474.93it/s] 62%|██████▏   | 247750/400000 [00:28<00:17, 8547.83it/s] 62%|██████▏   | 248623/400000 [00:28<00:17, 8600.17it/s] 62%|██████▏   | 249493/400000 [00:28<00:17, 8627.01it/s] 63%|██████▎   | 250365/400000 [00:29<00:17, 8651.91it/s] 63%|██████▎   | 251240/400000 [00:29<00:17, 8679.43it/s] 63%|██████▎   | 252113/400000 [00:29<00:17, 8692.17it/s] 63%|██████▎   | 252983/400000 [00:29<00:16, 8678.84it/s] 63%|██████▎   | 253857/400000 [00:29<00:16, 8696.46it/s] 64%|██████▎   | 254731/400000 [00:29<00:16, 8709.18it/s] 64%|██████▍   | 255603/400000 [00:29<00:17, 8390.22it/s] 64%|██████▍   | 256445/400000 [00:29<00:17, 8191.27it/s] 64%|██████▍   | 257314/400000 [00:29<00:17, 8334.03it/s] 65%|██████▍   | 258164/400000 [00:29<00:16, 8380.52it/s] 65%|██████▍   | 259034/400000 [00:30<00:16, 8471.51it/s] 65%|██████▍   | 259892/400000 [00:30<00:16, 8501.58it/s] 65%|██████▌   | 260744/400000 [00:30<00:16, 8354.98it/s] 65%|██████▌   | 261613/400000 [00:30<00:16, 8452.10it/s] 66%|██████▌   | 262487/400000 [00:30<00:16, 8536.40it/s] 66%|██████▌   | 263360/400000 [00:30<00:15, 8590.84it/s] 66%|██████▌   | 264234/400000 [00:30<00:15, 8633.37it/s] 66%|██████▋   | 265098/400000 [00:30<00:15, 8629.83it/s] 66%|██████▋   | 265974/400000 [00:30<00:15, 8666.77it/s] 67%|██████▋   | 266842/400000 [00:30<00:15, 8666.44it/s] 67%|██████▋   | 267709/400000 [00:31<00:15, 8571.16it/s] 67%|██████▋   | 268567/400000 [00:31<00:15, 8552.62it/s] 67%|██████▋   | 269442/400000 [00:31<00:15, 8608.83it/s] 68%|██████▊   | 270313/400000 [00:31<00:15, 8636.32it/s] 68%|██████▊   | 271177/400000 [00:31<00:14, 8635.85it/s] 68%|██████▊   | 272041/400000 [00:31<00:14, 8635.60it/s] 68%|██████▊   | 272905/400000 [00:31<00:14, 8626.77it/s] 68%|██████▊   | 273775/400000 [00:31<00:14, 8646.45it/s] 69%|██████▊   | 274651/400000 [00:31<00:14, 8677.04it/s] 69%|██████▉   | 275519/400000 [00:31<00:14, 8667.30it/s] 69%|██████▉   | 276386/400000 [00:32<00:14, 8650.92it/s] 69%|██████▉   | 277257/400000 [00:32<00:14, 8668.27it/s] 70%|██████▉   | 278124/400000 [00:32<00:14, 8520.61it/s] 70%|██████▉   | 278993/400000 [00:32<00:14, 8570.39it/s] 70%|██████▉   | 279864/400000 [00:32<00:13, 8610.57it/s] 70%|███████   | 280733/400000 [00:32<00:13, 8633.93it/s] 70%|███████   | 281609/400000 [00:32<00:13, 8671.02it/s] 71%|███████   | 282477/400000 [00:32<00:13, 8450.28it/s] 71%|███████   | 283324/400000 [00:32<00:13, 8431.74it/s] 71%|███████   | 284192/400000 [00:32<00:13, 8503.31it/s] 71%|███████▏  | 285054/400000 [00:33<00:13, 8536.36it/s] 71%|███████▏  | 285931/400000 [00:33<00:13, 8602.95it/s] 72%|███████▏  | 286809/400000 [00:33<00:13, 8654.83it/s] 72%|███████▏  | 287679/400000 [00:33<00:12, 8668.07it/s] 72%|███████▏  | 288548/400000 [00:33<00:12, 8672.73it/s] 72%|███████▏  | 289421/400000 [00:33<00:12, 8687.47it/s] 73%|███████▎  | 290295/400000 [00:33<00:12, 8700.49it/s] 73%|███████▎  | 291169/400000 [00:33<00:12, 8710.07it/s] 73%|███████▎  | 292043/400000 [00:33<00:12, 8716.50it/s] 73%|███████▎  | 292915/400000 [00:33<00:12, 8709.81it/s] 73%|███████▎  | 293787/400000 [00:34<00:12, 8704.36it/s] 74%|███████▎  | 294662/400000 [00:34<00:12, 8715.49it/s] 74%|███████▍  | 295534/400000 [00:34<00:12, 8433.75it/s] 74%|███████▍  | 296411/400000 [00:34<00:12, 8530.57it/s] 74%|███████▍  | 297266/400000 [00:34<00:12, 8467.21it/s] 75%|███████▍  | 298141/400000 [00:34<00:11, 8549.62it/s] 75%|███████▍  | 298998/400000 [00:34<00:12, 8300.68it/s] 75%|███████▍  | 299873/400000 [00:34<00:11, 8429.16it/s] 75%|███████▌  | 300741/400000 [00:34<00:11, 8501.31it/s] 75%|███████▌  | 301599/400000 [00:35<00:11, 8522.66it/s] 76%|███████▌  | 302471/400000 [00:35<00:11, 8579.06it/s] 76%|███████▌  | 303343/400000 [00:35<00:11, 8619.13it/s] 76%|███████▌  | 304215/400000 [00:35<00:11, 8647.17it/s] 76%|███████▋  | 305085/400000 [00:35<00:10, 8660.46it/s] 76%|███████▋  | 305952/400000 [00:35<00:10, 8654.92it/s] 77%|███████▋  | 306825/400000 [00:35<00:10, 8675.92it/s] 77%|███████▋  | 307693/400000 [00:35<00:11, 8201.24it/s] 77%|███████▋  | 308540/400000 [00:35<00:11, 8277.65it/s] 77%|███████▋  | 309412/400000 [00:35<00:10, 8403.01it/s] 78%|███████▊  | 310282/400000 [00:36<00:10, 8487.06it/s] 78%|███████▊  | 311155/400000 [00:36<00:10, 8557.40it/s] 78%|███████▊  | 312028/400000 [00:36<00:10, 8607.40it/s] 78%|███████▊  | 312907/400000 [00:36<00:10, 8659.58it/s] 78%|███████▊  | 313793/400000 [00:36<00:09, 8716.27it/s] 79%|███████▊  | 314666/400000 [00:36<00:09, 8718.75it/s] 79%|███████▉  | 315539/400000 [00:36<00:09, 8718.56it/s] 79%|███████▉  | 316412/400000 [00:36<00:09, 8706.00it/s] 79%|███████▉  | 317285/400000 [00:36<00:09, 8711.24it/s] 80%|███████▉  | 318157/400000 [00:36<00:09, 8701.67it/s] 80%|███████▉  | 319028/400000 [00:37<00:09, 8703.33it/s] 80%|███████▉  | 319901/400000 [00:37<00:09, 8709.98it/s] 80%|████████  | 320773/400000 [00:37<00:09, 8703.52it/s] 80%|████████  | 321647/400000 [00:37<00:08, 8714.07it/s] 81%|████████  | 322519/400000 [00:37<00:08, 8710.80it/s] 81%|████████  | 323391/400000 [00:37<00:08, 8648.24it/s] 81%|████████  | 324256/400000 [00:37<00:08, 8640.60it/s] 81%|████████▏ | 325129/400000 [00:37<00:08, 8666.77it/s] 81%|████████▏ | 325996/400000 [00:37<00:08, 8549.09it/s] 82%|████████▏ | 326852/400000 [00:37<00:08, 8384.42it/s] 82%|████████▏ | 327692/400000 [00:38<00:08, 8321.98it/s] 82%|████████▏ | 328566/400000 [00:38<00:08, 8441.52it/s] 82%|████████▏ | 329427/400000 [00:38<00:08, 8489.98it/s] 83%|████████▎ | 330294/400000 [00:38<00:08, 8542.45it/s] 83%|████████▎ | 331167/400000 [00:38<00:08, 8596.60it/s] 83%|████████▎ | 332036/400000 [00:38<00:07, 8624.00it/s] 83%|████████▎ | 332912/400000 [00:38<00:07, 8662.63it/s] 83%|████████▎ | 333791/400000 [00:38<00:07, 8700.08it/s] 84%|████████▎ | 334667/400000 [00:38<00:07, 8715.78it/s] 84%|████████▍ | 335541/400000 [00:38<00:07, 8722.15it/s] 84%|████████▍ | 336414/400000 [00:39<00:07, 8720.27it/s] 84%|████████▍ | 337288/400000 [00:39<00:07, 8724.10it/s] 85%|████████▍ | 338162/400000 [00:39<00:07, 8727.16it/s] 85%|████████▍ | 339035/400000 [00:39<00:06, 8711.22it/s] 85%|████████▍ | 339907/400000 [00:39<00:06, 8689.26it/s] 85%|████████▌ | 340780/400000 [00:39<00:06, 8699.31it/s] 85%|████████▌ | 341650/400000 [00:39<00:06, 8697.70it/s] 86%|████████▌ | 342522/400000 [00:39<00:06, 8703.86it/s] 86%|████████▌ | 343395/400000 [00:39<00:06, 8710.79it/s] 86%|████████▌ | 344267/400000 [00:39<00:06, 8489.10it/s] 86%|████████▋ | 345132/400000 [00:40<00:06, 8533.94it/s] 86%|████████▋ | 345995/400000 [00:40<00:06, 8561.68it/s] 87%|████████▋ | 346859/400000 [00:40<00:06, 8580.43it/s] 87%|████████▋ | 347728/400000 [00:40<00:06, 8612.12it/s] 87%|████████▋ | 348595/400000 [00:40<00:05, 8628.75it/s] 87%|████████▋ | 349467/400000 [00:40<00:05, 8654.75it/s] 88%|████████▊ | 350337/400000 [00:40<00:05, 8667.61it/s] 88%|████████▊ | 351211/400000 [00:40<00:05, 8686.46it/s] 88%|████████▊ | 352081/400000 [00:40<00:05, 8688.61it/s] 88%|████████▊ | 352954/400000 [00:40<00:05, 8699.52it/s] 88%|████████▊ | 353826/400000 [00:41<00:05, 8705.49it/s] 89%|████████▊ | 354698/400000 [00:41<00:05, 8706.81it/s] 89%|████████▉ | 355572/400000 [00:41<00:05, 8715.42it/s] 89%|████████▉ | 356444/400000 [00:41<00:05, 8710.12it/s] 89%|████████▉ | 357316/400000 [00:41<00:04, 8704.69it/s] 90%|████████▉ | 358188/400000 [00:41<00:04, 8708.41it/s] 90%|████████▉ | 359059/400000 [00:41<00:04, 8664.63it/s] 90%|████████▉ | 359926/400000 [00:41<00:04, 8423.94it/s] 90%|█████████ | 360796/400000 [00:41<00:04, 8502.26it/s] 90%|█████████ | 361668/400000 [00:41<00:04, 8563.55it/s] 91%|█████████ | 362538/400000 [00:42<00:04, 8603.91it/s] 91%|█████████ | 363410/400000 [00:42<00:04, 8637.26it/s] 91%|█████████ | 364282/400000 [00:42<00:04, 8659.82it/s] 91%|█████████▏| 365155/400000 [00:42<00:04, 8678.12it/s] 92%|█████████▏| 366029/400000 [00:42<00:03, 8693.89it/s] 92%|█████████▏| 366899/400000 [00:42<00:03, 8692.73it/s] 92%|█████████▏| 367769/400000 [00:42<00:03, 8692.24it/s] 92%|█████████▏| 368642/400000 [00:42<00:03, 8701.98it/s] 92%|█████████▏| 369513/400000 [00:42<00:03, 8624.53it/s] 93%|█████████▎| 370376/400000 [00:42<00:03, 8617.48it/s] 93%|█████████▎| 371238/400000 [00:43<00:03, 8605.18it/s] 93%|█████████▎| 372111/400000 [00:43<00:03, 8639.91it/s] 93%|█████████▎| 372983/400000 [00:43<00:03, 8663.36it/s] 93%|█████████▎| 373850/400000 [00:43<00:03, 8415.10it/s] 94%|█████████▎| 374694/400000 [00:43<00:03, 8392.34it/s] 94%|█████████▍| 375559/400000 [00:43<00:02, 8467.75it/s] 94%|█████████▍| 376441/400000 [00:43<00:02, 8568.64it/s] 94%|█████████▍| 377316/400000 [00:43<00:02, 8620.61it/s] 95%|█████████▍| 378181/400000 [00:43<00:02, 8627.99it/s] 95%|█████████▍| 379045/400000 [00:43<00:02, 8631.02it/s] 95%|█████████▍| 379916/400000 [00:44<00:02, 8654.25it/s] 95%|█████████▌| 380789/400000 [00:44<00:02, 8674.54it/s] 95%|█████████▌| 381657/400000 [00:44<00:02, 8638.15it/s] 96%|█████████▌| 382530/400000 [00:44<00:02, 8663.48it/s] 96%|█████████▌| 383407/400000 [00:44<00:01, 8692.30it/s] 96%|█████████▌| 384277/400000 [00:44<00:01, 8685.97it/s] 96%|█████████▋| 385146/400000 [00:44<00:01, 8648.59it/s] 97%|█████████▋| 386017/400000 [00:44<00:01, 8664.91it/s] 97%|█████████▋| 386886/400000 [00:44<00:01, 8671.97it/s] 97%|█████████▋| 387758/400000 [00:45<00:01, 8684.53it/s] 97%|█████████▋| 388627/400000 [00:45<00:01, 8659.97it/s] 97%|█████████▋| 389494/400000 [00:45<00:01, 8648.05it/s] 98%|█████████▊| 390359/400000 [00:45<00:01, 8554.83it/s] 98%|█████████▊| 391240/400000 [00:45<00:01, 8629.78it/s] 98%|█████████▊| 392117/400000 [00:45<00:00, 8670.94it/s] 98%|█████████▊| 392985/400000 [00:45<00:00, 8621.00it/s] 98%|█████████▊| 393848/400000 [00:45<00:00, 8609.04it/s] 99%|█████████▊| 394721/400000 [00:45<00:00, 8642.71it/s] 99%|█████████▉| 395594/400000 [00:45<00:00, 8665.87it/s] 99%|█████████▉| 396466/400000 [00:46<00:00, 8680.43it/s] 99%|█████████▉| 397336/400000 [00:46<00:00, 8683.65it/s]100%|█████████▉| 398205/400000 [00:46<00:00, 8493.17it/s]100%|█████████▉| 399073/400000 [00:46<00:00, 8547.61it/s]100%|█████████▉| 399946/400000 [00:46<00:00, 8601.29it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8616.06it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6f349e6320> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01143369261969483 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011305275569392686 	 Accuracy: 49

  model saves at 49% accuracy 

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

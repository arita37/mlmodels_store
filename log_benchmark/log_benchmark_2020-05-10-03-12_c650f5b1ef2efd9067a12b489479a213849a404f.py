
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fca84bbb470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 03:13:01.952597
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 03:13:01.956293
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 03:13:01.959646
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 03:13:01.962786
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fca6a936a20> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354333.7500
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 288493.5312
Epoch 3/10

1/1 [==============================] - 0s 93ms/step - loss: 220644.5312
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 156710.0312
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 104069.1406
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 66860.2812
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 43957.0039
Epoch 8/10

1/1 [==============================] - 0s 91ms/step - loss: 30347.4570
Epoch 9/10

1/1 [==============================] - 0s 90ms/step - loss: 21789.6250
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 16137.9512

  #### Inference Need return ypred, ytrue ######################### 
[[ 8.93162310e-01 -4.90023255e-01  1.53045550e-01 -1.54152080e-01
   5.52589893e-01 -7.18575478e-01  1.10763621e+00  5.53350151e-01
   8.69031787e-01 -9.16767538e-01  6.83882475e-01  5.29260516e-01
  -1.93351388e-01 -5.57178974e-01 -1.47491229e+00 -1.00365317e+00
   2.97566533e-01 -5.98781586e-01 -7.34444320e-01 -2.27041900e-01
   3.53386134e-01 -8.31007510e-02 -8.09583902e-01  3.77864063e-01
   1.19542956e+00  8.26222181e-01  8.16482782e-01  4.27424133e-01
  -2.07763016e-01  1.44468963e-01 -5.48678100e-01  5.13563275e-01
   2.66621143e-01 -1.60247326e+00  4.72609311e-01 -1.14783466e-01
  -3.58961105e-01 -6.26129389e-01 -3.07167381e-01  3.89900923e-01
   1.01495945e+00  1.11656941e-01 -5.80802262e-01  1.06882346e+00
   1.01245272e+00  3.00484538e-01 -4.50148940e-01  1.55284762e+00
  -1.56564593e+00 -1.38203427e-01  2.14344978e-01  3.94629449e-01
  -6.31537855e-01  1.17696679e+00 -4.05322492e-01 -7.94634461e-01
   6.25608921e-01 -8.22549015e-02 -4.44652021e-01 -3.08801800e-01
  -9.67572480e-02  3.66641760e-01  1.14362526e+00  3.22685212e-01
   5.55881202e-01  1.05043054e+00  8.62668335e-01 -9.85230431e-02
   1.51738799e+00  2.87762940e-01  3.46115679e-01 -4.85986412e-01
  -1.04655433e+00 -1.04600203e+00  4.52237204e-03 -6.38654530e-01
  -8.17885101e-01 -3.56819451e-01 -1.41204149e-01  1.79280591e+00
   6.30854547e-01  3.91187608e-01  1.06747997e+00 -9.25562158e-03
  -1.35266924e+00 -3.52451205e-02  1.02548826e+00  1.57101953e+00
  -7.09001482e-01  2.96532959e-02  1.01828754e+00 -4.84945595e-01
   6.53808713e-01 -8.14860612e-02  4.24510032e-01 -1.21270284e-01
  -5.86524069e-01  1.28519225e+00  5.75542331e-01 -1.36772561e+00
   6.05426311e-01 -1.23275146e-01  1.41156995e+00 -4.33676541e-02
  -1.32948011e-01 -2.52803206e-01  5.95246792e-01  2.76656598e-02
   7.96867371e-01 -3.50609541e-01  3.51794213e-01 -8.10621977e-01
  -6.09473526e-01 -3.81090403e-01  9.52737570e-01 -1.07137537e+00
  -6.32121801e-01  1.31502128e+00 -1.53344750e-01 -6.62375242e-02
  -1.44491106e-01  4.65067101e+00  4.65652561e+00  4.76978636e+00
   4.59624910e+00  5.59069777e+00  3.84798574e+00  3.90474916e+00
   3.58638334e+00  4.85718489e+00  5.62633991e+00  4.54779911e+00
   5.85853672e+00  4.40163898e+00  4.11102057e+00  5.79626417e+00
   5.96878433e+00  4.17450619e+00  6.17293453e+00  4.19195747e+00
   3.74031806e+00  5.38537836e+00  5.44695330e+00  6.11939764e+00
   5.09155321e+00  5.31453800e+00  4.91607666e+00  6.27104378e+00
   5.34362650e+00  3.96818495e+00  4.03910160e+00  5.90802526e+00
   5.56106329e+00  4.93168306e+00  4.86264658e+00  5.42769575e+00
   4.04207802e+00  4.79593086e+00  3.57543349e+00  4.02461243e+00
   3.77621484e+00  6.11457777e+00  4.63688660e+00  3.04195571e+00
   4.77495670e+00  5.41319275e+00  4.65670824e+00  5.57158422e+00
   4.70936155e+00  4.10598278e+00  6.15701771e+00  5.34217834e+00
   4.88913202e+00  4.97358227e+00  4.60309649e+00  4.38590813e+00
   4.57055426e+00  5.94161558e+00  4.97124577e+00  5.80832338e+00
   2.15382147e+00  2.03484631e+00  2.72055805e-01  5.88479221e-01
   2.35825872e+00  8.58002305e-01  6.99097991e-01  1.65839171e+00
   1.88625383e+00  9.35123026e-01  1.69169557e+00  1.37200844e+00
   8.08241844e-01  1.30007863e+00  1.57739830e+00  1.98447347e+00
   3.63518536e-01  3.11276913e-01  1.81583059e+00  2.80218840e-01
   1.63192129e+00  2.00876045e+00  1.85139894e+00  5.64568818e-01
   2.54739809e+00  6.49072230e-01  1.25638378e+00  1.71251965e+00
   3.63819242e-01  1.80103302e+00  1.62647843e+00  2.97453833e+00
   1.84007943e+00  4.87158298e-01  2.73335576e-01  2.26369810e+00
   4.63434994e-01  5.68171263e-01  4.16037798e-01  1.39951110e-01
   5.03226459e-01  1.21939373e+00  1.93873596e+00  3.74043941e-01
   6.15756214e-01  4.45312560e-01  7.32658327e-01  2.03063750e+00
   4.99652088e-01  1.73346615e+00  1.16220891e+00  3.86113763e-01
   4.76963758e-01  1.21430504e+00  1.67666781e+00  5.77598572e-01
   2.77102709e-01  3.64156604e-01  9.00839806e-01  2.57901764e+00
   2.29694366e-01  3.68330479e-01  1.54084849e+00  7.17927277e-01
   6.20879114e-01  2.73595333e-01  1.98839116e+00  1.06034732e+00
   4.25858617e-01  1.44867849e+00  3.90586078e-01  6.56203151e-01
   2.01812291e+00  7.42458880e-01  1.84536123e+00  5.95580876e-01
   5.24935544e-01  1.01800776e+00  2.00562572e+00  4.16354716e-01
   2.21488655e-01  7.74793983e-01  1.81571531e+00  1.46193087e+00
   6.98884308e-01  1.66864526e+00  2.24984670e+00  1.20962358e+00
   1.33727443e+00  1.59343123e-01  9.85756457e-01  2.43214488e-01
   1.69224000e+00  1.82457685e+00  1.01178956e+00  5.66089511e-01
   1.05633330e+00  1.95730162e+00  1.57122588e+00  1.05349541e+00
   4.91871476e-01  6.66590571e-01  1.08513832e+00  4.59289432e-01
   1.65573645e+00  9.43541110e-01  1.37182951e+00  8.46069336e-01
   3.78611505e-01  1.60269713e+00  1.05747259e+00  1.60806561e+00
   8.06474209e-01  2.98655868e-01  5.41166186e-01  2.08802056e+00
   4.34674382e-01  2.23600864e+00  8.88783336e-01  4.55012441e-01
   3.63358259e-02  5.59318447e+00  4.40680265e+00  6.15861988e+00
   4.80512333e+00  6.70892239e+00  6.35189533e+00  5.47408152e+00
   5.02513981e+00  6.09037209e+00  4.77391386e+00  4.18275928e+00
   4.74355030e+00  4.28727055e+00  6.24649763e+00  4.70018816e+00
   4.28266430e+00  5.93432283e+00  6.40894699e+00  6.05532360e+00
   6.23703241e+00  4.26506901e+00  5.31730652e+00  4.96295738e+00
   6.34329939e+00  4.67437649e+00  5.86647892e+00  6.67587996e+00
   6.04784679e+00  5.04559422e+00  5.25617123e+00  6.51470709e+00
   4.83838320e+00  4.95089626e+00  5.38658953e+00  5.35769510e+00
   5.49327326e+00  5.73414326e+00  5.06194162e+00  6.34631777e+00
   4.24026537e+00  5.32781458e+00  7.19296074e+00  6.48285484e+00
   4.08434343e+00  6.80974054e+00  4.88192892e+00  6.84231472e+00
   5.18051910e+00  5.37580538e+00  4.65554237e+00  3.97573662e+00
   6.20640707e+00  5.54935265e+00  5.79864120e+00  6.15321493e+00
   4.97226048e+00  5.96758127e+00  6.03228426e+00  6.02217436e+00
  -9.37755013e+00  1.46564651e+00  6.89675522e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 03:13:10.611898
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.0222
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 03:13:10.615723
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9433.17
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 03:13:10.618863
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.9388
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 03:13:10.621974
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -843.807
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140507098063816
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140505888060304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140505888060808
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140505888061312
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140505888061816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140505888062320

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fca78d8def0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.410802
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.377783
grad_step = 000002, loss = 0.354349
grad_step = 000003, loss = 0.330674
grad_step = 000004, loss = 0.306275
grad_step = 000005, loss = 0.282961
grad_step = 000006, loss = 0.260433
grad_step = 000007, loss = 0.239771
grad_step = 000008, loss = 0.218279
grad_step = 000009, loss = 0.203376
grad_step = 000010, loss = 0.201463
grad_step = 000011, loss = 0.194560
grad_step = 000012, loss = 0.181984
grad_step = 000013, loss = 0.171131
grad_step = 000014, loss = 0.163194
grad_step = 000015, loss = 0.157385
grad_step = 000016, loss = 0.151650
grad_step = 000017, loss = 0.145216
grad_step = 000018, loss = 0.138249
grad_step = 000019, loss = 0.130795
grad_step = 000020, loss = 0.123786
grad_step = 000021, loss = 0.117031
grad_step = 000022, loss = 0.110484
grad_step = 000023, loss = 0.104540
grad_step = 000024, loss = 0.099204
grad_step = 000025, loss = 0.093948
grad_step = 000026, loss = 0.088790
grad_step = 000027, loss = 0.083880
grad_step = 000028, loss = 0.078955
grad_step = 000029, loss = 0.073993
grad_step = 000030, loss = 0.069295
grad_step = 000031, loss = 0.064813
grad_step = 000032, loss = 0.060518
grad_step = 000033, loss = 0.056709
grad_step = 000034, loss = 0.053412
grad_step = 000035, loss = 0.050054
grad_step = 000036, loss = 0.046506
grad_step = 000037, loss = 0.043219
grad_step = 000038, loss = 0.040329
grad_step = 000039, loss = 0.037675
grad_step = 000040, loss = 0.035166
grad_step = 000041, loss = 0.032735
grad_step = 000042, loss = 0.030362
grad_step = 000043, loss = 0.028139
grad_step = 000044, loss = 0.026055
grad_step = 000045, loss = 0.024029
grad_step = 000046, loss = 0.022163
grad_step = 000047, loss = 0.020542
grad_step = 000048, loss = 0.019072
grad_step = 000049, loss = 0.017676
grad_step = 000050, loss = 0.016336
grad_step = 000051, loss = 0.015042
grad_step = 000052, loss = 0.013847
grad_step = 000053, loss = 0.012811
grad_step = 000054, loss = 0.011871
grad_step = 000055, loss = 0.010973
grad_step = 000056, loss = 0.010137
grad_step = 000057, loss = 0.009373
grad_step = 000058, loss = 0.008683
grad_step = 000059, loss = 0.008051
grad_step = 000060, loss = 0.007456
grad_step = 000061, loss = 0.006930
grad_step = 000062, loss = 0.006465
grad_step = 000063, loss = 0.006023
grad_step = 000064, loss = 0.005604
grad_step = 000065, loss = 0.005215
grad_step = 000066, loss = 0.004876
grad_step = 000067, loss = 0.004586
grad_step = 000068, loss = 0.004309
grad_step = 000069, loss = 0.004053
grad_step = 000070, loss = 0.003829
grad_step = 000071, loss = 0.003630
grad_step = 000072, loss = 0.003451
grad_step = 000073, loss = 0.003283
grad_step = 000074, loss = 0.003139
grad_step = 000075, loss = 0.003015
grad_step = 000076, loss = 0.002900
grad_step = 000077, loss = 0.002799
grad_step = 000078, loss = 0.002709
grad_step = 000079, loss = 0.002635
grad_step = 000080, loss = 0.002575
grad_step = 000081, loss = 0.002519
grad_step = 000082, loss = 0.002468
grad_step = 000083, loss = 0.002426
grad_step = 000084, loss = 0.002392
grad_step = 000085, loss = 0.002361
grad_step = 000086, loss = 0.002334
grad_step = 000087, loss = 0.002314
grad_step = 000088, loss = 0.002298
grad_step = 000089, loss = 0.002281
grad_step = 000090, loss = 0.002266
grad_step = 000091, loss = 0.002255
grad_step = 000092, loss = 0.002247
grad_step = 000093, loss = 0.002238
grad_step = 000094, loss = 0.002229
grad_step = 000095, loss = 0.002222
grad_step = 000096, loss = 0.002216
grad_step = 000097, loss = 0.002209
grad_step = 000098, loss = 0.002203
grad_step = 000099, loss = 0.002198
grad_step = 000100, loss = 0.002192
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002186
grad_step = 000102, loss = 0.002179
grad_step = 000103, loss = 0.002173
grad_step = 000104, loss = 0.002167
grad_step = 000105, loss = 0.002161
grad_step = 000106, loss = 0.002154
grad_step = 000107, loss = 0.002148
grad_step = 000108, loss = 0.002141
grad_step = 000109, loss = 0.002134
grad_step = 000110, loss = 0.002127
grad_step = 000111, loss = 0.002120
grad_step = 000112, loss = 0.002113
grad_step = 000113, loss = 0.002106
grad_step = 000114, loss = 0.002099
grad_step = 000115, loss = 0.002092
grad_step = 000116, loss = 0.002085
grad_step = 000117, loss = 0.002078
grad_step = 000118, loss = 0.002071
grad_step = 000119, loss = 0.002064
grad_step = 000120, loss = 0.002058
grad_step = 000121, loss = 0.002051
grad_step = 000122, loss = 0.002045
grad_step = 000123, loss = 0.002038
grad_step = 000124, loss = 0.002032
grad_step = 000125, loss = 0.002026
grad_step = 000126, loss = 0.002019
grad_step = 000127, loss = 0.002013
grad_step = 000128, loss = 0.002007
grad_step = 000129, loss = 0.002001
grad_step = 000130, loss = 0.001994
grad_step = 000131, loss = 0.001988
grad_step = 000132, loss = 0.001982
grad_step = 000133, loss = 0.001976
grad_step = 000134, loss = 0.001969
grad_step = 000135, loss = 0.001963
grad_step = 000136, loss = 0.001957
grad_step = 000137, loss = 0.001951
grad_step = 000138, loss = 0.001945
grad_step = 000139, loss = 0.001940
grad_step = 000140, loss = 0.001938
grad_step = 000141, loss = 0.001945
grad_step = 000142, loss = 0.001978
grad_step = 000143, loss = 0.002016
grad_step = 000144, loss = 0.002023
grad_step = 000145, loss = 0.001927
grad_step = 000146, loss = 0.001914
grad_step = 000147, loss = 0.001969
grad_step = 000148, loss = 0.001941
grad_step = 000149, loss = 0.001893
grad_step = 000150, loss = 0.001896
grad_step = 000151, loss = 0.001926
grad_step = 000152, loss = 0.001932
grad_step = 000153, loss = 0.001888
grad_step = 000154, loss = 0.001869
grad_step = 000155, loss = 0.001887
grad_step = 000156, loss = 0.001902
grad_step = 000157, loss = 0.001895
grad_step = 000158, loss = 0.001864
grad_step = 000159, loss = 0.001852
grad_step = 000160, loss = 0.001863
grad_step = 000161, loss = 0.001874
grad_step = 000162, loss = 0.001872
grad_step = 000163, loss = 0.001851
grad_step = 000164, loss = 0.001836
grad_step = 000165, loss = 0.001837
grad_step = 000166, loss = 0.001844
grad_step = 000167, loss = 0.001849
grad_step = 000168, loss = 0.001840
grad_step = 000169, loss = 0.001829
grad_step = 000170, loss = 0.001820
grad_step = 000171, loss = 0.001818
grad_step = 000172, loss = 0.001820
grad_step = 000173, loss = 0.001822
grad_step = 000174, loss = 0.001825
grad_step = 000175, loss = 0.001824
grad_step = 000176, loss = 0.001824
grad_step = 000177, loss = 0.001819
grad_step = 000178, loss = 0.001814
grad_step = 000179, loss = 0.001806
grad_step = 000180, loss = 0.001800
grad_step = 000181, loss = 0.001794
grad_step = 000182, loss = 0.001790
grad_step = 000183, loss = 0.001789
grad_step = 000184, loss = 0.001789
grad_step = 000185, loss = 0.001791
grad_step = 000186, loss = 0.001798
grad_step = 000187, loss = 0.001816
grad_step = 000188, loss = 0.001836
grad_step = 000189, loss = 0.001876
grad_step = 000190, loss = 0.001866
grad_step = 000191, loss = 0.001835
grad_step = 000192, loss = 0.001781
grad_step = 000193, loss = 0.001773
grad_step = 000194, loss = 0.001804
grad_step = 000195, loss = 0.001822
grad_step = 000196, loss = 0.001818
grad_step = 000197, loss = 0.001778
grad_step = 000198, loss = 0.001756
grad_step = 000199, loss = 0.001762
grad_step = 000200, loss = 0.001780
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001793
grad_step = 000202, loss = 0.001776
grad_step = 000203, loss = 0.001755
grad_step = 000204, loss = 0.001742
grad_step = 000205, loss = 0.001747
grad_step = 000206, loss = 0.001758
grad_step = 000207, loss = 0.001758
grad_step = 000208, loss = 0.001749
grad_step = 000209, loss = 0.001735
grad_step = 000210, loss = 0.001728
grad_step = 000211, loss = 0.001728
grad_step = 000212, loss = 0.001732
grad_step = 000213, loss = 0.001735
grad_step = 000214, loss = 0.001732
grad_step = 000215, loss = 0.001725
grad_step = 000216, loss = 0.001716
grad_step = 000217, loss = 0.001711
grad_step = 000218, loss = 0.001709
grad_step = 000219, loss = 0.001710
grad_step = 000220, loss = 0.001712
grad_step = 000221, loss = 0.001713
grad_step = 000222, loss = 0.001715
grad_step = 000223, loss = 0.001714
grad_step = 000224, loss = 0.001712
grad_step = 000225, loss = 0.001705
grad_step = 000226, loss = 0.001700
grad_step = 000227, loss = 0.001695
grad_step = 000228, loss = 0.001691
grad_step = 000229, loss = 0.001687
grad_step = 000230, loss = 0.001684
grad_step = 000231, loss = 0.001681
grad_step = 000232, loss = 0.001679
grad_step = 000233, loss = 0.001678
grad_step = 000234, loss = 0.001678
grad_step = 000235, loss = 0.001680
grad_step = 000236, loss = 0.001688
grad_step = 000237, loss = 0.001701
grad_step = 000238, loss = 0.001730
grad_step = 000239, loss = 0.001755
grad_step = 000240, loss = 0.001802
grad_step = 000241, loss = 0.001767
grad_step = 000242, loss = 0.001716
grad_step = 000243, loss = 0.001656
grad_step = 000244, loss = 0.001662
grad_step = 000245, loss = 0.001705
grad_step = 000246, loss = 0.001706
grad_step = 000247, loss = 0.001684
grad_step = 000248, loss = 0.001643
grad_step = 000249, loss = 0.001641
grad_step = 000250, loss = 0.001667
grad_step = 000251, loss = 0.001673
grad_step = 000252, loss = 0.001660
grad_step = 000253, loss = 0.001632
grad_step = 000254, loss = 0.001624
grad_step = 000255, loss = 0.001635
grad_step = 000256, loss = 0.001646
grad_step = 000257, loss = 0.001650
grad_step = 000258, loss = 0.001633
grad_step = 000259, loss = 0.001617
grad_step = 000260, loss = 0.001607
grad_step = 000261, loss = 0.001609
grad_step = 000262, loss = 0.001615
grad_step = 000263, loss = 0.001619
grad_step = 000264, loss = 0.001617
grad_step = 000265, loss = 0.001608
grad_step = 000266, loss = 0.001598
grad_step = 000267, loss = 0.001591
grad_step = 000268, loss = 0.001588
grad_step = 000269, loss = 0.001588
grad_step = 000270, loss = 0.001590
grad_step = 000271, loss = 0.001595
grad_step = 000272, loss = 0.001598
grad_step = 000273, loss = 0.001606
grad_step = 000274, loss = 0.001609
grad_step = 000275, loss = 0.001615
grad_step = 000276, loss = 0.001609
grad_step = 000277, loss = 0.001602
grad_step = 000278, loss = 0.001585
grad_step = 000279, loss = 0.001571
grad_step = 000280, loss = 0.001560
grad_step = 000281, loss = 0.001557
grad_step = 000282, loss = 0.001558
grad_step = 000283, loss = 0.001564
grad_step = 000284, loss = 0.001577
grad_step = 000285, loss = 0.001592
grad_step = 000286, loss = 0.001627
grad_step = 000287, loss = 0.001637
grad_step = 000288, loss = 0.001644
grad_step = 000289, loss = 0.001591
grad_step = 000290, loss = 0.001548
grad_step = 000291, loss = 0.001540
grad_step = 000292, loss = 0.001563
grad_step = 000293, loss = 0.001585
grad_step = 000294, loss = 0.001569
grad_step = 000295, loss = 0.001548
grad_step = 000296, loss = 0.001526
grad_step = 000297, loss = 0.001523
grad_step = 000298, loss = 0.001534
grad_step = 000299, loss = 0.001546
grad_step = 000300, loss = 0.001560
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001551
grad_step = 000302, loss = 0.001536
grad_step = 000303, loss = 0.001516
grad_step = 000304, loss = 0.001505
grad_step = 000305, loss = 0.001506
grad_step = 000306, loss = 0.001514
grad_step = 000307, loss = 0.001526
grad_step = 000308, loss = 0.001533
grad_step = 000309, loss = 0.001547
grad_step = 000310, loss = 0.001545
grad_step = 000311, loss = 0.001548
grad_step = 000312, loss = 0.001526
grad_step = 000313, loss = 0.001506
grad_step = 000314, loss = 0.001486
grad_step = 000315, loss = 0.001481
grad_step = 000316, loss = 0.001487
grad_step = 000317, loss = 0.001496
grad_step = 000318, loss = 0.001502
grad_step = 000319, loss = 0.001498
grad_step = 000320, loss = 0.001494
grad_step = 000321, loss = 0.001483
grad_step = 000322, loss = 0.001473
grad_step = 000323, loss = 0.001464
grad_step = 000324, loss = 0.001459
grad_step = 000325, loss = 0.001457
grad_step = 000326, loss = 0.001457
grad_step = 000327, loss = 0.001460
grad_step = 000328, loss = 0.001464
grad_step = 000329, loss = 0.001473
grad_step = 000330, loss = 0.001487
grad_step = 000331, loss = 0.001521
grad_step = 000332, loss = 0.001548
grad_step = 000333, loss = 0.001609
grad_step = 000334, loss = 0.001573
grad_step = 000335, loss = 0.001523
grad_step = 000336, loss = 0.001446
grad_step = 000337, loss = 0.001445
grad_step = 000338, loss = 0.001494
grad_step = 000339, loss = 0.001494
grad_step = 000340, loss = 0.001461
grad_step = 000341, loss = 0.001424
grad_step = 000342, loss = 0.001434
grad_step = 000343, loss = 0.001469
grad_step = 000344, loss = 0.001468
grad_step = 000345, loss = 0.001453
grad_step = 000346, loss = 0.001419
grad_step = 000347, loss = 0.001408
grad_step = 000348, loss = 0.001422
grad_step = 000349, loss = 0.001432
grad_step = 000350, loss = 0.001431
grad_step = 000351, loss = 0.001413
grad_step = 000352, loss = 0.001398
grad_step = 000353, loss = 0.001391
grad_step = 000354, loss = 0.001396
grad_step = 000355, loss = 0.001406
grad_step = 000356, loss = 0.001413
grad_step = 000357, loss = 0.001417
grad_step = 000358, loss = 0.001411
grad_step = 000359, loss = 0.001404
grad_step = 000360, loss = 0.001392
grad_step = 000361, loss = 0.001381
grad_step = 000362, loss = 0.001370
grad_step = 000363, loss = 0.001363
grad_step = 000364, loss = 0.001359
grad_step = 000365, loss = 0.001359
grad_step = 000366, loss = 0.001362
grad_step = 000367, loss = 0.001366
grad_step = 000368, loss = 0.001374
grad_step = 000369, loss = 0.001385
grad_step = 000370, loss = 0.001413
grad_step = 000371, loss = 0.001444
grad_step = 000372, loss = 0.001496
grad_step = 000373, loss = 0.001480
grad_step = 000374, loss = 0.001438
grad_step = 000375, loss = 0.001363
grad_step = 000376, loss = 0.001342
grad_step = 000377, loss = 0.001366
grad_step = 000378, loss = 0.001381
grad_step = 000379, loss = 0.001379
grad_step = 000380, loss = 0.001350
grad_step = 000381, loss = 0.001329
grad_step = 000382, loss = 0.001322
grad_step = 000383, loss = 0.001332
grad_step = 000384, loss = 0.001350
grad_step = 000385, loss = 0.001340
grad_step = 000386, loss = 0.001317
grad_step = 000387, loss = 0.001296
grad_step = 000388, loss = 0.001298
grad_step = 000389, loss = 0.001312
grad_step = 000390, loss = 0.001315
grad_step = 000391, loss = 0.001307
grad_step = 000392, loss = 0.001292
grad_step = 000393, loss = 0.001286
grad_step = 000394, loss = 0.001285
grad_step = 000395, loss = 0.001284
grad_step = 000396, loss = 0.001277
grad_step = 000397, loss = 0.001267
grad_step = 000398, loss = 0.001262
grad_step = 000399, loss = 0.001263
grad_step = 000400, loss = 0.001266
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001267
grad_step = 000402, loss = 0.001267
grad_step = 000403, loss = 0.001268
grad_step = 000404, loss = 0.001281
grad_step = 000405, loss = 0.001304
grad_step = 000406, loss = 0.001357
grad_step = 000407, loss = 0.001397
grad_step = 000408, loss = 0.001481
grad_step = 000409, loss = 0.001439
grad_step = 000410, loss = 0.001385
grad_step = 000411, loss = 0.001262
grad_step = 000412, loss = 0.001238
grad_step = 000413, loss = 0.001306
grad_step = 000414, loss = 0.001319
grad_step = 000415, loss = 0.001277
grad_step = 000416, loss = 0.001223
grad_step = 000417, loss = 0.001242
grad_step = 000418, loss = 0.001290
grad_step = 000419, loss = 0.001263
grad_step = 000420, loss = 0.001225
grad_step = 000421, loss = 0.001214
grad_step = 000422, loss = 0.001229
grad_step = 000423, loss = 0.001234
grad_step = 000424, loss = 0.001219
grad_step = 000425, loss = 0.001206
grad_step = 000426, loss = 0.001201
grad_step = 000427, loss = 0.001202
grad_step = 000428, loss = 0.001211
grad_step = 000429, loss = 0.001208
grad_step = 000430, loss = 0.001195
grad_step = 000431, loss = 0.001180
grad_step = 000432, loss = 0.001177
grad_step = 000433, loss = 0.001182
grad_step = 000434, loss = 0.001185
grad_step = 000435, loss = 0.001182
grad_step = 000436, loss = 0.001176
grad_step = 000437, loss = 0.001172
grad_step = 000438, loss = 0.001170
grad_step = 000439, loss = 0.001168
grad_step = 000440, loss = 0.001162
grad_step = 000441, loss = 0.001155
grad_step = 000442, loss = 0.001151
grad_step = 000443, loss = 0.001150
grad_step = 000444, loss = 0.001149
grad_step = 000445, loss = 0.001147
grad_step = 000446, loss = 0.001144
grad_step = 000447, loss = 0.001142
grad_step = 000448, loss = 0.001144
grad_step = 000449, loss = 0.001151
grad_step = 000450, loss = 0.001171
grad_step = 000451, loss = 0.001208
grad_step = 000452, loss = 0.001294
grad_step = 000453, loss = 0.001377
grad_step = 000454, loss = 0.001500
grad_step = 000455, loss = 0.001362
grad_step = 000456, loss = 0.001196
grad_step = 000457, loss = 0.001156
grad_step = 000458, loss = 0.001234
grad_step = 000459, loss = 0.001237
grad_step = 000460, loss = 0.001180
grad_step = 000461, loss = 0.001180
grad_step = 000462, loss = 0.001161
grad_step = 000463, loss = 0.001144
grad_step = 000464, loss = 0.001195
grad_step = 000465, loss = 0.001188
grad_step = 000466, loss = 0.001125
grad_step = 000467, loss = 0.001123
grad_step = 000468, loss = 0.001141
grad_step = 000469, loss = 0.001131
grad_step = 000470, loss = 0.001130
grad_step = 000471, loss = 0.001128
grad_step = 000472, loss = 0.001107
grad_step = 000473, loss = 0.001101
grad_step = 000474, loss = 0.001099
grad_step = 000475, loss = 0.001105
grad_step = 000476, loss = 0.001106
grad_step = 000477, loss = 0.001086
grad_step = 000478, loss = 0.001080
grad_step = 000479, loss = 0.001089
grad_step = 000480, loss = 0.001082
grad_step = 000481, loss = 0.001076
grad_step = 000482, loss = 0.001081
grad_step = 000483, loss = 0.001072
grad_step = 000484, loss = 0.001058
grad_step = 000485, loss = 0.001062
grad_step = 000486, loss = 0.001066
grad_step = 000487, loss = 0.001057
grad_step = 000488, loss = 0.001055
grad_step = 000489, loss = 0.001056
grad_step = 000490, loss = 0.001050
grad_step = 000491, loss = 0.001044
grad_step = 000492, loss = 0.001043
grad_step = 000493, loss = 0.001039
grad_step = 000494, loss = 0.001035
grad_step = 000495, loss = 0.001035
grad_step = 000496, loss = 0.001033
grad_step = 000497, loss = 0.001030
grad_step = 000498, loss = 0.001029
grad_step = 000499, loss = 0.001032
grad_step = 000500, loss = 0.001032
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001036
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

  date_run                              2020-05-10 03:13:29.210545
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.29882
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 03:13:29.216296
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.279645
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 03:13:29.222949
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.142613
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 03:13:29.228039
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -3.2493
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
0   2020-05-10 03:13:01.952597  ...    mean_absolute_error
1   2020-05-10 03:13:01.956293  ...     mean_squared_error
2   2020-05-10 03:13:01.959646  ...  median_absolute_error
3   2020-05-10 03:13:01.962786  ...               r2_score
4   2020-05-10 03:13:10.611898  ...    mean_absolute_error
5   2020-05-10 03:13:10.615723  ...     mean_squared_error
6   2020-05-10 03:13:10.618863  ...  median_absolute_error
7   2020-05-10 03:13:10.621974  ...               r2_score
8   2020-05-10 03:13:29.210545  ...    mean_absolute_error
9   2020-05-10 03:13:29.216296  ...     mean_squared_error
10  2020-05-10 03:13:29.222949  ...  median_absolute_error
11  2020-05-10 03:13:29.228039  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 137611.24it/s] 39%|███▊      | 3833856/9912422 [00:00<00:30, 196275.52it/s]9920512it [00:00, 32759655.10it/s]                           
0it [00:00, ?it/s]32768it [00:00, 605876.99it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 444632.77it/s]1654784it [00:00, 11731551.04it/s]                         
0it [00:00, ?it/s]8192it [00:00, 176779.45it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb8157b69e8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b2efd9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb815772e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b2efdda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb8157bb6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb8157bbf60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b2eff080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7c817eb70> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b2efdda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7c817eb70> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b2eff048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa04e5d81d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=efc1b5745055f1a4102776eced641e1d6e7a7c45a4a07b1ceba67d3cd9a02bab
  Stored in directory: /tmp/pip-ephem-wheel-cache-o68kcts4/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa03e2dbe48> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  278528/17464789 [..............................] - ETA: 3s
  663552/17464789 [>.............................] - ETA: 2s
 1064960/17464789 [>.............................] - ETA: 2s
 1482752/17464789 [=>............................] - ETA: 2s
 2023424/17464789 [==>...........................] - ETA: 2s
 2531328/17464789 [===>..........................] - ETA: 1s
 3121152/17464789 [====>.........................] - ETA: 1s
 3760128/17464789 [=====>........................] - ETA: 1s
 4456448/17464789 [======>.......................] - ETA: 1s
 5226496/17464789 [=======>......................] - ETA: 1s
 6078464/17464789 [=========>....................] - ETA: 1s
 7004160/17464789 [===========>..................] - ETA: 0s
 7995392/17464789 [============>.................] - ETA: 0s
 9019392/17464789 [==============>...............] - ETA: 0s
10117120/17464789 [================>.............] - ETA: 0s
11198464/17464789 [==================>...........] - ETA: 0s
12296192/17464789 [====================>.........] - ETA: 0s
13508608/17464789 [======================>.......] - ETA: 0s
14794752/17464789 [========================>.....] - ETA: 0s
16015360/17464789 [==========================>...] - ETA: 0s
17309696/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 03:14:55.894303: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 03:14:55.898965: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-10 03:14:55.899102: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562c389fb3c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 03:14:55.899115: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.3906 - accuracy: 0.5180
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5133 - accuracy: 0.5100 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5593 - accuracy: 0.5070
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.4796 - accuracy: 0.5122
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5107 - accuracy: 0.5102
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5593 - accuracy: 0.5070
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5325 - accuracy: 0.5088
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6252 - accuracy: 0.5027
11000/25000 [============>.................] - ETA: 3s - loss: 7.6415 - accuracy: 0.5016
12000/25000 [=============>................] - ETA: 3s - loss: 7.6385 - accuracy: 0.5018
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6489 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6677 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6675 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6705 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6841 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6494 - accuracy: 0.5011
25000/25000 [==============================] - 7s 278us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 03:15:09.339228
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 03:15:09.339228  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 03:15:15.342427: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 03:15:15.347887: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-10 03:15:15.348555: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562eadc9adb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 03:15:15.348601: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1d13dc0d30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0989 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.9985 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1d0b2246a0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6002 - accuracy: 0.5043
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5133 - accuracy: 0.5100
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5363 - accuracy: 0.5085
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5615 - accuracy: 0.5069
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5689 - accuracy: 0.5064
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6087 - accuracy: 0.5038
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5884 - accuracy: 0.5051
11000/25000 [============>.................] - ETA: 3s - loss: 7.5969 - accuracy: 0.5045
12000/25000 [=============>................] - ETA: 3s - loss: 7.6117 - accuracy: 0.5036
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6798 - accuracy: 0.4991
15000/25000 [=================>............] - ETA: 2s - loss: 7.6983 - accuracy: 0.4979
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6982 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7036 - accuracy: 0.4976
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6947 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6858 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6952 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1cc4ec42b0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:09:54, 11.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:02:49, 15.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:35:10, 22.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:25:07, 32.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:10:48, 46.0kB/s].vector_cache/glove.6B.zip:   1%|          | 7.54M/862M [00:01<3:36:41, 65.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.6M/862M [00:01<2:30:40, 93.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.6M/862M [00:01<1:45:15, 134kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.3M/862M [00:01<1:13:16, 191kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.0M/862M [00:01<51:01, 272kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.9M/862M [00:01<35:44, 388kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:02<24:58, 552kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<17:30, 783kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.4M/862M [00:02<12:15, 1.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<08:40, 1.57MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.7M/862M [00:02<06:51, 1.98MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.9M/862M [00:03<05:23, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<05:40, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:05<06:14, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:05<04:49, 2.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:05<03:33, 3.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<10:55, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<09:21, 1.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:07<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<07:28, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<06:41, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:09<05:02, 2.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<06:25, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<07:19, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<05:48, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:11<04:11, 3.13MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<34:38, 379kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<25:36, 513kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:13<18:13, 719kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<15:45, 829kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<13:41, 954kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<10:08, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:15<07:14, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<13:09, 988kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<10:32, 1.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:17<07:41, 1.69MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<08:24, 1.54MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<08:30, 1.52MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:19<06:30, 1.98MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:19<04:42, 2.73MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<10:50, 1.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<08:53, 1.45MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:21<06:32, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<07:33, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<07:54, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:23<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<04:25, 2.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<09:19, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<07:50, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<05:48, 2.19MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<04:30, 2.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<14:25:55, 14.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<10:08:26, 20.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<7:06:02, 29.6kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<4:59:06, 42.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<3:30:38, 59.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<2:27:30, 85.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<1:45:40, 118kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<1:16:36, 163kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<54:15, 230kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<37:58, 328kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<45:18, 274kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:33<32:59, 377kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<23:21, 531kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<19:11, 645kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<14:30, 852kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<10:27, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<10:13, 1.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<08:24, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:10, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<07:11, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<06:16, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:38, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<06:08, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<05:32, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<04:10, 2.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:46, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:16, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<03:59, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:37, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:22, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:04, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<03:39, 3.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<20:28, 584kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<15:39, 763kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<11:12, 1.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<07:58, 1.49MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<1:35:03, 125kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<1:09:09, 172kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<49:02, 242kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<34:20, 344kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<36:36, 323kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<26:55, 439kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<19:06, 617kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<15:52, 740kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<13:45, 854kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<10:16, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<07:19, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<19:15, 607kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<14:47, 789kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<10:39, 1.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<09:57, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:16, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:04, 1.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<06:43, 1.72MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<07:15, 1.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:37, 2.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<04:05, 2.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<07:45, 1.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<06:41, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:00, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:58, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<06:41, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:50, 2.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<14:49, 764kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<11:37, 973kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:26, 1.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<08:20, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:04, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:14, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:03, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:28, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:05, 2.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:16, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:09, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:55, 2.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<03:35, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<15:53, 695kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<12:20, 895kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:55, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:37, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:28, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:32, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:41, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<14:56, 731kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<11:40, 935kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:24, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<08:12, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<08:08, 1.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:17, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:31, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<14:17, 754kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<11:12, 961kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:07, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<07:59, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<07:58, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:05, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:24, 2.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<07:13, 1.47MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:14, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:36, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:32, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<05:02, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:46, 2.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:56, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:47, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:32, 2.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:19, 3.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:06, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:25, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:02, 2.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:06, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:51, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:23, 3.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<13:21, 770kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<10:29, 980kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<07:36, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<07:32, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<06:24, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:45, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:32, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:00, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:46, 2.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:50, 2.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:30, 2.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:25, 2.93MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:34, 2.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:25, 1.84MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<04:21, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:08, 3.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<09:34, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:47, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<05:43, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:28, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:05, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:39, 2.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<11:53, 825kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<09:23, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<06:48, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:52, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:59, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:19, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:52, 2.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<06:16, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:27, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:02, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:54, 1.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:28, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:17, 2.23MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<03:06, 3.07MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<07:53, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:33, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:50, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:29, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:57, 1.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:41, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<03:22, 2.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<12:24, 756kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<09:42, 966kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<07:01, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:58, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:57, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:18, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<03:50, 2.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:32, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:07, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:53, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:27, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:19, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:07, 2.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<11:56, 763kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<09:20, 974kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<06:43, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:42, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:41, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:13, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:53, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:22, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:17, 2.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:20, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:01, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:03, 2.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:02, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:43, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:41, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<02:42, 3.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:28, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:39, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:30, 2.49MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:19, 2.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:57, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:57, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:51, 3.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<08:18, 1.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:46, 1.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:08, 1.67MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<03:15, 2.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:41, 3.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:51, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<09:40, 884kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:53, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:52, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<03:43, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:55, 2.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:17, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:33, 991kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:58, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<05:13, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:00, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<03:08, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:37, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:39, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:18, 1.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:45, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<03:39, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<02:53, 2.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:46, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:45, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:18, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:37, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:50, 2.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:55, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<08:12, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:35, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:53, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:46, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:54, 2.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:33, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:38, 950kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:45, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:04, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:51, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:59, 2.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<07:49, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<08:46, 928kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:57, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:10, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:54, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<03:03, 2.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:04, 999kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:58, 899kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:03, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:12, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<03:56, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:02, 2.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<08:30, 941kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<08:56, 894kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:55, 1.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:13, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<03:00, 2.65MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<08:25, 942kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<08:52, 894kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<06:59, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:10, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:54, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:01, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:35, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<08:15, 951kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:25, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:52, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:38, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:49, 2.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<08:12, 950kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<08:53, 876kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:51, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:06, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:51, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:58, 2.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<08:55, 866kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<09:22, 824kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<07:19, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:22, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<04:01, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:05, 2.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<09:27, 809kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<09:44, 785kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:34, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:35, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<04:08, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<03:09, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<10:33, 719kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<10:13, 741kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:45, 976kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:42, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<04:14, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<15:18, 491kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<13:32, 555kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<10:01, 749kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<07:18, 1.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<05:21, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<03:57, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<10:12, 729kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<09:46, 762kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:23, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:26, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<04:02, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:22, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:57, 824kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:30, 982kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:35, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:05, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:01, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<10:22, 705kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<09:28, 771kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:07, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:13, 1.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<03:51, 1.88MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:05, 1.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<08:42, 831kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<07:11, 1.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:16, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:52, 1.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:52, 2.50MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<1:01:30, 117kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<46:56, 153kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<33:49, 212kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<23:55, 299kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<16:50, 423kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<14:16, 497kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<13:51, 512kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<10:27, 678kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:34, 935kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:26, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:49, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<08:12, 857kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:35, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:50, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<03:30, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<06:32, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<07:40, 907kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<06:03, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:28, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:15, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<06:40, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<07:43, 892kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<06:07, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:27, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<03:14, 2.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<09:52, 691kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<09:41, 704kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<07:28, 913kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:23, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<03:52, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<26:50, 252kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<21:20, 317kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<15:33, 434kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<11:02, 609kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<09:09, 731kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<08:43, 766kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<06:35, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:46, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:53, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:34, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<04:20, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<03:11, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:54, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:36, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:37, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:40, 2.44MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:44, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:27, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:29, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:33, 2.51MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:49, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:32, 2.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:04, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:33, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:34, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:37, 2.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:56, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:12, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<03:17, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:23, 2.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:50, 1.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:04, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:54, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:48, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:13, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:31, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:30, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:34, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:13, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:11, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:11, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:20, 2.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:03, 1.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:05, 983kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:54, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:37, 1.65MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<02:39, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:12, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:33, 903kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:16, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:50, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:48, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:41, 876kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<07:18, 802kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:45, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<04:11, 1.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<03:01, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:24, 689kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:26, 685kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:31, 887kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<04:41, 1.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<03:22, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<08:52, 644kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<08:45, 653kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:43, 849kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<04:49, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<03:28, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<10:38, 531kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<09:44, 580kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<07:23, 764kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<05:17, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<03:47, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<11:42, 477kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<10:27, 533kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<07:51, 709kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<05:38, 984kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<04:01, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<25:14, 218kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<19:43, 279kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<14:16, 386kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<10:06, 543kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<07:08, 763kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<15:03, 362kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<12:34, 433kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<09:19, 583kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<06:39, 814kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<04:43, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<29:08, 184kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<22:24, 240kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<16:10, 332kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<11:22, 469kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<08:00, 663kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<15:39, 339kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<12:57, 410kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<09:27, 560kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:46, 780kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<04:49, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:56, 882kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:07, 855kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:45, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:27, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:32, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<04:24, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:29, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:34, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:54, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<03:27, 1.47MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<04:20, 1.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:26, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:32, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<02:57, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:56, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:13, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<02:47, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:48, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:02, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:15, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:40, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:34, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:56, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:09, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:35, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:37, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<02:57, 1.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:10, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:35, 1.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:28, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:05, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:32, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:25, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:49, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:04, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:29, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:27, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:46, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:03, 2.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:31, 2.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:05, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:44, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:01, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:12, 2.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:30, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:51, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:04, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:31, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<13:09, 335kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<10:44, 410kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<07:54, 556kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<05:36, 779kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<04:53, 886kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:05, 853kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:56, 1.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:50, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<02:02, 2.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<25:14, 169kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<19:09, 223kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<13:42, 311kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<09:40, 439kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<06:47, 619kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<07:03, 595kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<06:23, 656kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:51, 862kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:28, 1.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<03:23, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:42, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:54, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:06, 1.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:33, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:30, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:52, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:03, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:12, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:36, 2.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<06:00, 664kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<05:35, 713kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:16, 932kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:04, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<03:04, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:31, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:44, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:58, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:28, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:10, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:28, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:41, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:57, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:18, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:46, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:14, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:38, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:05, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:35, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:03, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:29, 2.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:08, 3.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:06, 886kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<03:59, 914kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:01, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:10, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:31, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:47, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:10, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:33, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:09, 3.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<06:09, 569kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<05:18, 659kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:54, 893kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:46, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:59, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<13:29, 255kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<10:25, 329kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<07:28, 458kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<05:15, 645kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:44, 711kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:16, 786kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<03:11, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:16, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:38, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<06:36, 499kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<05:31, 597kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<04:02, 814kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:51, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<03:06, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:00, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:18, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:39, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:20, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:24, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:50, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:19, 2.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:04, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:09, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:40, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:12, 2.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:30, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:25, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:49, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:17, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:30, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:20, 1.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:49, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:18, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:09, 1.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:05, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:36, 1.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:08, 2.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:30, 802kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:01, 929kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<02:14, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<01:34, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:24, 622kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:36, 761kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:38, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:51, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<06:13, 430kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<05:02, 530kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:40, 723kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<02:34, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:13, 808kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:53, 901kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<02:09, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:31, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:56, 863kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:28, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:49, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:41, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:16, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:51, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:20, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:26, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:22, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:01, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<00:43, 3.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<06:17, 370kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<04:53, 475kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:30, 659kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<02:26, 931kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<04:07, 548kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:21, 671kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:26, 920kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:43, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:01, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:43, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:15, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:16, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:18, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:00, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:01, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:06, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:51, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:00, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:48, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:51, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:58, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:46, 2.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:49, 2.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:55, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:43, 2.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:47, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:53, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:41, 2.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:29, 3.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<05:09, 332kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:55, 435kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:47, 607kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:56, 855kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:01, 813kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:01, 810kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:32, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:06, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:06, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:04, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:49, 1.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:48, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:50, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:39, 2.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:44, 1.92MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:35, 2.42MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:37, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:41, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:32, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:34, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:39, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:30, 2.48MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:33, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:36, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:28, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:19, 3.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<03:50, 302kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<03:07, 371kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:17, 504kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:35, 709kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:20, 817kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:08, 957kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:50, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:43, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:41, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:31, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:44, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:36, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:26, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:30, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:31, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:23, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:19, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:13, 3.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<04:10, 179kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<03:13, 232kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<02:18, 321kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<01:34, 453kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:12, 561kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:58, 693kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:42, 945kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:33, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:30, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:22, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:19, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:25, 1.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:14, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<00:06, 3.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:26, 766kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:21, 907kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:15, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:09, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:34, 459kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:30, 518kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:22, 697kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:14, 972kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 810kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:09, 787kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:03, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.83MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 790/400000 [00:00<00:50, 7899.84it/s]  0%|          | 1668/400000 [00:00<00:48, 8143.40it/s]  1%|          | 2553/400000 [00:00<00:47, 8341.34it/s]  1%|          | 3444/400000 [00:00<00:46, 8501.99it/s]  1%|          | 4310/400000 [00:00<00:46, 8548.78it/s]  1%|▏         | 5145/400000 [00:00<00:46, 8485.31it/s]  2%|▏         | 6016/400000 [00:00<00:46, 8551.48it/s]  2%|▏         | 6890/400000 [00:00<00:45, 8604.75it/s]  2%|▏         | 7770/400000 [00:00<00:45, 8661.29it/s]  2%|▏         | 8612/400000 [00:01<00:45, 8585.18it/s]  2%|▏         | 9502/400000 [00:01<00:45, 8677.22it/s]  3%|▎         | 10362/400000 [00:01<00:45, 8653.24it/s]  3%|▎         | 11247/400000 [00:01<00:44, 8710.02it/s]  3%|▎         | 12114/400000 [00:01<00:44, 8696.12it/s]  3%|▎         | 12979/400000 [00:01<00:44, 8671.01it/s]  3%|▎         | 13872/400000 [00:01<00:44, 8745.11it/s]  4%|▎         | 14745/400000 [00:01<00:44, 8701.73it/s]  4%|▍         | 15642/400000 [00:01<00:43, 8779.79it/s]  4%|▍         | 16525/400000 [00:01<00:43, 8794.43it/s]  4%|▍         | 17419/400000 [00:02<00:43, 8837.29it/s]  5%|▍         | 18305/400000 [00:02<00:43, 8842.43it/s]  5%|▍         | 19191/400000 [00:02<00:43, 8846.76it/s]  5%|▌         | 20092/400000 [00:02<00:42, 8892.64it/s]  5%|▌         | 20982/400000 [00:02<00:42, 8870.40it/s]  5%|▌         | 21869/400000 [00:02<00:43, 8610.34it/s]  6%|▌         | 22737/400000 [00:02<00:43, 8628.92it/s]  6%|▌         | 23602/400000 [00:02<00:43, 8634.26it/s]  6%|▌         | 24480/400000 [00:02<00:43, 8676.69it/s]  6%|▋         | 25374/400000 [00:02<00:42, 8753.42it/s]  7%|▋         | 26250/400000 [00:03<00:42, 8722.03it/s]  7%|▋         | 27123/400000 [00:03<00:42, 8699.65it/s]  7%|▋         | 28065/400000 [00:03<00:41, 8902.11it/s]  7%|▋         | 28957/400000 [00:03<00:41, 8887.12it/s]  7%|▋         | 29847/400000 [00:03<00:41, 8851.37it/s]  8%|▊         | 30733/400000 [00:03<00:41, 8835.57it/s]  8%|▊         | 31623/400000 [00:03<00:41, 8852.55it/s]  8%|▊         | 32509/400000 [00:03<00:41, 8843.60it/s]  8%|▊         | 33396/400000 [00:03<00:41, 8848.64it/s]  9%|▊         | 34282/400000 [00:03<00:41, 8818.96it/s]  9%|▉         | 35165/400000 [00:04<00:41, 8783.37it/s]  9%|▉         | 36044/400000 [00:04<00:41, 8677.87it/s]  9%|▉         | 36953/400000 [00:04<00:41, 8795.39it/s]  9%|▉         | 37834/400000 [00:04<00:41, 8624.68it/s] 10%|▉         | 38723/400000 [00:04<00:41, 8701.01it/s] 10%|▉         | 39595/400000 [00:04<00:41, 8588.28it/s] 10%|█         | 40503/400000 [00:04<00:41, 8728.97it/s] 10%|█         | 41378/400000 [00:04<00:41, 8625.70it/s] 11%|█         | 42242/400000 [00:04<00:41, 8591.38it/s] 11%|█         | 43147/400000 [00:04<00:40, 8722.81it/s] 11%|█         | 44021/400000 [00:05<00:41, 8648.43it/s] 11%|█         | 44912/400000 [00:05<00:40, 8723.71it/s] 11%|█▏        | 45835/400000 [00:05<00:39, 8867.70it/s] 12%|█▏        | 46743/400000 [00:05<00:39, 8928.92it/s] 12%|█▏        | 47639/400000 [00:05<00:39, 8937.69it/s] 12%|█▏        | 48543/400000 [00:05<00:39, 8967.99it/s] 12%|█▏        | 49441/400000 [00:05<00:39, 8925.47it/s] 13%|█▎        | 50334/400000 [00:05<00:39, 8898.67it/s] 13%|█▎        | 51225/400000 [00:05<00:39, 8899.24it/s] 13%|█▎        | 52116/400000 [00:05<00:39, 8824.95it/s] 13%|█▎        | 53003/400000 [00:06<00:39, 8836.76it/s] 13%|█▎        | 53912/400000 [00:06<00:38, 8910.96it/s] 14%|█▎        | 54804/400000 [00:06<00:39, 8766.90it/s] 14%|█▍        | 55699/400000 [00:06<00:39, 8820.01it/s] 14%|█▍        | 56582/400000 [00:06<00:39, 8775.78it/s] 14%|█▍        | 57471/400000 [00:06<00:38, 8807.48it/s] 15%|█▍        | 58353/400000 [00:06<00:39, 8757.49it/s] 15%|█▍        | 59236/400000 [00:06<00:38, 8778.43it/s] 15%|█▌        | 60115/400000 [00:06<00:38, 8781.73it/s] 15%|█▌        | 60997/400000 [00:06<00:38, 8792.58it/s] 15%|█▌        | 61877/400000 [00:07<00:38, 8711.45it/s] 16%|█▌        | 62758/400000 [00:07<00:38, 8739.25it/s] 16%|█▌        | 63633/400000 [00:07<00:38, 8739.76it/s] 16%|█▌        | 64511/400000 [00:07<00:38, 8751.22it/s] 16%|█▋        | 65389/400000 [00:07<00:38, 8759.32it/s] 17%|█▋        | 66266/400000 [00:07<00:38, 8755.92it/s] 17%|█▋        | 67219/400000 [00:07<00:37, 8972.81it/s] 17%|█▋        | 68118/400000 [00:07<00:37, 8945.42it/s] 17%|█▋        | 69022/400000 [00:07<00:36, 8973.46it/s] 17%|█▋        | 69921/400000 [00:07<00:36, 8974.53it/s] 18%|█▊        | 70819/400000 [00:08<00:36, 8899.05it/s] 18%|█▊        | 71710/400000 [00:08<00:37, 8793.04it/s] 18%|█▊        | 72590/400000 [00:08<00:37, 8769.37it/s] 18%|█▊        | 73468/400000 [00:08<00:37, 8740.08it/s] 19%|█▊        | 74366/400000 [00:08<00:36, 8810.64it/s] 19%|█▉        | 75285/400000 [00:08<00:36, 8918.59it/s] 19%|█▉        | 76191/400000 [00:08<00:36, 8959.22it/s] 19%|█▉        | 77088/400000 [00:08<00:36, 8925.36it/s] 19%|█▉        | 77981/400000 [00:08<00:36, 8897.43it/s] 20%|█▉        | 78872/400000 [00:08<00:36, 8856.73it/s] 20%|█▉        | 79758/400000 [00:09<00:36, 8817.27it/s] 20%|██        | 80640/400000 [00:09<00:36, 8763.79it/s] 20%|██        | 81519/400000 [00:09<00:36, 8771.50it/s] 21%|██        | 82404/400000 [00:09<00:36, 8792.39it/s] 21%|██        | 83306/400000 [00:09<00:35, 8857.86it/s] 21%|██        | 84217/400000 [00:09<00:35, 8930.21it/s] 21%|██▏       | 85111/400000 [00:09<00:35, 8898.91it/s] 22%|██▏       | 86002/400000 [00:09<00:35, 8891.91it/s] 22%|██▏       | 86912/400000 [00:09<00:34, 8950.77it/s] 22%|██▏       | 87808/400000 [00:09<00:35, 8860.37it/s] 22%|██▏       | 88697/400000 [00:10<00:35, 8866.88it/s] 22%|██▏       | 89608/400000 [00:10<00:34, 8938.16it/s] 23%|██▎       | 90519/400000 [00:10<00:34, 8987.30it/s] 23%|██▎       | 91419/400000 [00:10<00:34, 8937.10it/s] 23%|██▎       | 92313/400000 [00:10<00:34, 8897.13it/s] 23%|██▎       | 93203/400000 [00:10<00:34, 8831.88it/s] 24%|██▎       | 94098/400000 [00:10<00:34, 8865.82it/s] 24%|██▍       | 95011/400000 [00:10<00:34, 8941.90it/s] 24%|██▍       | 95925/400000 [00:10<00:33, 8998.80it/s] 24%|██▍       | 96842/400000 [00:10<00:33, 9049.33it/s] 24%|██▍       | 97765/400000 [00:11<00:33, 9102.39it/s] 25%|██▍       | 98676/400000 [00:11<00:33, 8915.61it/s] 25%|██▍       | 99569/400000 [00:11<00:33, 8874.19it/s] 25%|██▌       | 100458/400000 [00:11<00:33, 8861.20it/s] 25%|██▌       | 101345/400000 [00:11<00:35, 8494.34it/s] 26%|██▌       | 102214/400000 [00:11<00:34, 8551.75it/s] 26%|██▌       | 103116/400000 [00:11<00:34, 8684.97it/s] 26%|██▌       | 103993/400000 [00:11<00:33, 8707.67it/s] 26%|██▌       | 104871/400000 [00:11<00:33, 8728.21it/s] 26%|██▋       | 105745/400000 [00:12<00:34, 8630.50it/s] 27%|██▋       | 106621/400000 [00:12<00:33, 8667.86it/s] 27%|██▋       | 107529/400000 [00:12<00:33, 8786.70it/s] 27%|██▋       | 108409/400000 [00:12<00:33, 8702.58it/s] 27%|██▋       | 109285/400000 [00:12<00:33, 8716.46it/s] 28%|██▊       | 110165/400000 [00:12<00:33, 8736.41it/s] 28%|██▊       | 111040/400000 [00:12<00:33, 8687.33it/s] 28%|██▊       | 111929/400000 [00:12<00:32, 8746.15it/s] 28%|██▊       | 112825/400000 [00:12<00:32, 8808.87it/s] 28%|██▊       | 113708/400000 [00:12<00:32, 8813.94it/s] 29%|██▊       | 114603/400000 [00:13<00:32, 8852.99it/s] 29%|██▉       | 115489/400000 [00:13<00:33, 8512.90it/s] 29%|██▉       | 116365/400000 [00:13<00:33, 8583.45it/s] 29%|██▉       | 117240/400000 [00:13<00:32, 8630.98it/s] 30%|██▉       | 118134/400000 [00:13<00:32, 8720.82it/s] 30%|██▉       | 119022/400000 [00:13<00:32, 8766.35it/s] 30%|██▉       | 119900/400000 [00:13<00:32, 8737.98it/s] 30%|███       | 120775/400000 [00:13<00:32, 8688.80it/s] 30%|███       | 121672/400000 [00:13<00:31, 8770.17it/s] 31%|███       | 122563/400000 [00:13<00:31, 8810.42it/s] 31%|███       | 123445/400000 [00:14<00:31, 8773.18it/s] 31%|███       | 124323/400000 [00:14<00:31, 8681.23it/s] 31%|███▏      | 125213/400000 [00:14<00:31, 8745.16it/s] 32%|███▏      | 126092/400000 [00:14<00:31, 8756.36it/s] 32%|███▏      | 126994/400000 [00:14<00:30, 8833.15it/s] 32%|███▏      | 127878/400000 [00:14<00:30, 8812.94it/s] 32%|███▏      | 128760/400000 [00:14<00:31, 8707.31it/s] 32%|███▏      | 129632/400000 [00:14<00:31, 8672.02it/s] 33%|███▎      | 130508/400000 [00:14<00:30, 8697.86it/s] 33%|███▎      | 131405/400000 [00:14<00:30, 8777.70it/s] 33%|███▎      | 132296/400000 [00:15<00:30, 8815.64it/s] 33%|███▎      | 133178/400000 [00:15<00:30, 8732.98it/s] 34%|███▎      | 134074/400000 [00:15<00:30, 8797.70it/s] 34%|███▎      | 134975/400000 [00:15<00:29, 8859.89it/s] 34%|███▍      | 135862/400000 [00:15<00:29, 8854.30it/s] 34%|███▍      | 136748/400000 [00:15<00:29, 8775.46it/s] 34%|███▍      | 137626/400000 [00:15<00:29, 8749.93it/s] 35%|███▍      | 138528/400000 [00:15<00:29, 8828.38it/s] 35%|███▍      | 139437/400000 [00:15<00:29, 8902.79it/s] 35%|███▌      | 140333/400000 [00:15<00:29, 8917.56it/s] 35%|███▌      | 141244/400000 [00:16<00:28, 8972.72it/s] 36%|███▌      | 142142/400000 [00:16<00:29, 8849.30it/s] 36%|███▌      | 143028/400000 [00:16<00:29, 8838.78it/s] 36%|███▌      | 143913/400000 [00:16<00:29, 8818.07it/s] 36%|███▌      | 144796/400000 [00:16<00:28, 8819.89it/s] 36%|███▋      | 145679/400000 [00:16<00:29, 8578.76it/s] 37%|███▋      | 146539/400000 [00:16<00:29, 8582.52it/s] 37%|███▋      | 147414/400000 [00:16<00:29, 8629.03it/s] 37%|███▋      | 148297/400000 [00:16<00:28, 8686.52it/s] 37%|███▋      | 149191/400000 [00:16<00:28, 8759.48it/s] 38%|███▊      | 150068/400000 [00:17<00:28, 8738.59it/s] 38%|███▊      | 150946/400000 [00:17<00:28, 8750.04it/s] 38%|███▊      | 151831/400000 [00:17<00:28, 8777.71it/s] 38%|███▊      | 152721/400000 [00:17<00:28, 8813.66it/s] 38%|███▊      | 153603/400000 [00:17<00:28, 8797.92it/s] 39%|███▊      | 154496/400000 [00:17<00:27, 8836.85it/s] 39%|███▉      | 155380/400000 [00:17<00:27, 8794.89it/s] 39%|███▉      | 156286/400000 [00:17<00:27, 8871.56it/s] 39%|███▉      | 157199/400000 [00:17<00:27, 8947.42it/s] 40%|███▉      | 158122/400000 [00:17<00:26, 9028.70it/s] 40%|███▉      | 159026/400000 [00:18<00:26, 9031.91it/s] 40%|███▉      | 159930/400000 [00:18<00:26, 9020.29it/s] 40%|████      | 160833/400000 [00:18<00:26, 9009.87it/s] 40%|████      | 161744/400000 [00:18<00:26, 9039.46it/s] 41%|████      | 162649/400000 [00:18<00:26, 8989.76it/s] 41%|████      | 163549/400000 [00:18<00:26, 8986.58it/s] 41%|████      | 164460/400000 [00:18<00:26, 9022.26it/s] 41%|████▏     | 165363/400000 [00:18<00:26, 8928.51it/s] 42%|████▏     | 166279/400000 [00:18<00:25, 8994.95it/s] 42%|████▏     | 167185/400000 [00:18<00:25, 9013.11it/s] 42%|████▏     | 168087/400000 [00:19<00:25, 9002.08it/s] 42%|████▏     | 168988/400000 [00:19<00:25, 8974.18it/s] 42%|████▏     | 169894/400000 [00:19<00:25, 8997.92it/s] 43%|████▎     | 170838/400000 [00:19<00:25, 9124.89it/s] 43%|████▎     | 171751/400000 [00:19<00:25, 9073.02it/s] 43%|████▎     | 172659/400000 [00:19<00:25, 8926.92it/s] 43%|████▎     | 173553/400000 [00:19<00:25, 8836.50it/s] 44%|████▎     | 174460/400000 [00:19<00:25, 8903.26it/s] 44%|████▍     | 175404/400000 [00:19<00:24, 9055.12it/s] 44%|████▍     | 176311/400000 [00:20<00:25, 8904.05it/s] 44%|████▍     | 177203/400000 [00:20<00:25, 8751.50it/s] 45%|████▍     | 178104/400000 [00:20<00:25, 8824.68it/s] 45%|████▍     | 178998/400000 [00:20<00:24, 8858.28it/s] 45%|████▍     | 179929/400000 [00:20<00:24, 8986.48it/s] 45%|████▌     | 180829/400000 [00:20<00:24, 8984.22it/s] 45%|████▌     | 181730/400000 [00:20<00:24, 8989.47it/s] 46%|████▌     | 182632/400000 [00:20<00:24, 8998.02it/s] 46%|████▌     | 183568/400000 [00:20<00:23, 9103.60it/s] 46%|████▌     | 184479/400000 [00:20<00:23, 9015.46it/s] 46%|████▋     | 185402/400000 [00:21<00:23, 9078.38it/s] 47%|████▋     | 186311/400000 [00:21<00:23, 8998.64it/s] 47%|████▋     | 187212/400000 [00:21<00:23, 8912.99it/s] 47%|████▋     | 188122/400000 [00:21<00:23, 8966.95it/s] 47%|████▋     | 189026/400000 [00:21<00:23, 8987.37it/s] 47%|████▋     | 189926/400000 [00:21<00:24, 8704.99it/s] 48%|████▊     | 190808/400000 [00:21<00:23, 8737.01it/s] 48%|████▊     | 191684/400000 [00:21<00:23, 8730.17it/s] 48%|████▊     | 192573/400000 [00:21<00:23, 8772.44it/s] 48%|████▊     | 193452/400000 [00:21<00:23, 8777.52it/s] 49%|████▊     | 194345/400000 [00:22<00:23, 8822.36it/s] 49%|████▉     | 195243/400000 [00:22<00:23, 8867.06it/s] 49%|████▉     | 196131/400000 [00:22<00:22, 8870.83it/s] 49%|████▉     | 197019/400000 [00:22<00:22, 8860.72it/s] 49%|████▉     | 197931/400000 [00:22<00:22, 8934.55it/s] 50%|████▉     | 198849/400000 [00:22<00:22, 9006.47it/s] 50%|████▉     | 199755/400000 [00:22<00:22, 9022.35it/s] 50%|█████     | 200664/400000 [00:22<00:22, 9040.10it/s] 50%|█████     | 201569/400000 [00:22<00:22, 8978.78it/s] 51%|█████     | 202468/400000 [00:22<00:22, 8911.81it/s] 51%|█████     | 203360/400000 [00:23<00:22, 8841.40it/s] 51%|█████     | 204259/400000 [00:23<00:22, 8883.72it/s] 51%|█████▏    | 205169/400000 [00:23<00:21, 8944.49it/s] 52%|█████▏    | 206064/400000 [00:23<00:21, 8853.99it/s] 52%|█████▏    | 206950/400000 [00:23<00:21, 8826.27it/s] 52%|█████▏    | 207845/400000 [00:23<00:21, 8861.21it/s] 52%|█████▏    | 208743/400000 [00:23<00:21, 8894.24it/s] 52%|█████▏    | 209641/400000 [00:23<00:21, 8919.60it/s] 53%|█████▎    | 210551/400000 [00:23<00:21, 8972.76it/s] 53%|█████▎    | 211452/400000 [00:23<00:20, 8982.69it/s] 53%|█████▎    | 212351/400000 [00:24<00:21, 8928.86it/s] 53%|█████▎    | 213245/400000 [00:24<00:20, 8906.19it/s] 54%|█████▎    | 214138/400000 [00:24<00:20, 8911.46it/s] 54%|█████▍    | 215051/400000 [00:24<00:20, 8973.28it/s] 54%|█████▍    | 216010/400000 [00:24<00:20, 9147.03it/s] 54%|█████▍    | 216926/400000 [00:24<00:20, 9121.62it/s] 54%|█████▍    | 217839/400000 [00:24<00:20, 9042.30it/s] 55%|█████▍    | 218768/400000 [00:24<00:19, 9113.02it/s] 55%|█████▍    | 219680/400000 [00:24<00:19, 9057.63it/s] 55%|█████▌    | 220587/400000 [00:24<00:19, 9036.25it/s] 55%|█████▌    | 221491/400000 [00:25<00:19, 8961.02it/s] 56%|█████▌    | 222397/400000 [00:25<00:19, 8990.24it/s] 56%|█████▌    | 223297/400000 [00:25<00:19, 8939.66it/s] 56%|█████▌    | 224192/400000 [00:25<00:19, 8848.70it/s] 56%|█████▋    | 225082/400000 [00:25<00:19, 8863.27it/s] 56%|█████▋    | 225970/400000 [00:25<00:19, 8865.42it/s] 57%|█████▋    | 226878/400000 [00:25<00:19, 8928.14it/s] 57%|█████▋    | 227772/400000 [00:25<00:19, 8821.63it/s] 57%|█████▋    | 228659/400000 [00:25<00:19, 8835.13it/s] 57%|█████▋    | 229564/400000 [00:25<00:19, 8896.52it/s] 58%|█████▊    | 230482/400000 [00:26<00:18, 8977.25it/s] 58%|█████▊    | 231391/400000 [00:26<00:18, 9009.51it/s] 58%|█████▊    | 232304/400000 [00:26<00:18, 9042.47it/s] 58%|█████▊    | 233209/400000 [00:26<00:18, 8983.49it/s] 59%|█████▊    | 234131/400000 [00:26<00:18, 9051.42it/s] 59%|█████▉    | 235049/400000 [00:26<00:18, 9087.43it/s] 59%|█████▉    | 235958/400000 [00:26<00:18, 9025.16it/s] 59%|█████▉    | 236861/400000 [00:26<00:18, 8954.15it/s] 59%|█████▉    | 237757/400000 [00:26<00:18, 8943.45it/s] 60%|█████▉    | 238689/400000 [00:26<00:17, 9051.53it/s] 60%|█████▉    | 239595/400000 [00:27<00:17, 9010.82it/s] 60%|██████    | 240535/400000 [00:27<00:17, 9122.60it/s] 60%|██████    | 241448/400000 [00:27<00:17, 9087.34it/s] 61%|██████    | 242368/400000 [00:27<00:17, 9118.95it/s] 61%|██████    | 243293/400000 [00:27<00:17, 9156.51it/s] 61%|██████    | 244209/400000 [00:27<00:17, 9038.88it/s] 61%|██████▏   | 245114/400000 [00:27<00:17, 8960.14it/s] 62%|██████▏   | 246011/400000 [00:27<00:17, 8888.42it/s] 62%|██████▏   | 246901/400000 [00:27<00:17, 8822.90it/s] 62%|██████▏   | 247784/400000 [00:28<00:17, 8748.06it/s] 62%|██████▏   | 248666/400000 [00:28<00:17, 8767.72it/s] 62%|██████▏   | 249544/400000 [00:28<00:17, 8733.18it/s] 63%|██████▎   | 250418/400000 [00:28<00:17, 8726.05it/s] 63%|██████▎   | 251291/400000 [00:28<00:17, 8700.11it/s] 63%|██████▎   | 252192/400000 [00:28<00:16, 8788.10it/s] 63%|██████▎   | 253100/400000 [00:28<00:16, 8871.30it/s] 63%|██████▎   | 253988/400000 [00:28<00:16, 8846.43it/s] 64%|██████▎   | 254880/400000 [00:28<00:16, 8867.48it/s] 64%|██████▍   | 255767/400000 [00:28<00:16, 8840.92it/s] 64%|██████▍   | 256664/400000 [00:29<00:16, 8878.95it/s] 64%|██████▍   | 257553/400000 [00:29<00:16, 8830.82it/s] 65%|██████▍   | 258437/400000 [00:29<00:16, 8805.15it/s] 65%|██████▍   | 259321/400000 [00:29<00:15, 8813.28it/s] 65%|██████▌   | 260206/400000 [00:29<00:15, 8823.12it/s] 65%|██████▌   | 261108/400000 [00:29<00:15, 8881.18it/s] 66%|██████▌   | 262001/400000 [00:29<00:15, 8894.98it/s] 66%|██████▌   | 262891/400000 [00:29<00:15, 8860.10it/s] 66%|██████▌   | 263800/400000 [00:29<00:15, 8925.73it/s] 66%|██████▌   | 264693/400000 [00:29<00:15, 8881.01it/s] 66%|██████▋   | 265582/400000 [00:30<00:15, 8873.78it/s] 67%|██████▋   | 266470/400000 [00:30<00:15, 8738.14it/s] 67%|██████▋   | 267345/400000 [00:30<00:15, 8723.13it/s] 67%|██████▋   | 268218/400000 [00:30<00:15, 8429.35it/s] 67%|██████▋   | 269108/400000 [00:30<00:15, 8562.95it/s] 68%|██████▊   | 270033/400000 [00:30<00:14, 8756.60it/s] 68%|██████▊   | 270912/400000 [00:30<00:14, 8672.42it/s] 68%|██████▊   | 271790/400000 [00:30<00:14, 8702.92it/s] 68%|██████▊   | 272677/400000 [00:30<00:14, 8750.48it/s] 68%|██████▊   | 273566/400000 [00:30<00:14, 8789.33it/s] 69%|██████▊   | 274446/400000 [00:31<00:14, 8541.51it/s] 69%|██████▉   | 275326/400000 [00:31<00:14, 8616.54it/s] 69%|██████▉   | 276220/400000 [00:31<00:14, 8709.57it/s] 69%|██████▉   | 277103/400000 [00:31<00:14, 8743.02it/s] 69%|██████▉   | 277992/400000 [00:31<00:13, 8784.01it/s] 70%|██████▉   | 278916/400000 [00:31<00:13, 8915.58it/s] 70%|██████▉   | 279809/400000 [00:31<00:13, 8802.28it/s] 70%|███████   | 280691/400000 [00:31<00:13, 8787.35it/s] 70%|███████   | 281571/400000 [00:31<00:13, 8761.11it/s] 71%|███████   | 282455/400000 [00:31<00:13, 8782.83it/s] 71%|███████   | 283348/400000 [00:32<00:13, 8826.37it/s] 71%|███████   | 284243/400000 [00:32<00:13, 8861.54it/s] 71%|███████▏  | 285130/400000 [00:32<00:13, 8758.95it/s] 72%|███████▏  | 286007/400000 [00:32<00:13, 8691.00it/s] 72%|███████▏  | 286891/400000 [00:32<00:12, 8735.03it/s] 72%|███████▏  | 287777/400000 [00:32<00:12, 8770.44it/s] 72%|███████▏  | 288655/400000 [00:32<00:12, 8772.29it/s] 72%|███████▏  | 289543/400000 [00:32<00:12, 8803.22it/s] 73%|███████▎  | 290437/400000 [00:32<00:12, 8841.79it/s] 73%|███████▎  | 291331/400000 [00:32<00:12, 8869.37it/s] 73%|███████▎  | 292219/400000 [00:33<00:12, 8867.60it/s] 73%|███████▎  | 293111/400000 [00:33<00:12, 8881.45it/s] 74%|███████▎  | 294028/400000 [00:33<00:11, 8965.38it/s] 74%|███████▎  | 294968/400000 [00:33<00:11, 9091.38it/s] 74%|███████▍  | 295878/400000 [00:33<00:11, 9033.28it/s] 74%|███████▍  | 296782/400000 [00:33<00:11, 8979.29it/s] 74%|███████▍  | 297681/400000 [00:33<00:11, 8939.32it/s] 75%|███████▍  | 298576/400000 [00:33<00:11, 8886.81it/s] 75%|███████▍  | 299465/400000 [00:33<00:11, 8875.57it/s] 75%|███████▌  | 300361/400000 [00:33<00:11, 8900.65it/s] 75%|███████▌  | 301252/400000 [00:34<00:11, 8892.40it/s] 76%|███████▌  | 302142/400000 [00:34<00:11, 8878.20it/s] 76%|███████▌  | 303068/400000 [00:34<00:10, 8989.06it/s] 76%|███████▌  | 303975/400000 [00:34<00:10, 9011.07it/s] 76%|███████▌  | 304877/400000 [00:34<00:10, 8965.25it/s] 76%|███████▋  | 305790/400000 [00:34<00:10, 9013.86it/s] 77%|███████▋  | 306694/400000 [00:34<00:10, 9020.01it/s] 77%|███████▋  | 307597/400000 [00:34<00:10, 8985.84it/s] 77%|███████▋  | 308497/400000 [00:34<00:10, 8989.08it/s] 77%|███████▋  | 309398/400000 [00:34<00:10, 8993.74it/s] 78%|███████▊  | 310310/400000 [00:35<00:09, 9031.18it/s] 78%|███████▊  | 311216/400000 [00:35<00:09, 9037.48it/s] 78%|███████▊  | 312148/400000 [00:35<00:09, 9118.48it/s] 78%|███████▊  | 313083/400000 [00:35<00:09, 9185.84it/s] 79%|███████▊  | 314002/400000 [00:35<00:09, 9073.07it/s] 79%|███████▊  | 314910/400000 [00:35<00:09, 9010.02it/s] 79%|███████▉  | 315812/400000 [00:35<00:09, 8954.33it/s] 79%|███████▉  | 316708/400000 [00:35<00:09, 8951.73it/s] 79%|███████▉  | 317604/400000 [00:35<00:09, 8857.70it/s] 80%|███████▉  | 318491/400000 [00:35<00:09, 8860.98it/s] 80%|███████▉  | 319456/400000 [00:36<00:08, 9083.02it/s] 80%|████████  | 320366/400000 [00:36<00:08, 9066.97it/s] 80%|████████  | 321274/400000 [00:36<00:08, 8997.75it/s] 81%|████████  | 322175/400000 [00:36<00:08, 8908.57it/s] 81%|████████  | 323073/400000 [00:36<00:08, 8925.41it/s] 81%|████████  | 323997/400000 [00:36<00:08, 9014.55it/s] 81%|████████  | 324900/400000 [00:36<00:08, 9017.28it/s] 81%|████████▏ | 325803/400000 [00:36<00:08, 8937.54it/s] 82%|████████▏ | 326698/400000 [00:36<00:08, 8862.76it/s] 82%|████████▏ | 327585/400000 [00:37<00:08, 8780.55it/s] 82%|████████▏ | 328488/400000 [00:37<00:08, 8852.38it/s] 82%|████████▏ | 329403/400000 [00:37<00:07, 8938.21it/s] 83%|████████▎ | 330307/400000 [00:37<00:07, 8966.61it/s] 83%|████████▎ | 331265/400000 [00:37<00:07, 9140.26it/s] 83%|████████▎ | 332183/400000 [00:37<00:07, 9151.54it/s] 83%|████████▎ | 333099/400000 [00:37<00:07, 9079.94it/s] 84%|████████▎ | 334008/400000 [00:37<00:07, 9037.53it/s] 84%|████████▎ | 334913/400000 [00:37<00:07, 8968.06it/s] 84%|████████▍ | 335831/400000 [00:37<00:07, 9028.65it/s] 84%|████████▍ | 336735/400000 [00:38<00:07, 8994.88it/s] 84%|████████▍ | 337666/400000 [00:38<00:06, 9085.50it/s] 85%|████████▍ | 338600/400000 [00:38<00:06, 9158.76it/s] 85%|████████▍ | 339517/400000 [00:38<00:06, 9161.57it/s] 85%|████████▌ | 340478/400000 [00:38<00:06, 9291.34it/s] 85%|████████▌ | 341408/400000 [00:38<00:06, 9130.44it/s] 86%|████████▌ | 342323/400000 [00:38<00:06, 9058.50it/s] 86%|████████▌ | 343230/400000 [00:38<00:06, 8973.53it/s] 86%|████████▌ | 344129/400000 [00:38<00:06, 8892.52it/s] 86%|████████▋ | 345049/400000 [00:38<00:06, 8981.29it/s] 86%|████████▋ | 345948/400000 [00:39<00:06, 8891.64it/s] 87%|████████▋ | 346865/400000 [00:39<00:05, 8972.54it/s] 87%|████████▋ | 347783/400000 [00:39<00:05, 9030.95it/s] 87%|████████▋ | 348710/400000 [00:39<00:05, 9099.25it/s] 87%|████████▋ | 349621/400000 [00:39<00:05, 9057.03it/s] 88%|████████▊ | 350528/400000 [00:39<00:05, 9010.14it/s] 88%|████████▊ | 351430/400000 [00:39<00:05, 9001.17it/s] 88%|████████▊ | 352331/400000 [00:39<00:05, 8888.46it/s] 88%|████████▊ | 353233/400000 [00:39<00:05, 8925.64it/s] 89%|████████▊ | 354162/400000 [00:39<00:05, 9031.43it/s] 89%|████████▉ | 355079/400000 [00:40<00:04, 9071.40it/s] 89%|████████▉ | 356032/400000 [00:40<00:04, 9201.59it/s] 89%|████████▉ | 356967/400000 [00:40<00:04, 9243.77it/s] 89%|████████▉ | 357892/400000 [00:40<00:04, 9226.26it/s] 90%|████████▉ | 358816/400000 [00:40<00:04, 9179.77it/s] 90%|████████▉ | 359735/400000 [00:40<00:04, 9106.70it/s] 90%|█████████ | 360647/400000 [00:40<00:04, 9035.41it/s] 90%|█████████ | 361551/400000 [00:40<00:04, 8970.82it/s] 91%|█████████ | 362449/400000 [00:40<00:04, 8951.89it/s] 91%|█████████ | 363359/400000 [00:40<00:04, 8993.71it/s] 91%|█████████ | 364259/400000 [00:41<00:04, 8788.30it/s] 91%|█████████▏| 365189/400000 [00:41<00:03, 8935.19it/s] 92%|█████████▏| 366114/400000 [00:41<00:03, 9026.83it/s] 92%|█████████▏| 367018/400000 [00:41<00:03, 8946.54it/s] 92%|█████████▏| 367914/400000 [00:41<00:03, 8947.87it/s] 92%|█████████▏| 368810/400000 [00:41<00:03, 8884.93it/s] 92%|█████████▏| 369700/400000 [00:41<00:03, 8855.55it/s] 93%|█████████▎| 370587/400000 [00:41<00:03, 8812.72it/s] 93%|█████████▎| 371469/400000 [00:41<00:03, 8785.32it/s] 93%|█████████▎| 372351/400000 [00:41<00:03, 8793.82it/s] 93%|█████████▎| 373235/400000 [00:42<00:03, 8807.02it/s] 94%|█████████▎| 374118/400000 [00:42<00:02, 8813.89it/s] 94%|█████████▍| 375007/400000 [00:42<00:02, 8834.02it/s] 94%|█████████▍| 375891/400000 [00:42<00:02, 8830.93it/s] 94%|█████████▍| 376779/400000 [00:42<00:02, 8845.47it/s] 94%|█████████▍| 377677/400000 [00:42<00:02, 8883.05it/s] 95%|█████████▍| 378566/400000 [00:42<00:02, 8870.03it/s] 95%|█████████▍| 379454/400000 [00:42<00:02, 8861.53it/s] 95%|█████████▌| 380341/400000 [00:42<00:02, 8837.43it/s] 95%|█████████▌| 381225/400000 [00:42<00:02, 8700.09it/s] 96%|█████████▌| 382096/400000 [00:43<00:02, 8608.35it/s] 96%|█████████▌| 383017/400000 [00:43<00:01, 8779.68it/s] 96%|█████████▌| 383957/400000 [00:43<00:01, 8954.65it/s] 96%|█████████▌| 384889/400000 [00:43<00:01, 9059.90it/s] 96%|█████████▋| 385797/400000 [00:43<00:01, 8779.45it/s] 97%|█████████▋| 386678/400000 [00:43<00:01, 8731.95it/s] 97%|█████████▋| 387562/400000 [00:43<00:01, 8763.04it/s] 97%|█████████▋| 388486/400000 [00:43<00:01, 8898.54it/s] 97%|█████████▋| 389384/400000 [00:43<00:01, 8921.10it/s] 98%|█████████▊| 390291/400000 [00:43<00:01, 8964.22it/s] 98%|█████████▊| 391189/400000 [00:44<00:00, 8922.78it/s] 98%|█████████▊| 392110/400000 [00:44<00:00, 9005.50it/s] 98%|█████████▊| 393013/400000 [00:44<00:00, 9009.91it/s] 98%|█████████▊| 393915/400000 [00:44<00:00, 8973.02it/s] 99%|█████████▊| 394813/400000 [00:44<00:00, 8957.24it/s] 99%|█████████▉| 395709/400000 [00:44<00:00, 8901.98it/s] 99%|█████████▉| 396600/400000 [00:44<00:00, 8872.00it/s] 99%|█████████▉| 397488/400000 [00:44<00:00, 8859.84it/s]100%|█████████▉| 398387/400000 [00:44<00:00, 8896.36it/s]100%|█████████▉| 399277/400000 [00:45<00:00, 8787.68it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8869.17it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1ccdef4ef0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01123530167196271 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011321443578471308 	 Accuracy: 50

  model saves at 50% accuracy 

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

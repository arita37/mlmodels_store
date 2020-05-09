
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6bea60570fb66cc439295de592a8a5b786b95000', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '6bea60570fb66cc439295de592a8a5b786b95000', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6bea60570fb66cc439295de592a8a5b786b95000

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6bea60570fb66cc439295de592a8a5b786b95000

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  timeseries 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_timeseries/test02/model_list.json 

  Model List [{'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}}] 

  


### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ##### 

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa3721e84a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 11:13:32.728870
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 11:13:32.733478
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 11:13:32.737596
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 11:13:32.740801
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -18.2877
metric_name                                             r2_score
Name: 3, dtype: object 

  


### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ##### 

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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa36a5384e0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355772.1562
Epoch 2/10

1/1 [==============================] - 0s 101ms/step - loss: 279725.3125
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 197553.8594
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 126846.8516
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 75664.1016
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 44231.6680
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 26668.7070
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 16991.7871
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 11533.5078
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 8315.4580

  #### Inference Need return ypred, ytrue ######################### 
[[-2.22400144e-01 -7.64818937e-02 -8.01630765e-02  3.72237444e-01
   8.19347680e-01  1.86600709e+00 -1.42128408e-01  4.71780539e-01
  -2.67894745e+00 -1.09596097e+00 -5.33981144e-01 -1.28297281e+00
   3.02483320e-01 -4.01059598e-01  4.92586613e-01 -2.64037549e-01
  -2.86045641e-01 -1.30172598e+00 -2.13666052e-01 -1.32679796e+00
  -1.41247666e+00 -4.17481363e-01  7.65200436e-01 -6.37106180e-01
   9.70955014e-01 -4.51597899e-01 -3.95702869e-01 -4.24981117e-04
   6.67068005e-01 -6.57298982e-01  1.24827057e-01 -7.51085639e-01
   1.09391749e-01  4.88132596e-01 -1.90662563e+00 -2.55150795e-01
  -3.24311435e-01 -6.80491030e-01 -1.70979157e-01  3.46338749e-01
   9.83982682e-01  5.19120336e-01 -2.69574314e-01 -6.14494562e-01
  -5.43535471e-01  1.14718676e-02 -6.89984024e-01  8.70062828e-01
   8.72271717e-01  1.58492243e+00 -1.47499204e-01 -2.58265942e-01
  -2.23450363e-02  5.71585298e-01 -8.74768853e-01  4.33679760e-01
   4.54055369e-02 -1.33152080e+00  5.60324788e-01 -1.74069002e-01
  -1.61786556e+00  1.43449557e+00 -2.69491583e-01 -1.32843900e+00
  -1.42640471e-01  3.73031735e-01  1.28994703e-01 -1.35541022e+00
   5.83368063e-01  6.09895229e-01 -1.60740292e+00 -3.96357298e-01
   7.50665843e-01 -1.15488887e-01 -2.47897118e-01  1.22241449e+00
  -8.07647347e-01 -8.57430339e-01 -2.24885798e+00  2.01112822e-01
  -9.72474396e-01  1.94758996e-02 -8.61889482e-01 -1.68074980e-01
  -1.36582851e+00 -4.16096479e-01  1.15563273e-01  1.91388214e+00
   1.01251304e-01  1.39546365e-01  1.46339428e+00  1.92172420e+00
  -9.24204826e-01  5.59608102e-01  1.41165257e+00 -1.18428922e+00
   7.61230946e-01  2.43588448e-01  1.62882054e+00  7.33089149e-02
   4.87564206e-01 -9.02227759e-01  7.92537332e-01  7.69859493e-01
   7.19407052e-02 -4.85403687e-01 -6.48597836e-01 -2.09139824e-01
   7.50314116e-01 -1.36884069e+00 -1.73147118e+00  5.76748967e-01
   2.24373341e-01 -5.77110887e-01  8.83104086e-01 -8.06416988e-01
  -2.29947135e-01  9.63715792e-01  1.24880564e+00  1.06688786e+00
   1.61213309e-01  7.18352556e+00  6.71060419e+00  5.97342396e+00
   9.29922009e+00  7.57450104e+00  5.35275221e+00  7.04000950e+00
   7.37647343e+00  8.10719299e+00  7.25824928e+00  7.91296005e+00
   8.56667614e+00  6.66173315e+00  6.88704395e+00  7.12348938e+00
   8.30944729e+00  6.61897707e+00  6.31305408e+00  7.25726080e+00
   8.05151749e+00  7.05307055e+00  6.99022484e+00  7.22627831e+00
   7.95700550e+00  7.22713900e+00  6.46292686e+00  6.63700771e+00
   5.29909611e+00  8.84181118e+00  7.03744936e+00  7.00501108e+00
   8.19237232e+00  6.59419203e+00  5.93538857e+00  4.85121727e+00
   5.94559956e+00  8.39881420e+00  6.50818825e+00  8.40555573e+00
   6.85669422e+00  7.71003532e+00  8.38545990e+00  7.35316563e+00
   6.37013721e+00  8.10188961e+00  7.43029213e+00  6.58674479e+00
   6.95365620e+00  6.52480364e+00  7.00491476e+00  7.22382736e+00
   6.32079935e+00  8.27258015e+00  7.20082331e+00  5.73311186e+00
   6.78609800e+00  7.58108139e+00  9.03370476e+00  6.40233612e+00
   1.93288612e+00  8.04451108e-01  2.29781687e-01  1.14643192e+00
   9.88217711e-01  1.74432433e+00  1.80752766e+00  2.10717678e+00
   5.92859924e-01  1.00814402e+00  5.88143587e-01  2.91237056e-01
   2.28745794e+00  1.46215653e+00  2.09774971e+00  1.19746184e+00
   6.07548952e-01  4.47310925e-01  1.34194899e+00  3.56440127e-01
   2.31148720e+00  7.29387164e-01  7.18678474e-01  4.55665231e-01
   5.10814190e-01  4.47285771e-01  8.83540392e-01  1.58301139e+00
   7.96338379e-01  1.29686093e+00  3.98478448e-01  1.62790585e+00
   6.68165207e-01  7.95869231e-01  1.83690214e+00  1.27859390e+00
   4.67838347e-01  2.09397602e+00  7.92514682e-01  1.70157719e+00
   1.02916384e+00  5.91495991e-01  1.15263367e+00  2.35661888e+00
   7.19343662e-01  2.79325366e-01  1.52518761e+00  9.24057305e-01
   1.16170001e+00  4.16324914e-01  3.14325094e-01  5.70940197e-01
   1.89609981e+00  1.29834104e+00  1.12680626e+00  3.62649560e-01
   5.90115964e-01  2.60154200e+00  3.38695526e-01  2.41651607e+00
   2.43288994e+00  3.27241659e-01  2.64252424e-01  7.05665588e-01
   1.12202060e+00  1.55010355e+00  5.54669380e-01  9.96032000e-01
   2.05224752e+00  9.84362662e-01  7.28535175e-01  7.63260007e-01
   2.74103522e-01  1.97507763e+00  9.79000807e-01  7.22942948e-01
   8.14288974e-01  1.31390750e-01  1.85752296e+00  7.52961278e-01
   1.08074427e+00  8.65019262e-01  9.79035318e-01  4.60979104e-01
   9.46966350e-01  1.48293388e+00  1.97421014e+00  1.53918159e+00
   1.80575395e+00  1.15712583e+00  2.51062870e-01  3.98947358e-01
   1.55701399e+00  7.86422193e-01  1.44001842e+00  5.13393223e-01
   3.62214923e-01  2.22333241e+00  1.73579514e+00  6.64834499e-01
   2.13642883e+00  2.21058369e+00  3.28459501e-01  5.90368330e-01
   4.16660130e-01  1.62085986e+00  6.46645486e-01  1.64424396e+00
   2.26101875e-01  1.64262462e+00  1.64308608e+00  1.25287175e+00
   4.73737121e-01  5.03128052e-01  5.40292144e-01  1.19312167e+00
   9.80704725e-01  7.06690311e-01  1.74382102e+00  1.76069069e+00
   5.11906147e-02  6.19790649e+00  7.85766220e+00  8.42304802e+00
   8.25714684e+00  9.10511208e+00  6.78888988e+00  7.66482496e+00
   8.72051620e+00  8.36007881e+00  7.81898832e+00  8.35024738e+00
   7.00358152e+00  6.32260561e+00  8.88379955e+00  7.49551105e+00
   7.10126734e+00  7.26837111e+00  6.88095427e+00  7.75243998e+00
   7.75080776e+00  7.95469809e+00  8.92475605e+00  7.63271141e+00
   8.24404430e+00  6.31840563e+00  6.37555313e+00  8.45110989e+00
   9.00581074e+00  8.28580570e+00  7.36838102e+00  8.74439621e+00
   9.06733227e+00  7.64195299e+00  5.82761765e+00  7.25586271e+00
   6.74458599e+00  9.24764729e+00  6.14992857e+00  7.57672501e+00
   8.59918880e+00  8.87803078e+00  6.52563715e+00  5.77022743e+00
   6.83017731e+00  7.53718996e+00  6.75842047e+00  6.86883163e+00
   8.89560318e+00  8.64269638e+00  9.09376717e+00  6.47841263e+00
   7.40306759e+00  8.19815350e+00  8.32432270e+00  8.12025928e+00
   6.35599232e+00  7.05529642e+00  6.49749470e+00  8.21856976e+00
  -3.23417950e+00 -6.98325634e+00  7.35434914e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 11:13:43.091200
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6399
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 11:13:43.094950
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8977.35
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 11:13:43.098196
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.7199
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 11:13:43.101297
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -802.985
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140339282073192
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140338340832144
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140338340832648
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140338340833152
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140338340833656
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140338340834160

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa3663b9f60> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.549581
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.504717
grad_step = 000002, loss = 0.469700
grad_step = 000003, loss = 0.434928
grad_step = 000004, loss = 0.397823
grad_step = 000005, loss = 0.360663
grad_step = 000006, loss = 0.329425
grad_step = 000007, loss = 0.318222
grad_step = 000008, loss = 0.321783
grad_step = 000009, loss = 0.306274
grad_step = 000010, loss = 0.283123
grad_step = 000011, loss = 0.267710
grad_step = 000012, loss = 0.260496
grad_step = 000013, loss = 0.255723
grad_step = 000014, loss = 0.248383
grad_step = 000015, loss = 0.237773
grad_step = 000016, loss = 0.224618
grad_step = 000017, loss = 0.210655
grad_step = 000018, loss = 0.198023
grad_step = 000019, loss = 0.188548
grad_step = 000020, loss = 0.181917
grad_step = 000021, loss = 0.175136
grad_step = 000022, loss = 0.166490
grad_step = 000023, loss = 0.157404
grad_step = 000024, loss = 0.149397
grad_step = 000025, loss = 0.142748
grad_step = 000026, loss = 0.136724
grad_step = 000027, loss = 0.130213
grad_step = 000028, loss = 0.123046
grad_step = 000029, loss = 0.116005
grad_step = 000030, loss = 0.109651
grad_step = 000031, loss = 0.103752
grad_step = 000032, loss = 0.098049
grad_step = 000033, loss = 0.092373
grad_step = 000034, loss = 0.086766
grad_step = 000035, loss = 0.081472
grad_step = 000036, loss = 0.076676
grad_step = 000037, loss = 0.072376
grad_step = 000038, loss = 0.068287
grad_step = 000039, loss = 0.064145
grad_step = 000040, loss = 0.060009
grad_step = 000041, loss = 0.056007
grad_step = 000042, loss = 0.052289
grad_step = 000043, loss = 0.048963
grad_step = 000044, loss = 0.045802
grad_step = 000045, loss = 0.042670
grad_step = 000046, loss = 0.039697
grad_step = 000047, loss = 0.037007
grad_step = 000048, loss = 0.034603
grad_step = 000049, loss = 0.032296
grad_step = 000050, loss = 0.030003
grad_step = 000051, loss = 0.027788
grad_step = 000052, loss = 0.025770
grad_step = 000053, loss = 0.023928
grad_step = 000054, loss = 0.022179
grad_step = 000055, loss = 0.020528
grad_step = 000056, loss = 0.018971
grad_step = 000057, loss = 0.017561
grad_step = 000058, loss = 0.016276
grad_step = 000059, loss = 0.015071
grad_step = 000060, loss = 0.013919
grad_step = 000061, loss = 0.012846
grad_step = 000062, loss = 0.011869
grad_step = 000063, loss = 0.010982
grad_step = 000064, loss = 0.010137
grad_step = 000065, loss = 0.009362
grad_step = 000066, loss = 0.008678
grad_step = 000067, loss = 0.008056
grad_step = 000068, loss = 0.007470
grad_step = 000069, loss = 0.006925
grad_step = 000070, loss = 0.006433
grad_step = 000071, loss = 0.005988
grad_step = 000072, loss = 0.005578
grad_step = 000073, loss = 0.005206
grad_step = 000074, loss = 0.004872
grad_step = 000075, loss = 0.004576
grad_step = 000076, loss = 0.004302
grad_step = 000077, loss = 0.004050
grad_step = 000078, loss = 0.003828
grad_step = 000079, loss = 0.003636
grad_step = 000080, loss = 0.003458
grad_step = 000081, loss = 0.003296
grad_step = 000082, loss = 0.003158
grad_step = 000083, loss = 0.003035
grad_step = 000084, loss = 0.002924
grad_step = 000085, loss = 0.002823
grad_step = 000086, loss = 0.002738
grad_step = 000087, loss = 0.002664
grad_step = 000088, loss = 0.002598
grad_step = 000089, loss = 0.002539
grad_step = 000090, loss = 0.002489
grad_step = 000091, loss = 0.002448
grad_step = 000092, loss = 0.002410
grad_step = 000093, loss = 0.002377
grad_step = 000094, loss = 0.002351
grad_step = 000095, loss = 0.002329
grad_step = 000096, loss = 0.002309
grad_step = 000097, loss = 0.002292
grad_step = 000098, loss = 0.002278
grad_step = 000099, loss = 0.002266
grad_step = 000100, loss = 0.002256
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002246
grad_step = 000102, loss = 0.002239
grad_step = 000103, loss = 0.002232
grad_step = 000104, loss = 0.002225
grad_step = 000105, loss = 0.002220
grad_step = 000106, loss = 0.002216
grad_step = 000107, loss = 0.002211
grad_step = 000108, loss = 0.002206
grad_step = 000109, loss = 0.002202
grad_step = 000110, loss = 0.002198
grad_step = 000111, loss = 0.002195
grad_step = 000112, loss = 0.002191
grad_step = 000113, loss = 0.002187
grad_step = 000114, loss = 0.002184
grad_step = 000115, loss = 0.002180
grad_step = 000116, loss = 0.002177
grad_step = 000117, loss = 0.002173
grad_step = 000118, loss = 0.002169
grad_step = 000119, loss = 0.002166
grad_step = 000120, loss = 0.002162
grad_step = 000121, loss = 0.002159
grad_step = 000122, loss = 0.002156
grad_step = 000123, loss = 0.002153
grad_step = 000124, loss = 0.002154
grad_step = 000125, loss = 0.002163
grad_step = 000126, loss = 0.002194
grad_step = 000127, loss = 0.002223
grad_step = 000128, loss = 0.002218
grad_step = 000129, loss = 0.002147
grad_step = 000130, loss = 0.002146
grad_step = 000131, loss = 0.002191
grad_step = 000132, loss = 0.002158
grad_step = 000133, loss = 0.002123
grad_step = 000134, loss = 0.002146
grad_step = 000135, loss = 0.002153
grad_step = 000136, loss = 0.002124
grad_step = 000137, loss = 0.002117
grad_step = 000138, loss = 0.002135
grad_step = 000139, loss = 0.002130
grad_step = 000140, loss = 0.002107
grad_step = 000141, loss = 0.002113
grad_step = 000142, loss = 0.002123
grad_step = 000143, loss = 0.002107
grad_step = 000144, loss = 0.002097
grad_step = 000145, loss = 0.002106
grad_step = 000146, loss = 0.002104
grad_step = 000147, loss = 0.002092
grad_step = 000148, loss = 0.002090
grad_step = 000149, loss = 0.002095
grad_step = 000150, loss = 0.002091
grad_step = 000151, loss = 0.002082
grad_step = 000152, loss = 0.002080
grad_step = 000153, loss = 0.002084
grad_step = 000154, loss = 0.002081
grad_step = 000155, loss = 0.002073
grad_step = 000156, loss = 0.002070
grad_step = 000157, loss = 0.002071
grad_step = 000158, loss = 0.002071
grad_step = 000159, loss = 0.002066
grad_step = 000160, loss = 0.002061
grad_step = 000161, loss = 0.002059
grad_step = 000162, loss = 0.002058
grad_step = 000163, loss = 0.002057
grad_step = 000164, loss = 0.002054
grad_step = 000165, loss = 0.002050
grad_step = 000166, loss = 0.002047
grad_step = 000167, loss = 0.002045
grad_step = 000168, loss = 0.002044
grad_step = 000169, loss = 0.002042
grad_step = 000170, loss = 0.002040
grad_step = 000171, loss = 0.002038
grad_step = 000172, loss = 0.002035
grad_step = 000173, loss = 0.002031
grad_step = 000174, loss = 0.002028
grad_step = 000175, loss = 0.002025
grad_step = 000176, loss = 0.002022
grad_step = 000177, loss = 0.002020
grad_step = 000178, loss = 0.002018
grad_step = 000179, loss = 0.002015
grad_step = 000180, loss = 0.002013
grad_step = 000181, loss = 0.002012
grad_step = 000182, loss = 0.002014
grad_step = 000183, loss = 0.002020
grad_step = 000184, loss = 0.002035
grad_step = 000185, loss = 0.002069
grad_step = 000186, loss = 0.002086
grad_step = 000187, loss = 0.002084
grad_step = 000188, loss = 0.002021
grad_step = 000189, loss = 0.001993
grad_step = 000190, loss = 0.002023
grad_step = 000191, loss = 0.002039
grad_step = 000192, loss = 0.002016
grad_step = 000193, loss = 0.001981
grad_step = 000194, loss = 0.001986
grad_step = 000195, loss = 0.002012
grad_step = 000196, loss = 0.002007
grad_step = 000197, loss = 0.001984
grad_step = 000198, loss = 0.001964
grad_step = 000199, loss = 0.001971
grad_step = 000200, loss = 0.001983
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001974
grad_step = 000202, loss = 0.001955
grad_step = 000203, loss = 0.001949
grad_step = 000204, loss = 0.001956
grad_step = 000205, loss = 0.001962
grad_step = 000206, loss = 0.001954
grad_step = 000207, loss = 0.001940
grad_step = 000208, loss = 0.001931
grad_step = 000209, loss = 0.001932
grad_step = 000210, loss = 0.001936
grad_step = 000211, loss = 0.001933
grad_step = 000212, loss = 0.001924
grad_step = 000213, loss = 0.001915
grad_step = 000214, loss = 0.001910
grad_step = 000215, loss = 0.001909
grad_step = 000216, loss = 0.001909
grad_step = 000217, loss = 0.001907
grad_step = 000218, loss = 0.001901
grad_step = 000219, loss = 0.001893
grad_step = 000220, loss = 0.001886
grad_step = 000221, loss = 0.001881
grad_step = 000222, loss = 0.001878
grad_step = 000223, loss = 0.001876
grad_step = 000224, loss = 0.001875
grad_step = 000225, loss = 0.001871
grad_step = 000226, loss = 0.001868
grad_step = 000227, loss = 0.001863
grad_step = 000228, loss = 0.001858
grad_step = 000229, loss = 0.001851
grad_step = 000230, loss = 0.001844
grad_step = 000231, loss = 0.001836
grad_step = 000232, loss = 0.001829
grad_step = 000233, loss = 0.001822
grad_step = 000234, loss = 0.001816
grad_step = 000235, loss = 0.001811
grad_step = 000236, loss = 0.001806
grad_step = 000237, loss = 0.001802
grad_step = 000238, loss = 0.001805
grad_step = 000239, loss = 0.001815
grad_step = 000240, loss = 0.001840
grad_step = 000241, loss = 0.001899
grad_step = 000242, loss = 0.001896
grad_step = 000243, loss = 0.001866
grad_step = 000244, loss = 0.001782
grad_step = 000245, loss = 0.001767
grad_step = 000246, loss = 0.001810
grad_step = 000247, loss = 0.001812
grad_step = 000248, loss = 0.001770
grad_step = 000249, loss = 0.001734
grad_step = 000250, loss = 0.001750
grad_step = 000251, loss = 0.001788
grad_step = 000252, loss = 0.001773
grad_step = 000253, loss = 0.001746
grad_step = 000254, loss = 0.001707
grad_step = 000255, loss = 0.001704
grad_step = 000256, loss = 0.001724
grad_step = 000257, loss = 0.001728
grad_step = 000258, loss = 0.001717
grad_step = 000259, loss = 0.001686
grad_step = 000260, loss = 0.001669
grad_step = 000261, loss = 0.001669
grad_step = 000262, loss = 0.001681
grad_step = 000263, loss = 0.001708
grad_step = 000264, loss = 0.001705
grad_step = 000265, loss = 0.001711
grad_step = 000266, loss = 0.001675
grad_step = 000267, loss = 0.001650
grad_step = 000268, loss = 0.001628
grad_step = 000269, loss = 0.001623
grad_step = 000270, loss = 0.001630
grad_step = 000271, loss = 0.001642
grad_step = 000272, loss = 0.001671
grad_step = 000273, loss = 0.001682
grad_step = 000274, loss = 0.001712
grad_step = 000275, loss = 0.001667
grad_step = 000276, loss = 0.001623
grad_step = 000277, loss = 0.001594
grad_step = 000278, loss = 0.001601
grad_step = 000279, loss = 0.001627
grad_step = 000280, loss = 0.001642
grad_step = 000281, loss = 0.001650
grad_step = 000282, loss = 0.001620
grad_step = 000283, loss = 0.001589
grad_step = 000284, loss = 0.001575
grad_step = 000285, loss = 0.001581
grad_step = 000286, loss = 0.001602
grad_step = 000287, loss = 0.001625
grad_step = 000288, loss = 0.001653
grad_step = 000289, loss = 0.001654
grad_step = 000290, loss = 0.001637
grad_step = 000291, loss = 0.001592
grad_step = 000292, loss = 0.001562
grad_step = 000293, loss = 0.001563
grad_step = 000294, loss = 0.001583
grad_step = 000295, loss = 0.001597
grad_step = 000296, loss = 0.001583
grad_step = 000297, loss = 0.001560
grad_step = 000298, loss = 0.001547
grad_step = 000299, loss = 0.001555
grad_step = 000300, loss = 0.001574
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001585
grad_step = 000302, loss = 0.001588
grad_step = 000303, loss = 0.001572
grad_step = 000304, loss = 0.001553
grad_step = 000305, loss = 0.001537
grad_step = 000306, loss = 0.001534
grad_step = 000307, loss = 0.001542
grad_step = 000308, loss = 0.001551
grad_step = 000309, loss = 0.001556
grad_step = 000310, loss = 0.001551
grad_step = 000311, loss = 0.001542
grad_step = 000312, loss = 0.001531
grad_step = 000313, loss = 0.001523
grad_step = 000314, loss = 0.001519
grad_step = 000315, loss = 0.001520
grad_step = 000316, loss = 0.001523
grad_step = 000317, loss = 0.001527
grad_step = 000318, loss = 0.001529
grad_step = 000319, loss = 0.001530
grad_step = 000320, loss = 0.001529
grad_step = 000321, loss = 0.001525
grad_step = 000322, loss = 0.001520
grad_step = 000323, loss = 0.001514
grad_step = 000324, loss = 0.001508
grad_step = 000325, loss = 0.001503
grad_step = 000326, loss = 0.001500
grad_step = 000327, loss = 0.001498
grad_step = 000328, loss = 0.001498
grad_step = 000329, loss = 0.001499
grad_step = 000330, loss = 0.001500
grad_step = 000331, loss = 0.001505
grad_step = 000332, loss = 0.001513
grad_step = 000333, loss = 0.001531
grad_step = 000334, loss = 0.001562
grad_step = 000335, loss = 0.001615
grad_step = 000336, loss = 0.001670
grad_step = 000337, loss = 0.001708
grad_step = 000338, loss = 0.001643
grad_step = 000339, loss = 0.001537
grad_step = 000340, loss = 0.001483
grad_step = 000341, loss = 0.001533
grad_step = 000342, loss = 0.001581
grad_step = 000343, loss = 0.001529
grad_step = 000344, loss = 0.001479
grad_step = 000345, loss = 0.001502
grad_step = 000346, loss = 0.001536
grad_step = 000347, loss = 0.001520
grad_step = 000348, loss = 0.001476
grad_step = 000349, loss = 0.001479
grad_step = 000350, loss = 0.001510
grad_step = 000351, loss = 0.001509
grad_step = 000352, loss = 0.001479
grad_step = 000353, loss = 0.001463
grad_step = 000354, loss = 0.001475
grad_step = 000355, loss = 0.001492
grad_step = 000356, loss = 0.001482
grad_step = 000357, loss = 0.001463
grad_step = 000358, loss = 0.001454
grad_step = 000359, loss = 0.001462
grad_step = 000360, loss = 0.001474
grad_step = 000361, loss = 0.001475
grad_step = 000362, loss = 0.001467
grad_step = 000363, loss = 0.001453
grad_step = 000364, loss = 0.001445
grad_step = 000365, loss = 0.001446
grad_step = 000366, loss = 0.001450
grad_step = 000367, loss = 0.001455
grad_step = 000368, loss = 0.001453
grad_step = 000369, loss = 0.001449
grad_step = 000370, loss = 0.001443
grad_step = 000371, loss = 0.001437
grad_step = 000372, loss = 0.001433
grad_step = 000373, loss = 0.001433
grad_step = 000374, loss = 0.001435
grad_step = 000375, loss = 0.001436
grad_step = 000376, loss = 0.001436
grad_step = 000377, loss = 0.001435
grad_step = 000378, loss = 0.001433
grad_step = 000379, loss = 0.001430
grad_step = 000380, loss = 0.001427
grad_step = 000381, loss = 0.001423
grad_step = 000382, loss = 0.001420
grad_step = 000383, loss = 0.001418
grad_step = 000384, loss = 0.001416
grad_step = 000385, loss = 0.001415
grad_step = 000386, loss = 0.001414
grad_step = 000387, loss = 0.001413
grad_step = 000388, loss = 0.001412
grad_step = 000389, loss = 0.001412
grad_step = 000390, loss = 0.001413
grad_step = 000391, loss = 0.001415
grad_step = 000392, loss = 0.001418
grad_step = 000393, loss = 0.001424
grad_step = 000394, loss = 0.001430
grad_step = 000395, loss = 0.001441
grad_step = 000396, loss = 0.001451
grad_step = 000397, loss = 0.001462
grad_step = 000398, loss = 0.001465
grad_step = 000399, loss = 0.001462
grad_step = 000400, loss = 0.001445
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001425
grad_step = 000402, loss = 0.001404
grad_step = 000403, loss = 0.001392
grad_step = 000404, loss = 0.001390
grad_step = 000405, loss = 0.001396
grad_step = 000406, loss = 0.001406
grad_step = 000407, loss = 0.001413
grad_step = 000408, loss = 0.001414
grad_step = 000409, loss = 0.001407
grad_step = 000410, loss = 0.001398
grad_step = 000411, loss = 0.001387
grad_step = 000412, loss = 0.001379
grad_step = 000413, loss = 0.001376
grad_step = 000414, loss = 0.001377
grad_step = 000415, loss = 0.001381
grad_step = 000416, loss = 0.001385
grad_step = 000417, loss = 0.001390
grad_step = 000418, loss = 0.001393
grad_step = 000419, loss = 0.001399
grad_step = 000420, loss = 0.001400
grad_step = 000421, loss = 0.001402
grad_step = 000422, loss = 0.001397
grad_step = 000423, loss = 0.001392
grad_step = 000424, loss = 0.001381
grad_step = 000425, loss = 0.001372
grad_step = 000426, loss = 0.001363
grad_step = 000427, loss = 0.001358
grad_step = 000428, loss = 0.001355
grad_step = 000429, loss = 0.001356
grad_step = 000430, loss = 0.001358
grad_step = 000431, loss = 0.001361
grad_step = 000432, loss = 0.001365
grad_step = 000433, loss = 0.001368
grad_step = 000434, loss = 0.001372
grad_step = 000435, loss = 0.001376
grad_step = 000436, loss = 0.001385
grad_step = 000437, loss = 0.001392
grad_step = 000438, loss = 0.001403
grad_step = 000439, loss = 0.001408
grad_step = 000440, loss = 0.001412
grad_step = 000441, loss = 0.001402
grad_step = 000442, loss = 0.001386
grad_step = 000443, loss = 0.001361
grad_step = 000444, loss = 0.001342
grad_step = 000445, loss = 0.001332
grad_step = 000446, loss = 0.001335
grad_step = 000447, loss = 0.001344
grad_step = 000448, loss = 0.001353
grad_step = 000449, loss = 0.001358
grad_step = 000450, loss = 0.001354
grad_step = 000451, loss = 0.001348
grad_step = 000452, loss = 0.001337
grad_step = 000453, loss = 0.001328
grad_step = 000454, loss = 0.001321
grad_step = 000455, loss = 0.001318
grad_step = 000456, loss = 0.001319
grad_step = 000457, loss = 0.001321
grad_step = 000458, loss = 0.001324
grad_step = 000459, loss = 0.001327
grad_step = 000460, loss = 0.001331
grad_step = 000461, loss = 0.001333
grad_step = 000462, loss = 0.001336
grad_step = 000463, loss = 0.001337
grad_step = 000464, loss = 0.001340
grad_step = 000465, loss = 0.001339
grad_step = 000466, loss = 0.001342
grad_step = 000467, loss = 0.001338
grad_step = 000468, loss = 0.001335
grad_step = 000469, loss = 0.001328
grad_step = 000470, loss = 0.001321
grad_step = 000471, loss = 0.001312
grad_step = 000472, loss = 0.001304
grad_step = 000473, loss = 0.001297
grad_step = 000474, loss = 0.001293
grad_step = 000475, loss = 0.001291
grad_step = 000476, loss = 0.001291
grad_step = 000477, loss = 0.001292
grad_step = 000478, loss = 0.001294
grad_step = 000479, loss = 0.001298
grad_step = 000480, loss = 0.001303
grad_step = 000481, loss = 0.001313
grad_step = 000482, loss = 0.001324
grad_step = 000483, loss = 0.001343
grad_step = 000484, loss = 0.001359
grad_step = 000485, loss = 0.001387
grad_step = 000486, loss = 0.001394
grad_step = 000487, loss = 0.001396
grad_step = 000488, loss = 0.001363
grad_step = 000489, loss = 0.001321
grad_step = 000490, loss = 0.001284
grad_step = 000491, loss = 0.001270
grad_step = 000492, loss = 0.001280
grad_step = 000493, loss = 0.001301
grad_step = 000494, loss = 0.001320
grad_step = 000495, loss = 0.001317
grad_step = 000496, loss = 0.001302
grad_step = 000497, loss = 0.001277
grad_step = 000498, loss = 0.001262
grad_step = 000499, loss = 0.001260
grad_step = 000500, loss = 0.001269
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001280
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

  date_run                              2020-05-09 11:14:01.973622
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.225768
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 11:14:01.979408
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.142103
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 11:14:01.986440
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.120867
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 11:14:01.991825
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.15931
metric_name                                             r2_score
Name: 11, dtype: object 

  


### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

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

  


### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

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

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json 

                        date_run  ...            metric_name
0   2020-05-09 11:13:32.728870  ...    mean_absolute_error
1   2020-05-09 11:13:32.733478  ...     mean_squared_error
2   2020-05-09 11:13:32.737596  ...  median_absolute_error
3   2020-05-09 11:13:32.740801  ...               r2_score
4   2020-05-09 11:13:43.091200  ...    mean_absolute_error
5   2020-05-09 11:13:43.094950  ...     mean_squared_error
6   2020-05-09 11:13:43.098196  ...  median_absolute_error
7   2020-05-09 11:13:43.101297  ...               r2_score
8   2020-05-09 11:14:01.973622  ...    mean_absolute_error
9   2020-05-09 11:14:01.979408  ...     mean_squared_error
10  2020-05-09 11:14:01.986440  ...  median_absolute_error
11  2020-05-09 11:14:01.991825  ...               r2_score

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

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 306417.68it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 395953.54it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 548074.43it/s] 30%|███       | 3022848/9912422 [00:00<00:08, 772053.79it/s] 58%|█████▊    | 5750784/9912422 [00:00<00:03, 1086370.51it/s] 88%|████████▊ | 8732672/9912422 [00:01<00:00, 1520452.37it/s]9920512it [00:01, 9302659.67it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 144747.85it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 303653.76it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 391895.54it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 542420.45it/s]1654784it [00:00, 2742940.14it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53574.84it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff353435ba8> <class 'mlmodels.model_tch.torchhub.Model'>
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

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2f0b84ac8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3533f8e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2f0b84e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ##### 

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3534417b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff353441fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2f0b87080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff353441fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2f0b87080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ##### 

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff353441fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3534417b8> <class 'mlmodels.model_tch.torchhub.Model'>

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

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7faaf6728240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=ae52bcf80e44e1555e7aa54b0714773ea489c67782187924638d1d1f5ae1d626
  Stored in directory: /tmp/pip-ephem-wheel-cache-gzyj8dfo/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ##### 

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7faa8e30d128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   73728/17464789 [..............................] - ETA: 46s
  106496/17464789 [..............................] - ETA: 42s
  139264/17464789 [..............................] - ETA: 40s
  163840/17464789 [..............................] - ETA: 41s
  196608/17464789 [..............................] - ETA: 40s
  229376/17464789 [..............................] - ETA: 39s
  262144/17464789 [..............................] - ETA: 38s
  294912/17464789 [..............................] - ETA: 37s
  319488/17464789 [..............................] - ETA: 38s
  352256/17464789 [..............................] - ETA: 38s
  385024/17464789 [..............................] - ETA: 37s
  417792/17464789 [..............................] - ETA: 37s
  458752/17464789 [..............................] - ETA: 36s
  491520/17464789 [..............................] - ETA: 36s
  524288/17464789 [..............................] - ETA: 35s
  557056/17464789 [..............................] - ETA: 35s
  614400/17464789 [>.............................] - ETA: 33s
  647168/17464789 [>.............................] - ETA: 33s
  679936/17464789 [>.............................] - ETA: 33s
  720896/17464789 [>.............................] - ETA: 33s
  770048/17464789 [>.............................] - ETA: 32s
  802816/17464789 [>.............................] - ETA: 32s
  835584/17464789 [>.............................] - ETA: 32s
  892928/17464789 [>.............................] - ETA: 31s
  925696/17464789 [>.............................] - ETA: 31s
  974848/17464789 [>.............................] - ETA: 30s
 1015808/17464789 [>.............................] - ETA: 30s
 1064960/17464789 [>.............................] - ETA: 30s
 1114112/17464789 [>.............................] - ETA: 29s
 1171456/17464789 [=>............................] - ETA: 29s
 1204224/17464789 [=>............................] - ETA: 29s
 1253376/17464789 [=>............................] - ETA: 28s
 1310720/17464789 [=>............................] - ETA: 28s
 1359872/17464789 [=>............................] - ETA: 27s
 1417216/17464789 [=>............................] - ETA: 27s
 1466368/17464789 [=>............................] - ETA: 27s
 1515520/17464789 [=>............................] - ETA: 26s
 1572864/17464789 [=>............................] - ETA: 26s
 1622016/17464789 [=>............................] - ETA: 26s
 1654784/17464789 [=>............................] - ETA: 26s
 1695744/17464789 [=>............................] - ETA: 26s
 1744896/17464789 [=>............................] - ETA: 25s
 1777664/17464789 [==>...........................] - ETA: 25s
 1810432/17464789 [==>...........................] - ETA: 25s
 1867776/17464789 [==>...........................] - ETA: 25s
 1916928/17464789 [==>...........................] - ETA: 25s
 1949696/17464789 [==>...........................] - ETA: 25s
 2007040/17464789 [==>...........................] - ETA: 25s
 2039808/17464789 [==>...........................] - ETA: 25s
 2088960/17464789 [==>...........................] - ETA: 24s
 2146304/17464789 [==>...........................] - ETA: 24s
 2195456/17464789 [==>...........................] - ETA: 24s
 2244608/17464789 [==>...........................] - ETA: 24s
 2285568/17464789 [==>...........................] - ETA: 24s
 2334720/17464789 [===>..........................] - ETA: 24s
 2392064/17464789 [===>..........................] - ETA: 23s
 2441216/17464789 [===>..........................] - ETA: 23s
 2490368/17464789 [===>..........................] - ETA: 23s
 2547712/17464789 [===>..........................] - ETA: 23s
 2580480/17464789 [===>..........................] - ETA: 23s
 2629632/17464789 [===>..........................] - ETA: 23s
 2670592/17464789 [===>..........................] - ETA: 23s
 2719744/17464789 [===>..........................] - ETA: 23s
 2752512/17464789 [===>..........................] - ETA: 23s
 2785280/17464789 [===>..........................] - ETA: 23s
 2826240/17464789 [===>..........................] - ETA: 23s
 2875392/17464789 [===>..........................] - ETA: 23s
 2908160/17464789 [===>..........................] - ETA: 23s
 2965504/17464789 [====>.........................] - ETA: 22s
 2998272/17464789 [====>.........................] - ETA: 22s
 3047424/17464789 [====>.........................] - ETA: 22s
 3088384/17464789 [====>.........................] - ETA: 22s
 3137536/17464789 [====>.........................] - ETA: 22s
 3170304/17464789 [====>.........................] - ETA: 22s
 3227648/17464789 [====>.........................] - ETA: 22s
 3276800/17464789 [====>.........................] - ETA: 22s
 3309568/17464789 [====>.........................] - ETA: 22s
 3366912/17464789 [====>.........................] - ETA: 22s
 3416064/17464789 [====>.........................] - ETA: 21s
 3465216/17464789 [====>.........................] - ETA: 21s
 3522560/17464789 [=====>........................] - ETA: 21s
 3571712/17464789 [=====>........................] - ETA: 21s
 3620864/17464789 [=====>........................] - ETA: 21s
 3678208/17464789 [=====>........................] - ETA: 21s
 3727360/17464789 [=====>........................] - ETA: 21s
 3784704/17464789 [=====>........................] - ETA: 20s
 3833856/17464789 [=====>........................] - ETA: 20s
 3899392/17464789 [=====>........................] - ETA: 20s
 3956736/17464789 [=====>........................] - ETA: 20s
 4005888/17464789 [=====>........................] - ETA: 20s
 4063232/17464789 [=====>........................] - ETA: 20s
 4128768/17464789 [======>.......................] - ETA: 20s
 4177920/17464789 [======>.......................] - ETA: 19s
 4251648/17464789 [======>.......................] - ETA: 19s
 4300800/17464789 [======>.......................] - ETA: 19s
 4374528/17464789 [======>.......................] - ETA: 19s
 4423680/17464789 [======>.......................] - ETA: 19s
 4497408/17464789 [======>.......................] - ETA: 18s
 4562944/17464789 [======>.......................] - ETA: 18s
 4620288/17464789 [======>.......................] - ETA: 18s
 4685824/17464789 [=======>......................] - ETA: 18s
 4759552/17464789 [=======>......................] - ETA: 18s
 4825088/17464789 [=======>......................] - ETA: 18s
 4898816/17464789 [=======>......................] - ETA: 17s
 4964352/17464789 [=======>......................] - ETA: 17s
 5038080/17464789 [=======>......................] - ETA: 17s
 5103616/17464789 [=======>......................] - ETA: 17s
 5177344/17464789 [=======>......................] - ETA: 17s
 5242880/17464789 [========>.....................] - ETA: 17s
 5316608/17464789 [========>.....................] - ETA: 16s
 5382144/17464789 [========>.....................] - ETA: 16s
 5472256/17464789 [========>.....................] - ETA: 16s
 5537792/17464789 [========>.....................] - ETA: 16s
 5611520/17464789 [========>.....................] - ETA: 16s
 5677056/17464789 [========>.....................] - ETA: 15s
 5767168/17464789 [========>.....................] - ETA: 15s
 5832704/17464789 [=========>....................] - ETA: 15s
 5922816/17464789 [=========>....................] - ETA: 15s
 5971968/17464789 [=========>....................] - ETA: 15s
 6029312/17464789 [=========>....................] - ETA: 15s
 6078464/17464789 [=========>....................] - ETA: 15s
 6152192/17464789 [=========>....................] - ETA: 14s
 6201344/17464789 [=========>....................] - ETA: 14s
 6266880/17464789 [=========>....................] - ETA: 14s
 6324224/17464789 [=========>....................] - ETA: 14s
 6389760/17464789 [=========>....................] - ETA: 14s
 6447104/17464789 [==========>...................] - ETA: 14s
 6512640/17464789 [==========>...................] - ETA: 14s
 6586368/17464789 [==========>...................] - ETA: 14s
 6651904/17464789 [==========>...................] - ETA: 14s
 6725632/17464789 [==========>...................] - ETA: 13s
 6791168/17464789 [==========>...................] - ETA: 13s
 6864896/17464789 [==========>...................] - ETA: 13s
 6930432/17464789 [==========>...................] - ETA: 13s
 7004160/17464789 [===========>..................] - ETA: 13s
 7069696/17464789 [===========>..................] - ETA: 13s
 7102464/17464789 [===========>..................] - ETA: 13s
 7176192/17464789 [===========>..................] - ETA: 13s
 7241728/17464789 [===========>..................] - ETA: 13s
 7315456/17464789 [===========>..................] - ETA: 13s
 7405568/17464789 [===========>..................] - ETA: 12s
 7471104/17464789 [===========>..................] - ETA: 12s
 7544832/17464789 [===========>..................] - ETA: 12s
 7610368/17464789 [============>.................] - ETA: 12s
 7684096/17464789 [============>.................] - ETA: 12s
 7766016/17464789 [============>.................] - ETA: 12s
 7839744/17464789 [============>.................] - ETA: 12s
 7921664/17464789 [============>.................] - ETA: 11s
 7995392/17464789 [============>.................] - ETA: 11s
 8060928/17464789 [============>.................] - ETA: 11s
 8151040/17464789 [=============>................] - ETA: 11s
 8241152/17464789 [=============>................] - ETA: 11s
 8306688/17464789 [=============>................] - ETA: 11s
 8396800/17464789 [=============>................] - ETA: 11s
 8478720/17464789 [=============>................] - ETA: 10s
 8552448/17464789 [=============>................] - ETA: 10s
 8634368/17464789 [=============>................] - ETA: 10s
 8724480/17464789 [=============>................] - ETA: 10s
 8814592/17464789 [==============>...............] - ETA: 10s
 8896512/17464789 [==============>...............] - ETA: 10s
 8970240/17464789 [==============>...............] - ETA: 10s
 9052160/17464789 [==============>...............] - ETA: 10s
 9142272/17464789 [==============>...............] - ETA: 9s 
 9248768/17464789 [==============>...............] - ETA: 9s
 9330688/17464789 [===============>..............] - ETA: 9s
 9420800/17464789 [===============>..............] - ETA: 9s
 9510912/17464789 [===============>..............] - ETA: 9s
 9592832/17464789 [===============>..............] - ETA: 9s
 9666560/17464789 [===============>..............] - ETA: 9s
 9732096/17464789 [===============>..............] - ETA: 9s
 9805824/17464789 [===============>..............] - ETA: 8s
 9871360/17464789 [===============>..............] - ETA: 8s
 9945088/17464789 [================>.............] - ETA: 8s
10010624/17464789 [================>.............] - ETA: 8s
10084352/17464789 [================>.............] - ETA: 8s
10166272/17464789 [================>.............] - ETA: 8s
10240000/17464789 [================>.............] - ETA: 8s
10305536/17464789 [================>.............] - ETA: 8s
10395648/17464789 [================>.............] - ETA: 8s
10485760/17464789 [=================>............] - ETA: 7s
10551296/17464789 [=================>............] - ETA: 7s
10641408/17464789 [=================>............] - ETA: 7s
10723328/17464789 [=================>............] - ETA: 7s
10797056/17464789 [=================>............] - ETA: 7s
10887168/17464789 [=================>............] - ETA: 7s
10969088/17464789 [=================>............] - ETA: 7s
11042816/17464789 [=================>............] - ETA: 7s
11091968/17464789 [==================>...........] - ETA: 7s
11165696/17464789 [==================>...........] - ETA: 7s
11231232/17464789 [==================>...........] - ETA: 7s
11304960/17464789 [==================>...........] - ETA: 6s
11370496/17464789 [==================>...........] - ETA: 6s
11444224/17464789 [==================>...........] - ETA: 6s
11509760/17464789 [==================>...........] - ETA: 6s
11583488/17464789 [==================>...........] - ETA: 6s
11665408/17464789 [===================>..........] - ETA: 6s
11739136/17464789 [===================>..........] - ETA: 6s
11804672/17464789 [===================>..........] - ETA: 6s
11894784/17464789 [===================>..........] - ETA: 6s
11993088/17464789 [===================>..........] - ETA: 6s
12083200/17464789 [===================>..........] - ETA: 5s
12173312/17464789 [===================>..........] - ETA: 5s
12271616/17464789 [====================>.........] - ETA: 5s
12361728/17464789 [====================>.........] - ETA: 5s
12451840/17464789 [====================>.........] - ETA: 5s
12558336/17464789 [====================>.........] - ETA: 5s
12656640/17464789 [====================>.........] - ETA: 5s
12746752/17464789 [====================>.........] - ETA: 5s
12853248/17464789 [=====================>........] - ETA: 4s
12951552/17464789 [=====================>........] - ETA: 4s
13041664/17464789 [=====================>........] - ETA: 4s
13148160/17464789 [=====================>........] - ETA: 4s
13254656/17464789 [=====================>........] - ETA: 4s
13352960/17464789 [=====================>........] - ETA: 4s
13459456/17464789 [======================>.......] - ETA: 4s
13565952/17464789 [======================>.......] - ETA: 4s
13672448/17464789 [======================>.......] - ETA: 3s
13770752/17464789 [======================>.......] - ETA: 3s
13877248/17464789 [======================>.......] - ETA: 3s
13983744/17464789 [=======================>......] - ETA: 3s
14090240/17464789 [=======================>......] - ETA: 3s
14188544/17464789 [=======================>......] - ETA: 3s
14311424/17464789 [=======================>......] - ETA: 3s
14417920/17464789 [=======================>......] - ETA: 3s
14524416/17464789 [=======================>......] - ETA: 3s
14639104/17464789 [========================>.....] - ETA: 2s
14745600/17464789 [========================>.....] - ETA: 2s
14852096/17464789 [========================>.....] - ETA: 2s
14974976/17464789 [========================>.....] - ETA: 2s
15097856/17464789 [========================>.....] - ETA: 2s
15204352/17464789 [=========================>....] - ETA: 2s
15319040/17464789 [=========================>....] - ETA: 2s
15425536/17464789 [=========================>....] - ETA: 2s
15548416/17464789 [=========================>....] - ETA: 1s
15671296/17464789 [=========================>....] - ETA: 1s
15794176/17464789 [==========================>...] - ETA: 1s
15917056/17464789 [==========================>...] - ETA: 1s
16039936/17464789 [==========================>...] - ETA: 1s
16154624/17464789 [==========================>...] - ETA: 1s
16261120/17464789 [==========================>...] - ETA: 1s
16384000/17464789 [===========================>..] - ETA: 1s
16523264/17464789 [===========================>..] - ETA: 0s
16646144/17464789 [===========================>..] - ETA: 0s
16769024/17464789 [===========================>..] - ETA: 0s
16891904/17464789 [============================>.] - ETA: 0s
17014784/17464789 [============================>.] - ETA: 0s
17129472/17464789 [============================>.] - ETA: 0s
17268736/17464789 [============================>.] - ETA: 0s
17391616/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 16s 1us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 11:15:47.551990: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 11:15:47.556452: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095205000 Hz
2020-05-09 11:15:47.556615: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b1982e8670 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 11:15:47.556626: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6697 - accuracy: 0.4998
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6469 - accuracy: 0.5013
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6609 - accuracy: 0.5004
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7058 - accuracy: 0.4974
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7111 - accuracy: 0.4971
11000/25000 [============>.................] - ETA: 3s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7008 - accuracy: 0.4978
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6810 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6883 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6811 - accuracy: 0.4991
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6797 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6739 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6940 - accuracy: 0.4982
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 7s 282us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 11:16:01.289770
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 11:16:01.289770  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}] 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ##### 

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
2020-05-09 11:16:07.364658: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 11:16:07.371844: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095205000 Hz
2020-05-09 11:16:07.372146: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561064514460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 11:16:07.372220: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f87dd024d30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6241 - crf_viterbi_accuracy: 0.3200 - val_loss: 1.5675 - val_crf_viterbi_accuracy: 0.2933

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ##### 

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f87d23cef60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6688 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6912 - accuracy: 0.4984
11000/25000 [============>.................] - ETA: 3s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6852 - accuracy: 0.4988
15000/25000 [=================>............] - ETA: 2s - loss: 7.6809 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7059 - accuracy: 0.4974
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6937 - accuracy: 0.4982
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6862 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6797 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6785 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'} {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'} 

  #### Setup Model   ############################################## 

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}} Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range 

  


### Running {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1} {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f878e148048> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ##### 

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<91:11:43, 2.63kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:03<63:49:25, 3.75kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.79M/862M [00:03<44:24:03, 5.36kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:03<30:45:03, 7.65kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:03<21:16:22, 10.9kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:03<14:43:34, 15.6kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:03<10:12:17, 22.3kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:03<7:03:48, 31.9kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:04<5:03:55, 44.4kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<3:32:13, 63.4kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:05<13:15:34, 16.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<9:15:54, 24.1kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:07<6:31:10, 34.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:07<4:34:57, 48.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<3:11:59, 69.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:09<2:20:15, 94.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<1:39:20, 134kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:11<1:11:09, 186kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:11<50:35, 262kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:13<37:16, 353kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:13<27:17, 482kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<19:14, 682kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:15<17:24, 752kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:15<13:31, 969kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<09:39, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:17<10:06, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:17<07:58, 1.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<05:47, 2.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:19<07:32, 1.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:19<06:42, 1.93MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<04:54, 2.64MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:21<06:47, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:21<06:34, 1.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<04:48, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:23<06:33, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:23<05:55, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:23<04:23, 2.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:25<06:02, 2.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:25<05:17, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:25<03:52, 3.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<06:54, 1.84MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<06:37, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<04:49, 2.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<06:53, 1.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<06:36, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<04:49, 2.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:31<06:32, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:38, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<04:05, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<22:02, 567kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<7:18:55, 28.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<5:07:10, 40.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<3:34:34, 58.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<2:37:11, 79.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<1:50:59, 112kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<1:17:36, 160kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<59:07, 209kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<42:22, 292kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<29:43, 414kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<25:54, 475kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:38<19:08, 642kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<13:30, 907kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<14:39, 835kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<11:18, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:01, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<10:55, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<08:40, 1.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:12, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<09:07, 1.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<07:24, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:18, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<08:44, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<07:05, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:04, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<10:11, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<08:07, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:48, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<09:53, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<07:44, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:35, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<07:30, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<06:11, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:30, 2.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<06:26, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<06:15, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<04:39, 2.52MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:24, 3.43MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:56<15:34, 750kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<11:51, 984kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:25, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<15:03, 772kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<6:30:52, 29.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<4:33:13, 42.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<3:13:05, 59.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<2:16:07, 84.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:35:05, 121kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<1:10:58, 162kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<50:36, 227kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<35:24, 323kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<31:13, 366kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<22:48, 500kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<16:01, 710kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<17:59, 631kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<13:31, 839kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:33, 1.18MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<12:53, 875kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<09:57, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:04, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<10:10, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<08:05, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:45, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<10:10, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<08:01, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:43, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<10:16, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<08:05, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:45, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<09:39, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<07:39, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:27, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<10:07, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<07:58, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:40, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<10:31, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<08:46, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<06:16, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<07:36, 1.42MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<06:12, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:45, 2.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<6:13:36, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<4:21:30, 41.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<3:02:29, 58.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<2:15:00, 79.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<1:35:19, 112kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<1:06:34, 160kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<51:54, 205kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<37:17, 285kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<26:09, 404kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<21:55, 481kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<16:12, 651kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<11:24, 920kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<13:33, 773kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<10:45, 973kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:39, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<08:29, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<06:58, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:00, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<06:36, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:57, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<07:23, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<06:00, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:19, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:46, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:08, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:25, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<06:07, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<05:14, 1.93MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:46, 2.67MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<06:30, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<05:22, 1.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:51, 2.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<09:17, 1.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<07:18, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:12, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<09:59, 994kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<08:07, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:47, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:06, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<5:35:46, 29.4kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<3:54:53, 42.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<2:43:41, 60.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<2:40:19, 61.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<1:52:56, 86.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<1:18:45, 124kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<1:02:13, 157kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<44:19, 220kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<30:58, 313kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<27:54, 347kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<20:25, 473kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<14:21, 670kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<13:38, 704kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<10:19, 930kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:17, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<11:15, 847kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<08:39, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:07, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<11:29, 824kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<08:48, 1.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:16, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<07:26, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<05:58, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:16, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:03<07:44, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<06:10, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:23, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:05<10:17, 900kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<07:56, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:38, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<09:08, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<07:08, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:03, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<11:38, 784kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<08:53, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:17, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<11:30, 787kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<08:48, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:29, 1.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<5:09:05, 29.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<3:36:24, 41.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<2:30:40, 59.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<2:07:00, 70.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<1:29:33, 99.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<1:02:25, 142kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<51:43, 171kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<37:28, 237kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<26:12, 337kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<21:55, 401kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<15:58, 550kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<11:13, 780kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<12:37, 691kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<09:28, 921kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:41, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<09:40, 895kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<07:33, 1.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:21, 1.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<08:09, 1.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<06:26, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:32, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<1:20:58, 105kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<57:21, 148kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<39:56, 212kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<1:28:01, 96.1kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<1:02:15, 136kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<43:21, 194kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<54:42, 153kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<39:30, 212kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<27:36, 302kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<22:24, 371kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<16:22, 508kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<11:27, 721kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<19:28, 423kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<14:50, 556kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<10:25, 787kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<10:54, 750kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<08:17, 985kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:38<06:59, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<05:32, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:10, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<4:39:54, 28.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<3:15:26, 41.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<2:17:51, 58.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<1:37:04, 82.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<1:08:37, 116kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<48:38, 163kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<34:57, 225kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<25:06, 313kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<18:36, 419kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<13:39, 570kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<10:38, 725kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<08:04, 955kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<06:45, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<05:25, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:51, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<07:13, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:41, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<05:04, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:10, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<04:00, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:25, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<03:29, 2.11MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<02:58, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:08, 3.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<08:39, 844kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<06:38, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:54, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<4:09:27, 29.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<2:54:12, 41.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<2:02:40, 58.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<1:26:22, 83.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<1:01:00, 117kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<43:15, 165kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<31:04, 227kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<22:18, 316kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<16:31, 423kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<12:10, 573kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<08:31, 813kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<13:42, 505kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<10:09, 680kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<08:04, 848kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<06:12, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<05:19, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<04:17, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:58, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:20, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:18, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<02:51, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<02:58, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<02:37, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<01:52, 3.46MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<33:13, 196kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<23:45, 273kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<16:49, 384kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<3:35:31, 30.0kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<2:30:18, 42.8kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<1:46:01, 60.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<1:15:04, 85.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<52:14, 121kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<39:20, 161kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<28:27, 222kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<19:51, 316kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<16:16, 384kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<12:11, 512kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<08:32, 726kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<08:40, 712kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<06:53, 896kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:51, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<05:59, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:35<05:00, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:32, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<04:57, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<04:19, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:04, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<04:37, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<04:03, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:53, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<04:28, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<03:58, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:49, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<04:52, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<03:53, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:44, 2.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<13:26, 429kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<09:53, 583kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<06:54, 826kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<13:29, 422kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<09:54, 574kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:07, 794kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<3:09:55, 29.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<2:12:15, 42.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<1:33:27, 59.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<1:05:47, 84.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<46:22, 119kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<32:52, 167kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<22:48, 239kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<50:47, 107kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<35:57, 151kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<25:41, 209kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<18:24, 292kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<12:46, 415kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<5:48:17, 15.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<4:03:55, 21.7kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<2:48:52, 31.0kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<2:22:09, 36.9kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<1:40:09, 52.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<1:09:32, 74.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<50:50, 102kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<35:57, 144kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<25:38, 199kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<18:20, 278kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<13:26, 374kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<09:44, 516kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<06:49, 730kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<07:00, 709kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<05:18, 934kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:42, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<1:20:51, 60.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<56:55, 85.8kB/s]  .vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<39:43, 122kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<3:04:45, 26.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<2:08:35, 37.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<1:30:29, 52.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<1:03:38, 75.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<44:03, 107kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<40:31, 116kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<28:42, 164kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<19:53, 234kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<24:53, 186kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<17:46, 261kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<12:19, 371kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<26:53, 170kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<19:27, 235kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<13:32, 334kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<11:28, 393kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<08:23, 536kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<05:51, 759kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<07:19, 606kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<05:30, 804kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:51, 1.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<05:45, 758kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<04:23, 993kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:04, 1.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<05:32, 775kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<04:11, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:56, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<04:39, 907kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<03:33, 1.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:30, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<04:18, 964kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<03:22, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:30, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<2:13:56, 30.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<1:33:04, 43.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<1:05:38, 61.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<46:12, 87.4kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<32:00, 125kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<24:29, 162kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<17:27, 227kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<12:03, 324kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<25:06, 156kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<17:52, 218kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<12:21, 311kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<21:33, 178kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<15:26, 248kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<10:43, 353kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<09:13, 409kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<06:45, 557kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<05:12, 711kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<03:56, 937kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:44, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<3:54:13, 15.5kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<2:43:55, 22.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<1:53:20, 31.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<1:19:28, 44.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<55:11, 63.3kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<39:06, 89.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<27:03, 127kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<20:26, 168kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<14:33, 235kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<10:12, 331kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<1:56:26, 29.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<1:20:38, 41.4kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<56:56, 58.2kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<40:04, 82.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<27:40, 118kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<21:10, 153kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<15:04, 215kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<10:23, 306kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<15:43, 202kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<11:14, 282kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<08:10, 380kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<05:58, 519kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<04:33, 667kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<03:26, 882kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<02:48, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<02:11, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:56, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:34, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:06, 2.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<02:36, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:03, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:49, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:30, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:26, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:13, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<00:57, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<1:27:41, 30.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<1:00:42, 43.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<42:31, 60.6kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<29:54, 86.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<20:31, 123kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<16:42, 150kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<12:36, 199kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<09:33, 262kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<07:17, 343kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<05:37, 444kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<04:26, 562kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<03:32, 704kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<02:55, 851kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<02:25, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<02:05, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:47, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:35, 1.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:24, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<01:16, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<01:09, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<01:04, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<00:59, 2.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<00:55, 2.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<00:53, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<00:53, 2.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<00:48, 3.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<00:48, 3.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<00:44, 3.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<00:43, 3.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<00:41, 3.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<00:38, 3.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:37, 3.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:34, 4.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<00:35, 4.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<07:48, 304kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<05:50, 406kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<04:14, 557kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<03:06, 757kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<02:17, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:45, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:21, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:05, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:53, 2.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<02:29, 924kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:58, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:29, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:08, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:56, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:45, 2.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:37, 3.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:53, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:31, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:13, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:57, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:45, 2.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:40, 3.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:37, 3.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:37, 3.48MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:37, 3.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:36, 3.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<17:00, 127kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<12:15, 176kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<08:43, 247kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<06:14, 344kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<04:30, 475kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<03:17, 647kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<02:26, 870kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:50, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:25, 1.48MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:07, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<02:27, 853kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<02:02, 1.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:33, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:12, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:57, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:47, 2.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:40, 3.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:34, 3.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:30, 3.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<03:46, 537kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<02:56, 689kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<02:09, 935kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:35, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:12, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:44, 2.64MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:36, 3.23MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<07:53, 248kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<05:42, 342kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<04:02, 479kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:54, 662kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:06, 910kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<02:18, 831kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<41:28, 46.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<29:00, 65.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<20:14, 93.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<14:09, 133kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<09:55, 188kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<06:58, 266kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<06:15, 295kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<04:32, 404kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<03:13, 565kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<02:18, 785kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:40, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:13, 1.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<03:03, 581kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:17, 770kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:39, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:12, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:53, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:31, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:13, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:54, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:40, 2.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:31, 3.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:56, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:41, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:32, 2.93MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:25, 3.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:34, 993kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:17, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:56, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:41, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:31, 2.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:08, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:55, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:40, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:30, 2.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:23, 3.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<02:56, 486kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<02:10, 655kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:32, 912kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:05, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<01:24, 971kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<01:06, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:47, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:35, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:26, 2.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<02:10, 593kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:37, 791kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:08, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:49, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<01:07, 1.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:52, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:38, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:33, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<31:39, 37.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<22:00, 53.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<15:07, 75.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<10:25, 108kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<07:51, 141kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<05:35, 197kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<03:51, 280kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<02:40, 396kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<02:18, 452kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<01:41, 613kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:10, 859kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:49, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:58, 989kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:45, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:49, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<00:38, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:27, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:19, 2.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:42, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:33, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:17, 2.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:43, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:34, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:23, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:16, 2.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<01:00, 696kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:44, 919kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:31, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:21, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<01:28, 428kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<01:04, 577kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:43, 811kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:29, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<04:33, 123kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<03:11, 173kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<02:06, 246kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<01:32, 319kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<01:06, 435kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:44, 615kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:34, 782kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<14:01, 31.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<09:27, 45.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<06:00, 64.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<04:21, 86.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<03:02, 122kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:55, 174kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<01:20, 229kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:56, 320kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:34, 454kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:26, 531kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:20, 699kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 986kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:10, 969kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:07, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:04, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:03, 1.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:01, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.70MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 886/400000 [00:00<00:45, 8851.71it/s]  0%|          | 1756/400000 [00:00<00:45, 8803.18it/s]  1%|          | 2638/400000 [00:00<00:45, 8806.04it/s]  1%|          | 3522/400000 [00:00<00:44, 8813.53it/s]  1%|          | 4411/400000 [00:00<00:44, 8833.66it/s]  1%|▏         | 5289/400000 [00:00<00:44, 8816.17it/s]  2%|▏         | 6114/400000 [00:00<00:45, 8635.34it/s]  2%|▏         | 6986/400000 [00:00<00:45, 8658.48it/s]  2%|▏         | 7867/400000 [00:00<00:45, 8701.13it/s]  2%|▏         | 8746/400000 [00:01<00:44, 8724.88it/s]  2%|▏         | 9631/400000 [00:01<00:44, 8759.61it/s]  3%|▎         | 10493/400000 [00:01<00:44, 8715.03it/s]  3%|▎         | 11368/400000 [00:01<00:44, 8723.25it/s]  3%|▎         | 12246/400000 [00:01<00:44, 8739.19it/s]  3%|▎         | 13115/400000 [00:01<00:44, 8713.85it/s]  4%|▎         | 14004/400000 [00:01<00:44, 8763.08it/s]  4%|▎         | 14878/400000 [00:01<00:44, 8706.87it/s]  4%|▍         | 15752/400000 [00:01<00:44, 8715.73it/s]  4%|▍         | 16623/400000 [00:01<00:44, 8622.30it/s]  4%|▍         | 17497/400000 [00:02<00:44, 8655.54it/s]  5%|▍         | 18383/400000 [00:02<00:43, 8715.76it/s]  5%|▍         | 19255/400000 [00:02<00:44, 8613.12it/s]  5%|▌         | 20136/400000 [00:02<00:43, 8668.44it/s]  5%|▌         | 21012/400000 [00:02<00:43, 8695.60it/s]  5%|▌         | 21906/400000 [00:02<00:43, 8764.86it/s]  6%|▌         | 22785/400000 [00:02<00:43, 8771.00it/s]  6%|▌         | 23682/400000 [00:02<00:42, 8827.46it/s]  6%|▌         | 24565/400000 [00:02<00:42, 8800.14it/s]  6%|▋         | 25458/400000 [00:02<00:42, 8837.16it/s]  7%|▋         | 26342/400000 [00:03<00:42, 8749.31it/s]  7%|▋         | 27234/400000 [00:03<00:42, 8797.43it/s]  7%|▋         | 28115/400000 [00:03<00:42, 8748.12it/s]  7%|▋         | 28991/400000 [00:03<00:43, 8525.38it/s]  7%|▋         | 29879/400000 [00:03<00:42, 8627.99it/s]  8%|▊         | 30774/400000 [00:03<00:42, 8720.73it/s]  8%|▊         | 31668/400000 [00:03<00:41, 8782.44it/s]  8%|▊         | 32560/400000 [00:03<00:41, 8820.94it/s]  8%|▊         | 33443/400000 [00:03<00:41, 8761.06it/s]  9%|▊         | 34320/400000 [00:03<00:41, 8762.43it/s]  9%|▉         | 35197/400000 [00:04<00:41, 8706.72it/s]  9%|▉         | 36069/400000 [00:04<00:41, 8710.12it/s]  9%|▉         | 36962/400000 [00:04<00:41, 8773.65it/s]  9%|▉         | 37840/400000 [00:04<00:42, 8620.38it/s] 10%|▉         | 38745/400000 [00:04<00:41, 8742.45it/s] 10%|▉         | 39647/400000 [00:04<00:40, 8822.39it/s] 10%|█         | 40531/400000 [00:04<00:41, 8671.02it/s] 10%|█         | 41400/400000 [00:04<00:41, 8676.69it/s] 11%|█         | 42269/400000 [00:04<00:41, 8664.70it/s] 11%|█         | 43163/400000 [00:04<00:40, 8743.43it/s] 11%|█         | 44055/400000 [00:05<00:40, 8794.46it/s] 11%|█         | 44952/400000 [00:05<00:40, 8845.38it/s] 11%|█▏        | 45848/400000 [00:05<00:39, 8877.01it/s] 12%|█▏        | 46737/400000 [00:05<00:39, 8832.98it/s] 12%|█▏        | 47630/400000 [00:05<00:39, 8859.38it/s] 12%|█▏        | 48521/400000 [00:05<00:39, 8874.48it/s] 12%|█▏        | 49419/400000 [00:05<00:39, 8905.82it/s] 13%|█▎        | 50319/400000 [00:05<00:39, 8932.24it/s] 13%|█▎        | 51213/400000 [00:05<00:39, 8854.61it/s] 13%|█▎        | 52104/400000 [00:05<00:39, 8869.15it/s] 13%|█▎        | 52997/400000 [00:06<00:39, 8886.04it/s] 13%|█▎        | 53886/400000 [00:06<00:38, 8884.92it/s] 14%|█▎        | 54786/400000 [00:06<00:38, 8916.94it/s] 14%|█▍        | 55678/400000 [00:06<00:38, 8885.12it/s] 14%|█▍        | 56582/400000 [00:06<00:38, 8928.35it/s] 14%|█▍        | 57475/400000 [00:06<00:38, 8912.03it/s] 15%|█▍        | 58368/400000 [00:06<00:38, 8915.22it/s] 15%|█▍        | 59272/400000 [00:06<00:38, 8951.02it/s] 15%|█▌        | 60168/400000 [00:06<00:38, 8893.46it/s] 15%|█▌        | 61058/400000 [00:06<00:38, 8872.26it/s] 15%|█▌        | 61946/400000 [00:07<00:38, 8758.39it/s] 16%|█▌        | 62846/400000 [00:07<00:38, 8828.49it/s] 16%|█▌        | 63749/400000 [00:07<00:37, 8885.42it/s] 16%|█▌        | 64638/400000 [00:07<00:37, 8853.13it/s] 16%|█▋        | 65531/400000 [00:07<00:37, 8874.33it/s] 17%|█▋        | 66430/400000 [00:07<00:37, 8908.06it/s] 17%|█▋        | 67329/400000 [00:07<00:37, 8930.75it/s] 17%|█▋        | 68223/400000 [00:07<00:37, 8931.16it/s] 17%|█▋        | 69117/400000 [00:07<00:37, 8864.85it/s] 18%|█▊        | 70010/400000 [00:07<00:37, 8883.02it/s] 18%|█▊        | 70914/400000 [00:08<00:36, 8926.53it/s] 18%|█▊        | 71811/400000 [00:08<00:36, 8938.35it/s] 18%|█▊        | 72705/400000 [00:08<00:37, 8773.44it/s] 18%|█▊        | 73585/400000 [00:08<00:37, 8779.29it/s] 19%|█▊        | 74474/400000 [00:08<00:36, 8812.04it/s] 19%|█▉        | 75379/400000 [00:08<00:36, 8879.25it/s] 19%|█▉        | 76271/400000 [00:08<00:36, 8888.65it/s] 19%|█▉        | 77161/400000 [00:08<00:36, 8890.43it/s] 20%|█▉        | 78051/400000 [00:08<00:36, 8861.69it/s] 20%|█▉        | 78938/400000 [00:08<00:36, 8832.28it/s] 20%|█▉        | 79822/400000 [00:09<00:36, 8770.44it/s] 20%|██        | 80715/400000 [00:09<00:36, 8816.24it/s] 20%|██        | 81609/400000 [00:09<00:35, 8851.60it/s] 21%|██        | 82496/400000 [00:09<00:35, 8856.88it/s] 21%|██        | 83389/400000 [00:09<00:35, 8876.45it/s] 21%|██        | 84285/400000 [00:09<00:35, 8899.49it/s] 21%|██▏       | 85176/400000 [00:09<00:35, 8879.62it/s] 22%|██▏       | 86085/400000 [00:09<00:35, 8941.24it/s] 22%|██▏       | 86980/400000 [00:09<00:35, 8793.14it/s] 22%|██▏       | 87860/400000 [00:09<00:35, 8676.56it/s] 22%|██▏       | 88747/400000 [00:10<00:35, 8732.98it/s] 22%|██▏       | 89631/400000 [00:10<00:35, 8760.41it/s] 23%|██▎       | 90512/400000 [00:10<00:35, 8773.83it/s] 23%|██▎       | 91390/400000 [00:10<00:35, 8748.35it/s] 23%|██▎       | 92266/400000 [00:10<00:35, 8692.64it/s] 23%|██▎       | 93151/400000 [00:10<00:35, 8738.31it/s] 24%|██▎       | 94047/400000 [00:10<00:34, 8801.02it/s] 24%|██▎       | 94928/400000 [00:10<00:35, 8716.27it/s] 24%|██▍       | 95801/400000 [00:10<00:35, 8671.33it/s] 24%|██▍       | 96696/400000 [00:10<00:34, 8752.66it/s] 24%|██▍       | 97585/400000 [00:11<00:34, 8791.84it/s] 25%|██▍       | 98477/400000 [00:11<00:34, 8828.97it/s] 25%|██▍       | 99366/400000 [00:11<00:33, 8845.78it/s] 25%|██▌       | 100251/400000 [00:11<00:33, 8836.42it/s] 25%|██▌       | 101135/400000 [00:11<00:33, 8800.67it/s] 26%|██▌       | 102020/400000 [00:11<00:33, 8813.53it/s] 26%|██▌       | 102902/400000 [00:11<00:34, 8711.74it/s] 26%|██▌       | 103796/400000 [00:11<00:33, 8777.60it/s] 26%|██▌       | 104686/400000 [00:11<00:33, 8811.66it/s] 26%|██▋       | 105568/400000 [00:12<00:33, 8688.94it/s] 27%|██▋       | 106458/400000 [00:12<00:33, 8748.44it/s] 27%|██▋       | 107349/400000 [00:12<00:33, 8796.24it/s] 27%|██▋       | 108250/400000 [00:12<00:32, 8857.60it/s] 27%|██▋       | 109143/400000 [00:12<00:32, 8874.34it/s] 28%|██▊       | 110031/400000 [00:12<00:32, 8837.50it/s] 28%|██▊       | 110920/400000 [00:12<00:32, 8853.16it/s] 28%|██▊       | 111808/400000 [00:12<00:32, 8860.71it/s] 28%|██▊       | 112705/400000 [00:12<00:32, 8892.05it/s] 28%|██▊       | 113595/400000 [00:12<00:32, 8892.60it/s] 29%|██▊       | 114485/400000 [00:13<00:32, 8850.86it/s] 29%|██▉       | 115385/400000 [00:13<00:31, 8894.57it/s] 29%|██▉       | 116277/400000 [00:13<00:31, 8901.96it/s] 29%|██▉       | 117168/400000 [00:13<00:31, 8896.08it/s] 30%|██▉       | 118063/400000 [00:13<00:31, 8911.96it/s] 30%|██▉       | 118955/400000 [00:13<00:31, 8878.57it/s] 30%|██▉       | 119843/400000 [00:13<00:31, 8864.37it/s] 30%|███       | 120743/400000 [00:13<00:31, 8902.21it/s] 30%|███       | 121634/400000 [00:13<00:31, 8860.71it/s] 31%|███       | 122521/400000 [00:13<00:31, 8836.81it/s] 31%|███       | 123405/400000 [00:14<00:31, 8796.79it/s] 31%|███       | 124291/400000 [00:14<00:31, 8815.55it/s] 31%|███▏      | 125173/400000 [00:14<00:31, 8746.09it/s] 32%|███▏      | 126064/400000 [00:14<00:31, 8792.75it/s] 32%|███▏      | 126962/400000 [00:14<00:30, 8845.86it/s] 32%|███▏      | 127847/400000 [00:14<00:30, 8833.29it/s] 32%|███▏      | 128733/400000 [00:14<00:30, 8839.09it/s] 32%|███▏      | 129618/400000 [00:14<00:30, 8789.11it/s] 33%|███▎      | 130508/400000 [00:14<00:30, 8819.40it/s] 33%|███▎      | 131391/400000 [00:14<00:30, 8822.21it/s] 33%|███▎      | 132274/400000 [00:15<00:30, 8758.60it/s] 33%|███▎      | 133165/400000 [00:15<00:30, 8802.25it/s] 34%|███▎      | 134046/400000 [00:15<00:30, 8797.95it/s] 34%|███▎      | 134944/400000 [00:15<00:29, 8851.36it/s] 34%|███▍      | 135839/400000 [00:15<00:29, 8879.86it/s] 34%|███▍      | 136728/400000 [00:15<00:29, 8784.91it/s] 34%|███▍      | 137614/400000 [00:15<00:29, 8805.69it/s] 35%|███▍      | 138515/400000 [00:15<00:29, 8863.22it/s] 35%|███▍      | 139407/400000 [00:15<00:29, 8879.61it/s] 35%|███▌      | 140304/400000 [00:15<00:29, 8906.19it/s] 35%|███▌      | 141195/400000 [00:16<00:29, 8799.59it/s] 36%|███▌      | 142090/400000 [00:16<00:29, 8843.97it/s] 36%|███▌      | 142975/400000 [00:16<00:29, 8722.54it/s] 36%|███▌      | 143877/400000 [00:16<00:29, 8809.43it/s] 36%|███▌      | 144773/400000 [00:16<00:28, 8853.04it/s] 36%|███▋      | 145659/400000 [00:16<00:28, 8825.43it/s] 37%|███▋      | 146542/400000 [00:16<00:28, 8740.23it/s] 37%|███▋      | 147432/400000 [00:16<00:28, 8785.73it/s] 37%|███▋      | 148327/400000 [00:16<00:28, 8832.47it/s] 37%|███▋      | 149212/400000 [00:16<00:28, 8835.22it/s] 38%|███▊      | 150096/400000 [00:17<00:28, 8810.93it/s] 38%|███▊      | 150978/400000 [00:17<00:28, 8794.22it/s] 38%|███▊      | 151870/400000 [00:17<00:28, 8829.73it/s] 38%|███▊      | 152762/400000 [00:17<00:27, 8855.42it/s] 38%|███▊      | 153659/400000 [00:17<00:27, 8889.15it/s] 39%|███▊      | 154549/400000 [00:17<00:27, 8883.18it/s] 39%|███▉      | 155438/400000 [00:17<00:27, 8875.74it/s] 39%|███▉      | 156345/400000 [00:17<00:27, 8931.94it/s] 39%|███▉      | 157239/400000 [00:17<00:27, 8845.62it/s] 40%|███▉      | 158124/400000 [00:17<00:27, 8758.13it/s] 40%|███▉      | 159001/400000 [00:18<00:28, 8534.01it/s] 40%|███▉      | 159894/400000 [00:18<00:27, 8648.37it/s] 40%|████      | 160786/400000 [00:18<00:27, 8727.72it/s] 40%|████      | 161660/400000 [00:18<00:27, 8694.88it/s] 41%|████      | 162559/400000 [00:18<00:27, 8778.15it/s] 41%|████      | 163443/400000 [00:18<00:26, 8794.42it/s] 41%|████      | 164337/400000 [00:18<00:26, 8835.46it/s] 41%|████▏     | 165232/400000 [00:18<00:26, 8867.49it/s] 42%|████▏     | 166124/400000 [00:18<00:26, 8881.64it/s] 42%|████▏     | 167020/400000 [00:18<00:26, 8904.99it/s] 42%|████▏     | 167911/400000 [00:19<00:26, 8859.44it/s] 42%|████▏     | 168798/400000 [00:19<00:26, 8861.79it/s] 42%|████▏     | 169700/400000 [00:19<00:25, 8908.02it/s] 43%|████▎     | 170591/400000 [00:19<00:25, 8892.70it/s] 43%|████▎     | 171482/400000 [00:19<00:25, 8897.80it/s] 43%|████▎     | 172372/400000 [00:19<00:25, 8860.82it/s] 43%|████▎     | 173260/400000 [00:19<00:25, 8865.45it/s] 44%|████▎     | 174147/400000 [00:19<00:25, 8856.32it/s] 44%|████▍     | 175034/400000 [00:19<00:25, 8860.10it/s] 44%|████▍     | 175924/400000 [00:19<00:25, 8871.17it/s] 44%|████▍     | 176812/400000 [00:20<00:25, 8760.74it/s] 44%|████▍     | 177689/400000 [00:20<00:25, 8740.14it/s] 45%|████▍     | 178581/400000 [00:20<00:25, 8791.08it/s] 45%|████▍     | 179478/400000 [00:20<00:24, 8843.54it/s] 45%|████▌     | 180364/400000 [00:20<00:24, 8847.28it/s] 45%|████▌     | 181249/400000 [00:20<00:24, 8801.36it/s] 46%|████▌     | 182130/400000 [00:20<00:25, 8690.87it/s] 46%|████▌     | 183032/400000 [00:20<00:24, 8784.58it/s] 46%|████▌     | 183931/400000 [00:20<00:24, 8843.86it/s] 46%|████▌     | 184816/400000 [00:20<00:24, 8812.24it/s] 46%|████▋     | 185698/400000 [00:21<00:24, 8805.58it/s] 47%|████▋     | 186579/400000 [00:21<00:24, 8672.89it/s] 47%|████▋     | 187452/400000 [00:21<00:24, 8688.48it/s] 47%|████▋     | 188340/400000 [00:21<00:24, 8743.78it/s] 47%|████▋     | 189238/400000 [00:21<00:23, 8812.23it/s] 48%|████▊     | 190129/400000 [00:21<00:23, 8841.09it/s] 48%|████▊     | 191014/400000 [00:21<00:23, 8798.26it/s] 48%|████▊     | 191898/400000 [00:21<00:23, 8810.39it/s] 48%|████▊     | 192791/400000 [00:21<00:23, 8844.07it/s] 48%|████▊     | 193678/400000 [00:21<00:23, 8849.50it/s] 49%|████▊     | 194567/400000 [00:22<00:23, 8860.95it/s] 49%|████▉     | 195454/400000 [00:22<00:23, 8801.44it/s] 49%|████▉     | 196341/400000 [00:22<00:23, 8820.33it/s] 49%|████▉     | 197232/400000 [00:22<00:22, 8845.43it/s] 50%|████▉     | 198117/400000 [00:22<00:22, 8825.38it/s] 50%|████▉     | 199015/400000 [00:22<00:22, 8869.79it/s] 50%|████▉     | 199903/400000 [00:22<00:22, 8819.31it/s] 50%|█████     | 200794/400000 [00:22<00:22, 8845.96it/s] 50%|█████     | 201679/400000 [00:22<00:22, 8788.78it/s] 51%|█████     | 202559/400000 [00:22<00:22, 8765.23it/s] 51%|█████     | 203444/400000 [00:23<00:22, 8790.21it/s] 51%|█████     | 204330/400000 [00:23<00:22, 8808.98it/s] 51%|█████▏    | 205226/400000 [00:23<00:22, 8852.43it/s] 52%|█████▏    | 206112/400000 [00:23<00:22, 8789.44it/s] 52%|█████▏    | 206998/400000 [00:23<00:21, 8810.34it/s] 52%|█████▏    | 207891/400000 [00:23<00:21, 8843.34it/s] 52%|█████▏    | 208778/400000 [00:23<00:21, 8848.80it/s] 52%|█████▏    | 209664/400000 [00:23<00:21, 8850.54it/s] 53%|█████▎    | 210563/400000 [00:23<00:21, 8889.94it/s] 53%|█████▎    | 211453/400000 [00:24<00:21, 8889.57it/s] 53%|█████▎    | 212348/400000 [00:24<00:21, 8905.81it/s] 53%|█████▎    | 213239/400000 [00:24<00:21, 8867.91it/s] 54%|█████▎    | 214126/400000 [00:24<00:21, 8830.38it/s] 54%|█████▍    | 215019/400000 [00:24<00:20, 8857.25it/s] 54%|█████▍    | 215905/400000 [00:24<00:20, 8839.50it/s] 54%|█████▍    | 216793/400000 [00:24<00:20, 8851.36it/s] 54%|█████▍    | 217679/400000 [00:24<00:20, 8812.73it/s] 55%|█████▍    | 218562/400000 [00:24<00:20, 8814.95it/s] 55%|█████▍    | 219464/400000 [00:24<00:20, 8874.57it/s] 55%|█████▌    | 220352/400000 [00:25<00:20, 8839.75it/s] 55%|█████▌    | 221241/400000 [00:25<00:20, 8851.77it/s] 56%|█████▌    | 222127/400000 [00:25<00:20, 8816.40it/s] 56%|█████▌    | 223009/400000 [00:25<00:20, 8803.49it/s] 56%|█████▌    | 223892/400000 [00:25<00:19, 8809.73it/s] 56%|█████▌    | 224774/400000 [00:25<00:20, 8723.86it/s] 56%|█████▋    | 225665/400000 [00:25<00:19, 8778.44it/s] 57%|█████▋    | 226546/400000 [00:25<00:19, 8787.70it/s] 57%|█████▋    | 227440/400000 [00:25<00:19, 8830.34it/s] 57%|█████▋    | 228338/400000 [00:25<00:19, 8872.28it/s] 57%|█████▋    | 229226/400000 [00:26<00:19, 8874.34it/s] 58%|█████▊    | 230120/400000 [00:26<00:19, 8892.60it/s] 58%|█████▊    | 231016/400000 [00:26<00:18, 8910.22it/s] 58%|█████▊    | 231908/400000 [00:26<00:18, 8867.62it/s] 58%|█████▊    | 232807/400000 [00:26<00:18, 8902.03it/s] 58%|█████▊    | 233701/400000 [00:26<00:18, 8913.33it/s] 59%|█████▊    | 234593/400000 [00:26<00:18, 8883.51it/s] 59%|█████▉    | 235482/400000 [00:26<00:18, 8858.89it/s] 59%|█████▉    | 236368/400000 [00:26<00:18, 8795.19it/s] 59%|█████▉    | 237248/400000 [00:26<00:18, 8765.28it/s] 60%|█████▉    | 238130/400000 [00:27<00:18, 8778.96it/s] 60%|█████▉    | 239023/400000 [00:27<00:18, 8822.38it/s] 60%|█████▉    | 239908/400000 [00:27<00:18, 8828.40it/s] 60%|██████    | 240791/400000 [00:27<00:18, 8818.37it/s] 60%|██████    | 241690/400000 [00:27<00:17, 8867.24it/s] 61%|██████    | 242589/400000 [00:27<00:17, 8901.72it/s] 61%|██████    | 243489/400000 [00:27<00:17, 8930.40it/s] 61%|██████    | 244397/400000 [00:27<00:17, 8974.16it/s] 61%|██████▏   | 245295/400000 [00:27<00:17, 8911.82it/s] 62%|██████▏   | 246194/400000 [00:27<00:17, 8933.10it/s] 62%|██████▏   | 247088/400000 [00:28<00:17, 8915.10it/s] 62%|██████▏   | 247983/400000 [00:28<00:17, 8923.22it/s] 62%|██████▏   | 248879/400000 [00:28<00:16, 8931.33it/s] 62%|██████▏   | 249773/400000 [00:28<00:16, 8880.00it/s] 63%|██████▎   | 250672/400000 [00:28<00:16, 8911.08it/s] 63%|██████▎   | 251564/400000 [00:28<00:16, 8912.24it/s] 63%|██████▎   | 252461/400000 [00:28<00:16, 8927.51it/s] 63%|██████▎   | 253365/400000 [00:28<00:16, 8960.70it/s] 64%|██████▎   | 254262/400000 [00:28<00:16, 8807.22it/s] 64%|██████▍   | 255162/400000 [00:28<00:16, 8863.55it/s] 64%|██████▍   | 256057/400000 [00:29<00:16, 8888.39it/s] 64%|██████▍   | 256949/400000 [00:29<00:16, 8896.33it/s] 64%|██████▍   | 257854/400000 [00:29<00:15, 8939.82it/s] 65%|██████▍   | 258751/400000 [00:29<00:15, 8947.39it/s] 65%|██████▍   | 259646/400000 [00:29<00:15, 8825.91it/s] 65%|██████▌   | 260543/400000 [00:29<00:15, 8867.72it/s] 65%|██████▌   | 261447/400000 [00:29<00:15, 8916.83it/s] 66%|██████▌   | 262348/400000 [00:29<00:15, 8944.27it/s] 66%|██████▌   | 263243/400000 [00:29<00:15, 8914.78it/s] 66%|██████▌   | 264135/400000 [00:29<00:15, 8908.58it/s] 66%|██████▋   | 265027/400000 [00:30<00:15, 8906.05it/s] 66%|██████▋   | 265921/400000 [00:30<00:15, 8913.64it/s] 67%|██████▋   | 266813/400000 [00:30<00:14, 8906.76it/s] 67%|██████▋   | 267704/400000 [00:30<00:14, 8884.33it/s] 67%|██████▋   | 268609/400000 [00:30<00:14, 8932.43it/s] 67%|██████▋   | 269519/400000 [00:30<00:14, 8981.89it/s] 68%|██████▊   | 270418/400000 [00:30<00:14, 8984.02it/s] 68%|██████▊   | 271331/400000 [00:30<00:14, 9024.66it/s] 68%|██████▊   | 272234/400000 [00:30<00:14, 8999.07it/s] 68%|██████▊   | 273134/400000 [00:30<00:14, 8979.23it/s] 69%|██████▊   | 274032/400000 [00:31<00:14, 8950.60it/s] 69%|██████▊   | 274933/400000 [00:31<00:13, 8967.17it/s] 69%|██████▉   | 275835/400000 [00:31<00:13, 8982.73it/s] 69%|██████▉   | 276749/400000 [00:31<00:13, 9026.47it/s] 69%|██████▉   | 277652/400000 [00:31<00:13, 8962.41it/s] 70%|██████▉   | 278555/400000 [00:31<00:13, 8981.24it/s] 70%|██████▉   | 279454/400000 [00:31<00:13, 8802.19it/s] 70%|███████   | 280339/400000 [00:31<00:13, 8814.19it/s] 70%|███████   | 281238/400000 [00:31<00:13, 8864.76it/s] 71%|███████   | 282125/400000 [00:31<00:13, 8826.55it/s] 71%|███████   | 283009/400000 [00:32<00:13, 8762.42it/s] 71%|███████   | 283907/400000 [00:32<00:13, 8826.14it/s] 71%|███████   | 284807/400000 [00:32<00:12, 8875.30it/s] 71%|███████▏  | 285703/400000 [00:32<00:12, 8898.16it/s] 72%|███████▏  | 286594/400000 [00:32<00:12, 8890.56it/s] 72%|███████▏  | 287501/400000 [00:32<00:12, 8941.61it/s] 72%|███████▏  | 288396/400000 [00:32<00:12, 8930.15it/s] 72%|███████▏  | 289290/400000 [00:32<00:12, 8912.55it/s] 73%|███████▎  | 290182/400000 [00:32<00:12, 8893.78it/s] 73%|███████▎  | 291072/400000 [00:32<00:12, 8696.17it/s] 73%|███████▎  | 291960/400000 [00:33<00:12, 8749.97it/s] 73%|███████▎  | 292843/400000 [00:33<00:12, 8771.95it/s] 73%|███████▎  | 293727/400000 [00:33<00:12, 8790.76it/s] 74%|███████▎  | 294608/400000 [00:33<00:11, 8795.24it/s] 74%|███████▍  | 295488/400000 [00:33<00:12, 8690.00it/s] 74%|███████▍  | 296370/400000 [00:33<00:11, 8727.54it/s] 74%|███████▍  | 297255/400000 [00:33<00:11, 8762.43it/s] 75%|███████▍  | 298132/400000 [00:33<00:11, 8741.13it/s] 75%|███████▍  | 299019/400000 [00:33<00:11, 8776.86it/s] 75%|███████▍  | 299897/400000 [00:33<00:11, 8667.73it/s] 75%|███████▌  | 300791/400000 [00:34<00:11, 8745.32it/s] 75%|███████▌  | 301675/400000 [00:34<00:11, 8770.59it/s] 76%|███████▌  | 302561/400000 [00:34<00:11, 8795.99it/s] 76%|███████▌  | 303443/400000 [00:34<00:10, 8801.88it/s] 76%|███████▌  | 304324/400000 [00:34<00:10, 8786.81it/s] 76%|███████▋  | 305208/400000 [00:34<00:10, 8802.29it/s] 77%|███████▋  | 306100/400000 [00:34<00:10, 8836.33it/s] 77%|███████▋  | 306997/400000 [00:34<00:10, 8873.30it/s] 77%|███████▋  | 307885/400000 [00:34<00:10, 8864.29it/s] 77%|███████▋  | 308772/400000 [00:34<00:10, 8815.65it/s] 77%|███████▋  | 309666/400000 [00:35<00:10, 8852.22it/s] 78%|███████▊  | 310552/400000 [00:35<00:10, 8834.77it/s] 78%|███████▊  | 311436/400000 [00:35<00:10, 8715.94it/s] 78%|███████▊  | 312321/400000 [00:35<00:10, 8754.08it/s] 78%|███████▊  | 313197/400000 [00:35<00:09, 8745.54it/s] 79%|███████▊  | 314081/400000 [00:35<00:09, 8773.13it/s] 79%|███████▊  | 314968/400000 [00:35<00:09, 8799.36it/s] 79%|███████▉  | 315857/400000 [00:35<00:09, 8824.05it/s] 79%|███████▉  | 316743/400000 [00:35<00:09, 8832.22it/s] 79%|███████▉  | 317627/400000 [00:35<00:09, 8689.00it/s] 80%|███████▉  | 318503/400000 [00:36<00:09, 8707.87it/s] 80%|███████▉  | 319385/400000 [00:36<00:09, 8741.20it/s] 80%|████████  | 320273/400000 [00:36<00:09, 8779.51it/s] 80%|████████  | 321156/400000 [00:36<00:08, 8792.93it/s] 81%|████████  | 322036/400000 [00:36<00:08, 8741.42it/s] 81%|████████  | 322923/400000 [00:36<00:08, 8776.75it/s] 81%|████████  | 323801/400000 [00:36<00:08, 8506.17it/s] 81%|████████  | 324680/400000 [00:36<00:08, 8586.78it/s] 81%|████████▏ | 325541/400000 [00:36<00:08, 8570.68it/s] 82%|████████▏ | 326400/400000 [00:37<00:08, 8497.41it/s] 82%|████████▏ | 327251/400000 [00:37<00:08, 8440.27it/s] 82%|████████▏ | 328146/400000 [00:37<00:08, 8584.77it/s] 82%|████████▏ | 329031/400000 [00:37<00:08, 8661.05it/s] 82%|████████▏ | 329907/400000 [00:37<00:08, 8687.72it/s] 83%|████████▎ | 330781/400000 [00:37<00:07, 8702.07it/s] 83%|████████▎ | 331668/400000 [00:37<00:07, 8749.36it/s] 83%|████████▎ | 332556/400000 [00:37<00:07, 8786.57it/s] 83%|████████▎ | 333451/400000 [00:37<00:07, 8832.00it/s] 84%|████████▎ | 334343/400000 [00:37<00:07, 8855.61it/s] 84%|████████▍ | 335229/400000 [00:38<00:07, 8752.29it/s] 84%|████████▍ | 336124/400000 [00:38<00:07, 8808.52it/s] 84%|████████▍ | 337007/400000 [00:38<00:07, 8814.44it/s] 84%|████████▍ | 337893/400000 [00:38<00:07, 8826.71it/s] 85%|████████▍ | 338784/400000 [00:38<00:06, 8849.25it/s] 85%|████████▍ | 339670/400000 [00:38<00:06, 8746.33it/s] 85%|████████▌ | 340546/400000 [00:38<00:07, 8436.55it/s] 85%|████████▌ | 341421/400000 [00:38<00:06, 8526.55it/s] 86%|████████▌ | 342313/400000 [00:38<00:06, 8638.55it/s] 86%|████████▌ | 343195/400000 [00:38<00:06, 8691.79it/s] 86%|████████▌ | 344066/400000 [00:39<00:06, 8635.65it/s] 86%|████████▌ | 344950/400000 [00:39<00:06, 8693.75it/s] 86%|████████▋ | 345821/400000 [00:39<00:06, 8592.83it/s] 87%|████████▋ | 346700/400000 [00:39<00:06, 8648.63it/s] 87%|████████▋ | 347582/400000 [00:39<00:06, 8696.93it/s] 87%|████████▋ | 348453/400000 [00:39<00:05, 8597.35it/s] 87%|████████▋ | 349338/400000 [00:39<00:05, 8668.81it/s] 88%|████████▊ | 350221/400000 [00:39<00:05, 8714.65it/s] 88%|████████▊ | 351120/400000 [00:39<00:05, 8792.79it/s] 88%|████████▊ | 352010/400000 [00:39<00:05, 8823.58it/s] 88%|████████▊ | 352893/400000 [00:40<00:05, 8761.65it/s] 88%|████████▊ | 353779/400000 [00:40<00:05, 8788.91it/s] 89%|████████▊ | 354659/400000 [00:40<00:05, 8787.03it/s] 89%|████████▉ | 355538/400000 [00:40<00:05, 8692.19it/s] 89%|████████▉ | 356418/400000 [00:40<00:04, 8723.26it/s] 89%|████████▉ | 357291/400000 [00:40<00:04, 8708.42it/s] 90%|████████▉ | 358177/400000 [00:40<00:04, 8751.20it/s] 90%|████████▉ | 359063/400000 [00:40<00:04, 8780.98it/s] 90%|████████▉ | 359947/400000 [00:40<00:04, 8797.68it/s] 90%|█████████ | 360827/400000 [00:40<00:04, 8770.26it/s] 90%|█████████ | 361705/400000 [00:41<00:04, 8771.64it/s] 91%|█████████ | 362583/400000 [00:41<00:04, 8482.20it/s] 91%|█████████ | 363469/400000 [00:41<00:04, 8590.18it/s] 91%|█████████ | 364361/400000 [00:41<00:04, 8683.77it/s] 91%|█████████▏| 365231/400000 [00:41<00:04, 8674.65it/s] 92%|█████████▏| 366100/400000 [00:41<00:03, 8651.47it/s] 92%|█████████▏| 366989/400000 [00:41<00:03, 8719.72it/s] 92%|█████████▏| 367875/400000 [00:41<00:03, 8760.62it/s] 92%|█████████▏| 368752/400000 [00:41<00:03, 8671.54it/s] 92%|█████████▏| 369620/400000 [00:41<00:03, 8602.07it/s] 93%|█████████▎| 370481/400000 [00:42<00:03, 8373.28it/s] 93%|█████████▎| 371359/400000 [00:42<00:03, 8490.69it/s] 93%|█████████▎| 372243/400000 [00:42<00:03, 8590.21it/s] 93%|█████████▎| 373130/400000 [00:42<00:03, 8671.09it/s] 94%|█████████▎| 374017/400000 [00:42<00:02, 8729.07it/s] 94%|█████████▎| 374891/400000 [00:42<00:02, 8655.94it/s] 94%|█████████▍| 375777/400000 [00:42<00:02, 8715.85it/s] 94%|█████████▍| 376658/400000 [00:42<00:02, 8742.26it/s] 94%|█████████▍| 377533/400000 [00:42<00:02, 8712.79it/s] 95%|█████████▍| 378413/400000 [00:42<00:02, 8736.68it/s] 95%|█████████▍| 379290/400000 [00:43<00:02, 8744.86it/s] 95%|█████████▌| 380175/400000 [00:43<00:02, 8773.83it/s] 95%|█████████▌| 381060/400000 [00:43<00:02, 8796.30it/s] 95%|█████████▌| 381946/400000 [00:43<00:02, 8814.21it/s] 96%|█████████▌| 382833/400000 [00:43<00:01, 8830.30it/s] 96%|█████████▌| 383717/400000 [00:43<00:01, 8812.17it/s] 96%|█████████▌| 384606/400000 [00:43<00:01, 8832.39it/s] 96%|█████████▋| 385497/400000 [00:43<00:01, 8853.44it/s] 97%|█████████▋| 386383/400000 [00:43<00:01, 8843.13it/s] 97%|█████████▋| 387268/400000 [00:43<00:01, 8780.70it/s] 97%|█████████▋| 388147/400000 [00:44<00:01, 8705.89it/s] 97%|█████████▋| 389034/400000 [00:44<00:01, 8752.03it/s] 97%|█████████▋| 389916/400000 [00:44<00:01, 8769.71it/s] 98%|█████████▊| 390798/400000 [00:44<00:01, 8784.68it/s] 98%|█████████▊| 391682/400000 [00:44<00:00, 8800.90it/s] 98%|█████████▊| 392563/400000 [00:44<00:00, 8739.13it/s] 98%|█████████▊| 393438/400000 [00:44<00:00, 8734.50it/s] 99%|█████████▊| 394334/400000 [00:44<00:00, 8799.30it/s] 99%|█████████▉| 395221/400000 [00:44<00:00, 8818.25it/s] 99%|█████████▉| 396110/400000 [00:44<00:00, 8839.56it/s] 99%|█████████▉| 396995/400000 [00:45<00:00, 8790.03it/s] 99%|█████████▉| 397875/400000 [00:45<00:00, 8786.89it/s]100%|█████████▉| 398754/400000 [00:45<00:00, 8770.77it/s]100%|█████████▉| 399632/400000 [00:45<00:00, 8752.07it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8802.18it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f879715b4e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011253307305692092 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011085802695424262 	 Accuracy: 52

  model saves at 52% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16072 out of table with 16070 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


### Running {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ##### 

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
RuntimeError: index out of range: Tried to access index 16072 out of table with 16070 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

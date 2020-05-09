
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f3180099fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 12:13:27.061838
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 12:13:27.066466
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 12:13:27.070053
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 12:13:27.073554
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f31798899e8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353192.3438
Epoch 2/10

1/1 [==============================] - 0s 112ms/step - loss: 259226.0000
Epoch 3/10

1/1 [==============================] - 0s 112ms/step - loss: 174656.6875
Epoch 4/10

1/1 [==============================] - 0s 115ms/step - loss: 106125.4141
Epoch 5/10

1/1 [==============================] - 0s 106ms/step - loss: 63034.9922
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 39067.3594
Epoch 7/10

1/1 [==============================] - 0s 105ms/step - loss: 25625.7930
Epoch 8/10

1/1 [==============================] - 0s 103ms/step - loss: 17637.8438
Epoch 9/10

1/1 [==============================] - 0s 110ms/step - loss: 12672.6094
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 9478.7988

  #### Inference Need return ypred, ytrue ######################### 
[[-1.8531046  -1.4370944  -0.9206581  -1.1087267  -0.21703112  0.55551136
   0.3659628   0.91159546  0.6124216   0.09577125  0.4105483  -0.04333341
  -0.39668968  1.3354545   0.29680097 -1.2131083  -1.0438926   0.44942945
  -0.00969738 -1.0581127  -1.8393936  -0.35383713 -1.223822    0.97890615
   0.01135474 -0.38541573  0.42487743  0.52140474  1.4733938   0.10357949
  -1.3312469   0.24711305 -0.16467318 -1.5178058  -0.47159967  0.64275753
   0.7253851   1.7473288  -1.132575   -0.7298194   0.28782877  1.4096444
   0.05384701 -1.9216039   1.2422396   1.0649856   0.21954674  0.07540411
  -0.73326373  0.20485777  0.86718047 -0.45493317 -0.6095866   0.54876554
   1.5934546   0.40533608 -1.3063598  -0.63008654  1.1819165   1.0035477
   0.32116684 -0.19026345  0.18529746  0.09486002 -1.756198    1.2058727
   0.49549246  0.68812025 -0.13843109  1.2489057   0.8648341  -0.43696946
  -1.4606624  -1.1115743   0.5174656   1.492483    0.0316008   0.02581301
  -1.695387   -0.7455076  -0.2485103   0.6744754   2.3820524   0.26261246
  -0.41863137 -1.3096894   0.62558234 -1.0783131   0.5916628   0.09577626
  -0.7123273  -0.42583236  0.52755195 -0.19518897  0.17993712 -0.8912253
   0.63352597  0.5821248   0.81443954 -0.1670717  -0.8113853  -0.45742106
  -0.3341236  -1.7345314   0.48834735 -1.5357432  -0.19066793 -0.8966918
  -0.5150505  -0.92047626  0.34909847  0.11347765  0.48733318 -0.6965227
   0.26155412 -0.9610375  -1.4560869   0.1454866   1.5836663  -0.48771673
  -0.01065966  6.0936937   6.3517084   5.942658    6.5202284   5.7542505
   5.89193     6.655136    6.752237    6.2822247   6.2178044   6.0249863
   7.9174438   6.959637    6.6640673   7.162338    6.4650903   5.572049
   6.0523915   5.295692    6.674918    7.584832    6.916736    6.1856427
   6.013511    5.9088173   5.313096    7.925372    6.6283765   4.7052627
   6.2640333   6.4668617   7.8714786   7.0477505   4.9949737   7.765789
   7.4790454   7.166866    7.2818055   7.3250647   6.5437236   6.801454
   4.661535    7.3006716   7.308539    6.3677206   6.2994585   6.574698
   5.832842    6.3210697   6.8385906   6.2450933   5.8327208   7.6410236
   6.230664    7.443873    4.676959    6.1287584   7.841016    5.946128
   1.3398136   0.21895874  2.4316773   1.9639028   0.6911878   0.58913195
   0.13806236  1.5207791   1.9522445   0.28642428  0.8879336   1.107789
   0.9935349   1.4266798   0.932369    0.8881076   1.2353965   0.77774847
   0.3376745   3.0732865   1.6362422   0.25297785  0.69290936  1.119818
   1.6394767   0.30792964  0.7640536   2.1209536   0.3361498   0.7301819
   0.29034662  0.94998467  0.70329905  1.2537515   0.18027484  1.4109668
   1.2202206   0.5871293   1.3751029   1.0827765   0.48921907  0.84732366
   0.32728404  1.213527    2.4190512   0.47124743  0.7674157   0.26863885
   0.22910798  0.25452924  0.24180901  0.95019424  0.37796015  0.89146894
   0.5662748   0.70317805  0.19306922  1.3099569   1.0751628   0.37724465
   1.6469522   0.5864829   1.162802    1.7048066   1.0240529   0.37635398
   0.9577768   0.65627885  2.13732     1.2376493   1.1935635   0.6063578
   1.201114    1.3777652   0.27868414  1.1258237   1.924764    1.2523372
   0.65388036  0.40024626  1.4828806   1.8598529   1.3560389   1.0233934
   0.10931295  0.6831957   2.4175687   0.7476701   0.66977024  2.2567663
   0.5530792   0.3165604   0.36176628  0.8318893   1.848072    0.9046323
   0.40020907  0.36301517  2.7196054   0.665905    1.4463999   0.9268556
   1.0022798   1.5670905   1.255602    2.5325398   0.1054914   0.28992397
   1.9964731   1.5550531   0.99838984  0.9141239   1.3800564   0.28281415
   0.58968174  0.7145946   0.37621254  0.95364565  1.6343905   0.9432422
   0.09332687  6.3533177   8.101887    6.100329    7.373723    7.8222895
   7.8582277   7.953664    7.181265    7.772881    7.7955947   6.527249
   8.181255    6.724082    7.7672763   6.460908    7.3643546   6.973142
   7.3233542   8.307488    7.2555113   7.729001    7.479932    6.1260266
   6.900146    7.717654    7.152161    5.070764    6.6540174   5.6570253
   6.763933    6.8728795   7.230667    6.6937613   5.137071    7.012776
   6.185396    7.7969294   6.511374    5.712056    6.7663937   6.141035
   7.2115335   7.540048    7.6299696   8.373117    6.5724726   7.4527187
   6.755161    6.065315    7.02559     7.8724594   6.2764254   7.4362826
   6.5738792   6.722166    7.277577    7.616585    8.076005    6.524202
  -8.66992    -2.6383598   9.61309   ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 12:13:36.143593
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.7588
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 12:13:36.148060
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9192.52
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 12:13:36.151676
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.6985
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 12:13:36.155542
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -822.255
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139850219171912
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139847706461016
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139847706461520
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139847706462024
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139847706462528
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139847706463032

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f3187ce0eb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.665851
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.628566
grad_step = 000002, loss = 0.601493
grad_step = 000003, loss = 0.573411
grad_step = 000004, loss = 0.541839
grad_step = 000005, loss = 0.512406
grad_step = 000006, loss = 0.492897
grad_step = 000007, loss = 0.479645
grad_step = 000008, loss = 0.461267
grad_step = 000009, loss = 0.439412
grad_step = 000010, loss = 0.422960
grad_step = 000011, loss = 0.412750
grad_step = 000012, loss = 0.401969
grad_step = 000013, loss = 0.388487
grad_step = 000014, loss = 0.372088
grad_step = 000015, loss = 0.355705
grad_step = 000016, loss = 0.339488
grad_step = 000017, loss = 0.321883
grad_step = 000018, loss = 0.302283
grad_step = 000019, loss = 0.284254
grad_step = 000020, loss = 0.273234
grad_step = 000021, loss = 0.263953
grad_step = 000022, loss = 0.250210
grad_step = 000023, loss = 0.236451
grad_step = 000024, loss = 0.226399
grad_step = 000025, loss = 0.218044
grad_step = 000026, loss = 0.208536
grad_step = 000027, loss = 0.197487
grad_step = 000028, loss = 0.186469
grad_step = 000029, loss = 0.176343
grad_step = 000030, loss = 0.167577
grad_step = 000031, loss = 0.158894
grad_step = 000032, loss = 0.149599
grad_step = 000033, loss = 0.140740
grad_step = 000034, loss = 0.132637
grad_step = 000035, loss = 0.125023
grad_step = 000036, loss = 0.117512
grad_step = 000037, loss = 0.109896
grad_step = 000038, loss = 0.102430
grad_step = 000039, loss = 0.095500
grad_step = 000040, loss = 0.089558
grad_step = 000041, loss = 0.084209
grad_step = 000042, loss = 0.078578
grad_step = 000043, loss = 0.072595
grad_step = 000044, loss = 0.067227
grad_step = 000045, loss = 0.062633
grad_step = 000046, loss = 0.058200
grad_step = 000047, loss = 0.053730
grad_step = 000048, loss = 0.049527
grad_step = 000049, loss = 0.045707
grad_step = 000050, loss = 0.042040
grad_step = 000051, loss = 0.038656
grad_step = 000052, loss = 0.035551
grad_step = 000053, loss = 0.032672
grad_step = 000054, loss = 0.030045
grad_step = 000055, loss = 0.027480
grad_step = 000056, loss = 0.024937
grad_step = 000057, loss = 0.022675
grad_step = 000058, loss = 0.020756
grad_step = 000059, loss = 0.018913
grad_step = 000060, loss = 0.017109
grad_step = 000061, loss = 0.015469
grad_step = 000062, loss = 0.014004
grad_step = 000063, loss = 0.012658
grad_step = 000064, loss = 0.011391
grad_step = 000065, loss = 0.010270
grad_step = 000066, loss = 0.009334
grad_step = 000067, loss = 0.008450
grad_step = 000068, loss = 0.007590
grad_step = 000069, loss = 0.006872
grad_step = 000070, loss = 0.006238
grad_step = 000071, loss = 0.005650
grad_step = 000072, loss = 0.005136
grad_step = 000073, loss = 0.004714
grad_step = 000074, loss = 0.004367
grad_step = 000075, loss = 0.004031
grad_step = 000076, loss = 0.003727
grad_step = 000077, loss = 0.003501
grad_step = 000078, loss = 0.003304
grad_step = 000079, loss = 0.003128
grad_step = 000080, loss = 0.002989
grad_step = 000081, loss = 0.002876
grad_step = 000082, loss = 0.002775
grad_step = 000083, loss = 0.002683
grad_step = 000084, loss = 0.002622
grad_step = 000085, loss = 0.002578
grad_step = 000086, loss = 0.002533
grad_step = 000087, loss = 0.002499
grad_step = 000088, loss = 0.002473
grad_step = 000089, loss = 0.002450
grad_step = 000090, loss = 0.002431
grad_step = 000091, loss = 0.002416
grad_step = 000092, loss = 0.002406
grad_step = 000093, loss = 0.002390
grad_step = 000094, loss = 0.002379
grad_step = 000095, loss = 0.002372
grad_step = 000096, loss = 0.002363
grad_step = 000097, loss = 0.002358
grad_step = 000098, loss = 0.002360
grad_step = 000099, loss = 0.002374
grad_step = 000100, loss = 0.002406
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002450
grad_step = 000102, loss = 0.002499
grad_step = 000103, loss = 0.002438
grad_step = 000104, loss = 0.002330
grad_step = 000105, loss = 0.002264
grad_step = 000106, loss = 0.002301
grad_step = 000107, loss = 0.002353
grad_step = 000108, loss = 0.002309
grad_step = 000109, loss = 0.002237
grad_step = 000110, loss = 0.002227
grad_step = 000111, loss = 0.002264
grad_step = 000112, loss = 0.002264
grad_step = 000113, loss = 0.002212
grad_step = 000114, loss = 0.002189
grad_step = 000115, loss = 0.002212
grad_step = 000116, loss = 0.002219
grad_step = 000117, loss = 0.002191
grad_step = 000118, loss = 0.002166
grad_step = 000119, loss = 0.002173
grad_step = 000120, loss = 0.002186
grad_step = 000121, loss = 0.002174
grad_step = 000122, loss = 0.002152
grad_step = 000123, loss = 0.002147
grad_step = 000124, loss = 0.002156
grad_step = 000125, loss = 0.002157
grad_step = 000126, loss = 0.002143
grad_step = 000127, loss = 0.002131
grad_step = 000128, loss = 0.002132
grad_step = 000129, loss = 0.002136
grad_step = 000130, loss = 0.002134
grad_step = 000131, loss = 0.002124
grad_step = 000132, loss = 0.002116
grad_step = 000133, loss = 0.002115
grad_step = 000134, loss = 0.002117
grad_step = 000135, loss = 0.002116
grad_step = 000136, loss = 0.002110
grad_step = 000137, loss = 0.002103
grad_step = 000138, loss = 0.002099
grad_step = 000139, loss = 0.002098
grad_step = 000140, loss = 0.002097
grad_step = 000141, loss = 0.002096
grad_step = 000142, loss = 0.002092
grad_step = 000143, loss = 0.002087
grad_step = 000144, loss = 0.002082
grad_step = 000145, loss = 0.002079
grad_step = 000146, loss = 0.002077
grad_step = 000147, loss = 0.002075
grad_step = 000148, loss = 0.002074
grad_step = 000149, loss = 0.002072
grad_step = 000150, loss = 0.002069
grad_step = 000151, loss = 0.002067
grad_step = 000152, loss = 0.002064
grad_step = 000153, loss = 0.002061
grad_step = 000154, loss = 0.002058
grad_step = 000155, loss = 0.002055
grad_step = 000156, loss = 0.002053
grad_step = 000157, loss = 0.002051
grad_step = 000158, loss = 0.002050
grad_step = 000159, loss = 0.002050
grad_step = 000160, loss = 0.002052
grad_step = 000161, loss = 0.002059
grad_step = 000162, loss = 0.002073
grad_step = 000163, loss = 0.002103
grad_step = 000164, loss = 0.002148
grad_step = 000165, loss = 0.002222
grad_step = 000166, loss = 0.002266
grad_step = 000167, loss = 0.002276
grad_step = 000168, loss = 0.002160
grad_step = 000169, loss = 0.002047
grad_step = 000170, loss = 0.002021
grad_step = 000171, loss = 0.002084
grad_step = 000172, loss = 0.002145
grad_step = 000173, loss = 0.002109
grad_step = 000174, loss = 0.002035
grad_step = 000175, loss = 0.002007
grad_step = 000176, loss = 0.002046
grad_step = 000177, loss = 0.002084
grad_step = 000178, loss = 0.002061
grad_step = 000179, loss = 0.002014
grad_step = 000180, loss = 0.001998
grad_step = 000181, loss = 0.002022
grad_step = 000182, loss = 0.002044
grad_step = 000183, loss = 0.002029
grad_step = 000184, loss = 0.001999
grad_step = 000185, loss = 0.001988
grad_step = 000186, loss = 0.002001
grad_step = 000187, loss = 0.002015
grad_step = 000188, loss = 0.002008
grad_step = 000189, loss = 0.001990
grad_step = 000190, loss = 0.001979
grad_step = 000191, loss = 0.001984
grad_step = 000192, loss = 0.001993
grad_step = 000193, loss = 0.001993
grad_step = 000194, loss = 0.001983
grad_step = 000195, loss = 0.001973
grad_step = 000196, loss = 0.001970
grad_step = 000197, loss = 0.001973
grad_step = 000198, loss = 0.001977
grad_step = 000199, loss = 0.001976
grad_step = 000200, loss = 0.001970
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001964
grad_step = 000202, loss = 0.001960
grad_step = 000203, loss = 0.001960
grad_step = 000204, loss = 0.001962
grad_step = 000205, loss = 0.001963
grad_step = 000206, loss = 0.001961
grad_step = 000207, loss = 0.001958
grad_step = 000208, loss = 0.001954
grad_step = 000209, loss = 0.001950
grad_step = 000210, loss = 0.001949
grad_step = 000211, loss = 0.001948
grad_step = 000212, loss = 0.001948
grad_step = 000213, loss = 0.001948
grad_step = 000214, loss = 0.001947
grad_step = 000215, loss = 0.001947
grad_step = 000216, loss = 0.001946
grad_step = 000217, loss = 0.001944
grad_step = 000218, loss = 0.001943
grad_step = 000219, loss = 0.001942
grad_step = 000220, loss = 0.001940
grad_step = 000221, loss = 0.001939
grad_step = 000222, loss = 0.001939
grad_step = 000223, loss = 0.001939
grad_step = 000224, loss = 0.001939
grad_step = 000225, loss = 0.001942
grad_step = 000226, loss = 0.001947
grad_step = 000227, loss = 0.001956
grad_step = 000228, loss = 0.001972
grad_step = 000229, loss = 0.002003
grad_step = 000230, loss = 0.002047
grad_step = 000231, loss = 0.002122
grad_step = 000232, loss = 0.002189
grad_step = 000233, loss = 0.002260
grad_step = 000234, loss = 0.002205
grad_step = 000235, loss = 0.002091
grad_step = 000236, loss = 0.001955
grad_step = 000237, loss = 0.001918
grad_step = 000238, loss = 0.001982
grad_step = 000239, loss = 0.002053
grad_step = 000240, loss = 0.002062
grad_step = 000241, loss = 0.001982
grad_step = 000242, loss = 0.001916
grad_step = 000243, loss = 0.001923
grad_step = 000244, loss = 0.001973
grad_step = 000245, loss = 0.001999
grad_step = 000246, loss = 0.001964
grad_step = 000247, loss = 0.001916
grad_step = 000248, loss = 0.001907
grad_step = 000249, loss = 0.001934
grad_step = 000250, loss = 0.001957
grad_step = 000251, loss = 0.001943
grad_step = 000252, loss = 0.001913
grad_step = 000253, loss = 0.001899
grad_step = 000254, loss = 0.001911
grad_step = 000255, loss = 0.001928
grad_step = 000256, loss = 0.001926
grad_step = 000257, loss = 0.001909
grad_step = 000258, loss = 0.001895
grad_step = 000259, loss = 0.001895
grad_step = 000260, loss = 0.001905
grad_step = 000261, loss = 0.001910
grad_step = 000262, loss = 0.001905
grad_step = 000263, loss = 0.001894
grad_step = 000264, loss = 0.001887
grad_step = 000265, loss = 0.001888
grad_step = 000266, loss = 0.001893
grad_step = 000267, loss = 0.001896
grad_step = 000268, loss = 0.001893
grad_step = 000269, loss = 0.001886
grad_step = 000270, loss = 0.001881
grad_step = 000271, loss = 0.001880
grad_step = 000272, loss = 0.001882
grad_step = 000273, loss = 0.001883
grad_step = 000274, loss = 0.001883
grad_step = 000275, loss = 0.001881
grad_step = 000276, loss = 0.001877
grad_step = 000277, loss = 0.001874
grad_step = 000278, loss = 0.001872
grad_step = 000279, loss = 0.001872
grad_step = 000280, loss = 0.001872
grad_step = 000281, loss = 0.001872
grad_step = 000282, loss = 0.001872
grad_step = 000283, loss = 0.001871
grad_step = 000284, loss = 0.001869
grad_step = 000285, loss = 0.001867
grad_step = 000286, loss = 0.001865
grad_step = 000287, loss = 0.001864
grad_step = 000288, loss = 0.001862
grad_step = 000289, loss = 0.001861
grad_step = 000290, loss = 0.001860
grad_step = 000291, loss = 0.001859
grad_step = 000292, loss = 0.001858
grad_step = 000293, loss = 0.001857
grad_step = 000294, loss = 0.001857
grad_step = 000295, loss = 0.001856
grad_step = 000296, loss = 0.001856
grad_step = 000297, loss = 0.001856
grad_step = 000298, loss = 0.001857
grad_step = 000299, loss = 0.001860
grad_step = 000300, loss = 0.001865
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001874
grad_step = 000302, loss = 0.001893
grad_step = 000303, loss = 0.001924
grad_step = 000304, loss = 0.001985
grad_step = 000305, loss = 0.002076
grad_step = 000306, loss = 0.002231
grad_step = 000307, loss = 0.002357
grad_step = 000308, loss = 0.002453
grad_step = 000309, loss = 0.002276
grad_step = 000310, loss = 0.002009
grad_step = 000311, loss = 0.001843
grad_step = 000312, loss = 0.001918
grad_step = 000313, loss = 0.002088
grad_step = 000314, loss = 0.002103
grad_step = 000315, loss = 0.001962
grad_step = 000316, loss = 0.001840
grad_step = 000317, loss = 0.001890
grad_step = 000318, loss = 0.001999
grad_step = 000319, loss = 0.001986
grad_step = 000320, loss = 0.001882
grad_step = 000321, loss = 0.001834
grad_step = 000322, loss = 0.001893
grad_step = 000323, loss = 0.001945
grad_step = 000324, loss = 0.001902
grad_step = 000325, loss = 0.001838
grad_step = 000326, loss = 0.001840
grad_step = 000327, loss = 0.001885
grad_step = 000328, loss = 0.001893
grad_step = 000329, loss = 0.001851
grad_step = 000330, loss = 0.001824
grad_step = 000331, loss = 0.001844
grad_step = 000332, loss = 0.001866
grad_step = 000333, loss = 0.001855
grad_step = 000334, loss = 0.001827
grad_step = 000335, loss = 0.001822
grad_step = 000336, loss = 0.001838
grad_step = 000337, loss = 0.001846
grad_step = 000338, loss = 0.001833
grad_step = 000339, loss = 0.001817
grad_step = 000340, loss = 0.001818
grad_step = 000341, loss = 0.001829
grad_step = 000342, loss = 0.001830
grad_step = 000343, loss = 0.001821
grad_step = 000344, loss = 0.001812
grad_step = 000345, loss = 0.001812
grad_step = 000346, loss = 0.001818
grad_step = 000347, loss = 0.001820
grad_step = 000348, loss = 0.001814
grad_step = 000349, loss = 0.001807
grad_step = 000350, loss = 0.001806
grad_step = 000351, loss = 0.001809
grad_step = 000352, loss = 0.001811
grad_step = 000353, loss = 0.001808
grad_step = 000354, loss = 0.001804
grad_step = 000355, loss = 0.001801
grad_step = 000356, loss = 0.001801
grad_step = 000357, loss = 0.001803
grad_step = 000358, loss = 0.001803
grad_step = 000359, loss = 0.001800
grad_step = 000360, loss = 0.001797
grad_step = 000361, loss = 0.001796
grad_step = 000362, loss = 0.001795
grad_step = 000363, loss = 0.001796
grad_step = 000364, loss = 0.001796
grad_step = 000365, loss = 0.001794
grad_step = 000366, loss = 0.001792
grad_step = 000367, loss = 0.001791
grad_step = 000368, loss = 0.001790
grad_step = 000369, loss = 0.001790
grad_step = 000370, loss = 0.001791
grad_step = 000371, loss = 0.001791
grad_step = 000372, loss = 0.001793
grad_step = 000373, loss = 0.001796
grad_step = 000374, loss = 0.001802
grad_step = 000375, loss = 0.001813
grad_step = 000376, loss = 0.001828
grad_step = 000377, loss = 0.001850
grad_step = 000378, loss = 0.001866
grad_step = 000379, loss = 0.001881
grad_step = 000380, loss = 0.001882
grad_step = 000381, loss = 0.001881
grad_step = 000382, loss = 0.001870
grad_step = 000383, loss = 0.001854
grad_step = 000384, loss = 0.001826
grad_step = 000385, loss = 0.001799
grad_step = 000386, loss = 0.001779
grad_step = 000387, loss = 0.001775
grad_step = 000388, loss = 0.001783
grad_step = 000389, loss = 0.001795
grad_step = 000390, loss = 0.001804
grad_step = 000391, loss = 0.001809
grad_step = 000392, loss = 0.001813
grad_step = 000393, loss = 0.001817
grad_step = 000394, loss = 0.001821
grad_step = 000395, loss = 0.001821
grad_step = 000396, loss = 0.001817
grad_step = 000397, loss = 0.001808
grad_step = 000398, loss = 0.001801
grad_step = 000399, loss = 0.001795
grad_step = 000400, loss = 0.001791
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001786
grad_step = 000402, loss = 0.001780
grad_step = 000403, loss = 0.001774
grad_step = 000404, loss = 0.001770
grad_step = 000405, loss = 0.001768
grad_step = 000406, loss = 0.001767
grad_step = 000407, loss = 0.001766
grad_step = 000408, loss = 0.001766
grad_step = 000409, loss = 0.001765
grad_step = 000410, loss = 0.001765
grad_step = 000411, loss = 0.001767
grad_step = 000412, loss = 0.001771
grad_step = 000413, loss = 0.001780
grad_step = 000414, loss = 0.001796
grad_step = 000415, loss = 0.001821
grad_step = 000416, loss = 0.001864
grad_step = 000417, loss = 0.001930
grad_step = 000418, loss = 0.002040
grad_step = 000419, loss = 0.002166
grad_step = 000420, loss = 0.002326
grad_step = 000421, loss = 0.002361
grad_step = 000422, loss = 0.002284
grad_step = 000423, loss = 0.002020
grad_step = 000424, loss = 0.001810
grad_step = 000425, loss = 0.001770
grad_step = 000426, loss = 0.001870
grad_step = 000427, loss = 0.001982
grad_step = 000428, loss = 0.001978
grad_step = 000429, loss = 0.001885
grad_step = 000430, loss = 0.001781
grad_step = 000431, loss = 0.001767
grad_step = 000432, loss = 0.001829
grad_step = 000433, loss = 0.001880
grad_step = 000434, loss = 0.001860
grad_step = 000435, loss = 0.001780
grad_step = 000436, loss = 0.001742
grad_step = 000437, loss = 0.001775
grad_step = 000438, loss = 0.001820
grad_step = 000439, loss = 0.001819
grad_step = 000440, loss = 0.001768
grad_step = 000441, loss = 0.001734
grad_step = 000442, loss = 0.001749
grad_step = 000443, loss = 0.001780
grad_step = 000444, loss = 0.001785
grad_step = 000445, loss = 0.001758
grad_step = 000446, loss = 0.001732
grad_step = 000447, loss = 0.001734
grad_step = 000448, loss = 0.001751
grad_step = 000449, loss = 0.001759
grad_step = 000450, loss = 0.001749
grad_step = 000451, loss = 0.001733
grad_step = 000452, loss = 0.001728
grad_step = 000453, loss = 0.001733
grad_step = 000454, loss = 0.001738
grad_step = 000455, loss = 0.001737
grad_step = 000456, loss = 0.001732
grad_step = 000457, loss = 0.001727
grad_step = 000458, loss = 0.001726
grad_step = 000459, loss = 0.001725
grad_step = 000460, loss = 0.001725
grad_step = 000461, loss = 0.001724
grad_step = 000462, loss = 0.001722
grad_step = 000463, loss = 0.001722
grad_step = 000464, loss = 0.001722
grad_step = 000465, loss = 0.001719
grad_step = 000466, loss = 0.001716
grad_step = 000467, loss = 0.001713
grad_step = 000468, loss = 0.001713
grad_step = 000469, loss = 0.001714
grad_step = 000470, loss = 0.001715
grad_step = 000471, loss = 0.001716
grad_step = 000472, loss = 0.001715
grad_step = 000473, loss = 0.001713
grad_step = 000474, loss = 0.001712
grad_step = 000475, loss = 0.001711
grad_step = 000476, loss = 0.001712
grad_step = 000477, loss = 0.001713
grad_step = 000478, loss = 0.001714
grad_step = 000479, loss = 0.001714
grad_step = 000480, loss = 0.001715
grad_step = 000481, loss = 0.001716
grad_step = 000482, loss = 0.001719
grad_step = 000483, loss = 0.001724
grad_step = 000484, loss = 0.001732
grad_step = 000485, loss = 0.001744
grad_step = 000486, loss = 0.001763
grad_step = 000487, loss = 0.001787
grad_step = 000488, loss = 0.001824
grad_step = 000489, loss = 0.001860
grad_step = 000490, loss = 0.001909
grad_step = 000491, loss = 0.001937
grad_step = 000492, loss = 0.001958
grad_step = 000493, loss = 0.001926
grad_step = 000494, loss = 0.001875
grad_step = 000495, loss = 0.001796
grad_step = 000496, loss = 0.001729
grad_step = 000497, loss = 0.001694
grad_step = 000498, loss = 0.001698
grad_step = 000499, loss = 0.001729
grad_step = 000500, loss = 0.001764
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001788
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

  date_run                              2020-05-09 12:14:00.106741
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.23276
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 12:14:00.113288
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.122033
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 12:14:00.120451
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.152722
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 12:14:00.126176
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.854329
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
0   2020-05-09 12:13:27.061838  ...    mean_absolute_error
1   2020-05-09 12:13:27.066466  ...     mean_squared_error
2   2020-05-09 12:13:27.070053  ...  median_absolute_error
3   2020-05-09 12:13:27.073554  ...               r2_score
4   2020-05-09 12:13:36.143593  ...    mean_absolute_error
5   2020-05-09 12:13:36.148060  ...     mean_squared_error
6   2020-05-09 12:13:36.151676  ...  median_absolute_error
7   2020-05-09 12:13:36.155542  ...               r2_score
8   2020-05-09 12:14:00.106741  ...    mean_absolute_error
9   2020-05-09 12:14:00.113288  ...     mean_squared_error
10  2020-05-09 12:14:00.120451  ...  median_absolute_error
11  2020-05-09 12:14:00.126176  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3710976/9912422 [00:00<00:00, 36968507.17it/s]9920512it [00:00, 34554768.79it/s]                             
0it [00:00, ?it/s]32768it [00:00, 627589.46it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462765.73it/s]1654784it [00:00, 11435828.19it/s]                         
0it [00:00, ?it/s]8192it [00:00, 187521.43it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a22a07ba8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39c0156da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a229cae80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39c0156da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a22a07ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a22a13fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a22a137b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39d53d6ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a22a137b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3a229cae80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39c0156da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd43c72d208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=dac81d8a1b529be4b37abd23316b85c17940820decb1374e39e17ec3e72b7572
  Stored in directory: /tmp/pip-ephem-wheel-cache-minc3dbr/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd43289b080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1114112/17464789 [>.............................] - ETA: 0s
 5341184/17464789 [========>.....................] - ETA: 0s
10559488/17464789 [=================>............] - ETA: 0s
16039936/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 12:15:27.140051: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 12:15:27.145108: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-09 12:15:27.145287: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5579d22e89e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 12:15:27.145305: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4520 - accuracy: 0.5140
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5286 - accuracy: 0.5090 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6104 - accuracy: 0.5037
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7395 - accuracy: 0.4952
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8353 - accuracy: 0.4890
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7995 - accuracy: 0.4913
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7389 - accuracy: 0.4953
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7109 - accuracy: 0.4971
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7372 - accuracy: 0.4954
11000/25000 [============>.................] - ETA: 4s - loss: 7.7503 - accuracy: 0.4945
12000/25000 [=============>................] - ETA: 4s - loss: 7.7548 - accuracy: 0.4942
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7480 - accuracy: 0.4947
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7488 - accuracy: 0.4946
15000/25000 [=================>............] - ETA: 3s - loss: 7.7412 - accuracy: 0.4951
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7452 - accuracy: 0.4949
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7234 - accuracy: 0.4963
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7211 - accuracy: 0.4964
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7150 - accuracy: 0.4968
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7103 - accuracy: 0.4972
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6861 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6853 - accuracy: 0.4988
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6922 - accuracy: 0.4983
25000/25000 [==============================] - 10s 390us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 12:15:44.315622
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 12:15:44.315622  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 12:15:50.673266: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 12:15:50.677703: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-09 12:15:50.677850: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56548656db90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 12:15:50.677865: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd59618fd68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4762 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.4839 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd5b261f048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7535 - accuracy: 0.4943 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6072 - accuracy: 0.5039
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5985 - accuracy: 0.5044
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6084 - accuracy: 0.5038
11000/25000 [============>.................] - ETA: 4s - loss: 7.6164 - accuracy: 0.5033
12000/25000 [=============>................] - ETA: 4s - loss: 7.6296 - accuracy: 0.5024
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6466 - accuracy: 0.5013
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
15000/25000 [=================>............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6580 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6567 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6709 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6401 - accuracy: 0.5017
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6553 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 10s 396us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd5473c92b0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:29:55, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:35:13, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:15:55, 23.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:11:40, 33.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:01:24, 47.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.07M/862M [00:01<3:29:44, 67.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:26:18, 96.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:41:52, 138kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<1:11:04, 197kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:01<49:30, 281kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:01<34:31, 400kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:02<24:05, 569kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<16:50, 809kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.4M/862M [00:02<11:48, 1.15MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<09:08, 1.48MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<08:17, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<07:37, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:04<05:43, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<06:33, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:33, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:06<04:23, 3.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<19:12, 692kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<14:47, 899kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:08<10:40, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<10:31, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<10:10, 1.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<07:48, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:10<05:36, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<1:38:01, 134kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<1:09:57, 188kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<49:13, 267kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<37:23, 350kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<27:30, 476kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:14<19:32, 668kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<14:21, 907kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<10:06:17, 21.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<7:04:43, 30.6kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:16<4:56:36, 43.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<3:33:24, 60.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<2:31:58, 85.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<1:46:55, 121kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:18<1:14:41, 173kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<13:20:08, 16.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<9:21:08, 23.0kB/s] .vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<6:32:20, 32.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<4:36:52, 46.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<3:14:47, 65.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<2:16:27, 93.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:22<1:35:25, 134kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<2:15:50, 93.9kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<1:37:39, 131kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<1:08:56, 185kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<50:07, 253kB/s]   .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<36:21, 349kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<25:43, 492kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<20:54, 603kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<17:24, 724kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:48, 984kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<09:05, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<14:24, 870kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:24, 1.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:15, 1.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<08:39, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:37, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:34, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:47, 2.59MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<08:01, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:55, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:09, 2.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:21, 2.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<03:10, 3.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<1:24:38, 145kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<1:01:44, 199kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<43:48, 280kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<30:41, 398kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<2:01:04, 101kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<1:25:57, 142kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<1:00:21, 202kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<44:57, 270kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<32:42, 371kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<23:08, 523kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<18:56, 637kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<15:44, 766kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<11:36, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<10:02, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<08:17, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:05, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:01, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:20, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:44, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:08, 2.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<11:25:26, 17.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<8:00:46, 24.7kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<5:36:06, 35.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<3:57:19, 49.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<2:48:29, 70.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<1:58:20, 99.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<1:22:44, 142kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:03:57, 183kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<45:57, 255kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<32:23, 361kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<25:20, 460kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<18:55, 615kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<13:31, 860kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<12:10, 951kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<10:52, 1.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<08:12, 1.41MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:34, 1.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:29, 1.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<04:49, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:02, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:33, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:10, 2.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:08, 2.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<8:01:16, 23.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<5:37:10, 33.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<3:55:11, 48.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<2:55:09, 64.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<2:04:53, 90.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:27:49, 129kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<1:01:26, 183kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<48:27, 232kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<35:04, 320kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<24:47, 452kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<19:54, 561kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<16:13, 689kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<11:54, 937kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<10:06, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<08:11, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<06:00, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:47, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:01, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:27, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:35, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:03, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:46, 2.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:11, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:54, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:22, 3.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:27:38, 17.3kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<7:20:11, 24.6kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<5:07:39, 35.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<3:37:09, 49.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<2:33:00, 70.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<1:47:07, 100kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<1:17:17, 138kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<56:16, 190kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<39:48, 268kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<27:54, 381kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<25:40, 414kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<19:59, 532kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<14:23, 738kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<10:15, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<10:33, 1.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<09:35, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:14, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:45, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:48, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:22, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:48, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:52, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:31, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:22, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:43, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:19, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:14, 2.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:37, 2.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:18, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:13, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:36, 2.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:13, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:10, 2.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:33, 2.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:17, 1.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:11, 2.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:33, 2.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:13, 1.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:05, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<02:59, 3.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:43, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:09, 1.93MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:12, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:33, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:22, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:39, 2.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:14, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:05, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<02:57, 3.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<10:38, 915kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<7:09:54, 22.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<5:01:41, 32.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<3:30:50, 46.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<2:29:23, 64.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<1:48:31, 89.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<1:16:54, 126kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<53:53, 179kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<39:47, 241kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<29:43, 323kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<21:15, 451kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<16:21, 583kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<13:22, 712kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<09:49, 969kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:23, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:43, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:52, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:37, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:47, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:31, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:40, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:09, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:04, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:20, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:51, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:46, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:45, 3.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:15, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:57, 1.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:18, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:11, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:25, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:14, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:26, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:53, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:52, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:09, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:44, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:45, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:04, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:36, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:40, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:59, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:36, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:36, 2.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<02:37, 3.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<08:55, 983kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<08:02, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:04, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:38, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:39, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<04:23, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:27, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:39, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:24, 1.60MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:34, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:56, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:52, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:05, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:32, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:32, 2.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:38, 3.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:23, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:43, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:43, 2.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:57, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:25, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:30, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:48, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:17, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:24, 2.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:43, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:17, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:20, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<02:25, 3.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:22, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:45, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:07, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:53, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:05, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:58, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:04, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:29, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:32, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:48, 2.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<5:49:06, 22.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<4:04:49, 32.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<2:51:01, 46.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<2:01:05, 65.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<1:27:44, 89.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<1:02:01, 127kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<43:31, 181kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<31:49, 246kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<23:43, 329kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<16:58, 460kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<13:04, 593kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<10:39, 727kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:46, 994kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<05:30, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<09:40, 794kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<08:13, 934kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:07, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:30, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:21, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:03, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:56, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<05:19, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:11, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:00, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:00, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:50, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:43, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:35, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<02:35, 2.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:44:17, 71.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:14:29, 99.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<52:20, 141kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<36:35, 201kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<28:47, 255kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<21:35, 340kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<15:24, 475kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<10:48, 673kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<12:47, 569kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<10:25, 697kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<07:38, 949kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:29, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:56, 1.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:27, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:37, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<05:19, 1.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:04, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:10, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:25, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:45, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:58, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:08, 3.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<59:12, 117kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<42:46, 162kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<30:13, 229kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<22:07, 310kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<18:16, 375kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<13:28, 508kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<09:33, 715kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<08:12, 828kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<07:04, 958kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:17, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:46, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:42, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:36, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:36, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:56, 2.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:08, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:14, 1.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:00, 1.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:49, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:43, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:55, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:04, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:11, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:29, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:42, 2.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<01:57, 3.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:08, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:36, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:13, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:58, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:01, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:07, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:11, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:30, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:45, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:55, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:15, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:34, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:02, 2.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<4:34:52, 22.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<3:12:47, 31.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<2:14:29, 45.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<1:35:04, 63.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<1:08:49, 87.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<48:42, 124kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<34:04, 176kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<25:07, 237kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<18:45, 318kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<13:20, 446kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<09:24, 630kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:38, 682kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<07:14, 813kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:21, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:40, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:24, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:21, 1.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:16, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:28, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:42, 2.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:48, 2.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:04, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<02:25, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:36, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:55, 1.93MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:18, 2.42MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:30, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:53, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:17, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:29, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:48, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:14, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:26, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:48, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:13, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:25, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:47, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:10, 2.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:34, 3.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:21, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:04, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:06, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:13, 2.35MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<09:20, 558kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<07:37, 684kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<05:34, 931kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<03:55, 1.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<12:24, 415kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<09:41, 530kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:01, 730kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<05:41, 891kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:58, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:43, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:40, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:36, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:32, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:43, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:57, 2.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:55, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:42, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:50, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:45, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:52, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:12, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:36, 2.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:13, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:11, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:28, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:28, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:41, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:08, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:31, 3.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<11:28, 407kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<08:58, 519kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<06:29, 716kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<04:32, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<33:01, 139kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<23:59, 191kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<16:58, 270kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<12:29, 363kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<09:39, 469kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<06:57, 648kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<05:32, 804kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:44, 938kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:32, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:09, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:02, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:18, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:40, 2.59MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:37, 2.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<3:03:44, 23.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<2:08:50, 33.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<1:29:43, 47.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<1:03:19, 67.2kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<46:02, 92.4kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<32:36, 130kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<22:47, 185kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<16:46, 249kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<12:33, 333kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<08:58, 465kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:52, 599kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<05:38, 729kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:08, 991kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:31, 1.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:13, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:26, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:20, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:27, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:54, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:07, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:39, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<01:11, 3.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<57:48, 66.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<41:10, 93.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<28:54, 132kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<20:35, 183kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<15:09, 249kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<10:45, 349kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<08:02, 461kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<06:21, 582kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:37, 799kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:46, 961kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:21, 1.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:31, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:19, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:19, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:47, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:48, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:56, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:31, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:37, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:49, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:24, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:02, 3.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:59, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:03, 1.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:35, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:38, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:48, 1.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:25, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:30, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:41, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:20, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:26, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:38, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:18, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:24, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:36, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:16, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:22, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:32, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:13, 2.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:19, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:32, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:11, 2.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<00:51, 3.36MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:06, 1.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:38, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<02:05, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:31, 1.86MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:41, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:33, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:10, 2.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:20, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:24, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:06, 2.46MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:12, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:20, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:02, 2.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:09, 2.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:16, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:00, 2.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:07, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:14, 2.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:57, 2.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<00:41, 3.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:15, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<1:49:03, 22.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<1:16:18, 32.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<53:06, 46.1kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<36:58, 65.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<26:41, 89.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<18:48, 127kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<13:05, 181kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<09:33, 244kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<07:05, 328kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<05:02, 459kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:50, 590kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:05, 732kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:14, 1.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:33, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:54, 561kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:53, 756kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:08, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:48, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:39, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:14, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:11, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:12, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:55, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:58, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:02, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:48, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:52, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:57, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:43, 2.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:31, 3.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:22, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:17, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:58, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:58, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:59, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:46, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:48, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:52, 1.95MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:41, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:48, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:37, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:26, 3.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:51, 846kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:35, 992kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:10, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:02, 1.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:00, 1.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:45, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:47, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:36, 2.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:41, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:32, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:34, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:38, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:30, 2.56MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:32, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:36, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:28, 2.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:30, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:34, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:26, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:18, 3.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:04, 1.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:56, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:41, 1.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:28, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:56, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:50, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:26, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:23, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:17, 2.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:12, 3.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:43, 1.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:39, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:28, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<00:19, 2.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:54, 822kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:46, 969kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:33, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:28, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:27, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:15, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<24:00, 22.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<16:37, 32.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<11:01, 46.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<07:21, 65.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<05:17, 90.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<03:41, 128kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<02:28, 181kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:40, 246kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:13, 330kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:50, 462kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:34, 594kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:27, 736kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:19, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:11, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:56, 290kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:43, 374kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:30, 517kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:18, 729kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:17, 679kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:14, 830kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.93MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.55MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 657/400000 [00:00<01:00, 6563.89it/s]  0%|          | 1465/400000 [00:00<00:57, 6953.27it/s]  1%|          | 2272/400000 [00:00<00:54, 7254.42it/s]  1%|          | 3055/400000 [00:00<00:53, 7417.78it/s]  1%|          | 3841/400000 [00:00<00:52, 7543.94it/s]  1%|          | 4600/400000 [00:00<00:52, 7556.60it/s]  1%|▏         | 5395/400000 [00:00<00:51, 7668.59it/s]  2%|▏         | 6182/400000 [00:00<00:50, 7727.05it/s]  2%|▏         | 6988/400000 [00:00<00:50, 7823.29it/s]  2%|▏         | 7784/400000 [00:01<00:49, 7862.37it/s]  2%|▏         | 8553/400000 [00:01<00:50, 7778.01it/s]  2%|▏         | 9319/400000 [00:01<00:50, 7733.90it/s]  3%|▎         | 10118/400000 [00:01<00:49, 7805.90it/s]  3%|▎         | 10920/400000 [00:01<00:49, 7867.69it/s]  3%|▎         | 11703/400000 [00:01<00:49, 7838.74it/s]  3%|▎         | 12485/400000 [00:01<00:49, 7787.55it/s]  3%|▎         | 13278/400000 [00:01<00:49, 7829.33it/s]  4%|▎         | 14104/400000 [00:01<00:48, 7952.58it/s]  4%|▎         | 14901/400000 [00:01<00:48, 7956.00it/s]  4%|▍         | 15714/400000 [00:02<00:48, 8005.61it/s]  4%|▍         | 16518/400000 [00:02<00:47, 8014.71it/s]  4%|▍         | 17320/400000 [00:02<00:48, 7918.67it/s]  5%|▍         | 18120/400000 [00:02<00:48, 7941.39it/s]  5%|▍         | 18915/400000 [00:02<00:48, 7933.43it/s]  5%|▍         | 19715/400000 [00:02<00:47, 7952.87it/s]  5%|▌         | 20517/400000 [00:02<00:47, 7971.59it/s]  5%|▌         | 21315/400000 [00:02<00:47, 7957.32it/s]  6%|▌         | 22120/400000 [00:02<00:47, 7983.55it/s]  6%|▌         | 22919/400000 [00:02<00:47, 7948.42it/s]  6%|▌         | 23714/400000 [00:03<00:47, 7840.15it/s]  6%|▌         | 24499/400000 [00:03<00:49, 7608.61it/s]  6%|▋         | 25262/400000 [00:03<00:50, 7391.24it/s]  7%|▋         | 26058/400000 [00:03<00:49, 7552.45it/s]  7%|▋         | 26868/400000 [00:03<00:48, 7707.39it/s]  7%|▋         | 27673/400000 [00:03<00:47, 7805.76it/s]  7%|▋         | 28456/400000 [00:03<00:47, 7784.87it/s]  7%|▋         | 29255/400000 [00:03<00:47, 7841.93it/s]  8%|▊         | 30041/400000 [00:03<00:47, 7828.36it/s]  8%|▊         | 30838/400000 [00:03<00:46, 7869.37it/s]  8%|▊         | 31644/400000 [00:04<00:46, 7923.35it/s]  8%|▊         | 32437/400000 [00:04<00:46, 7874.77it/s]  8%|▊         | 33226/400000 [00:04<00:46, 7877.87it/s]  9%|▊         | 34020/400000 [00:04<00:46, 7895.15it/s]  9%|▊         | 34810/400000 [00:04<00:46, 7836.97it/s]  9%|▉         | 35594/400000 [00:04<00:47, 7695.85it/s]  9%|▉         | 36365/400000 [00:04<00:47, 7694.00it/s]  9%|▉         | 37161/400000 [00:04<00:46, 7770.95it/s]  9%|▉         | 37969/400000 [00:04<00:46, 7859.15it/s] 10%|▉         | 38764/400000 [00:04<00:45, 7886.18it/s] 10%|▉         | 39555/400000 [00:05<00:45, 7891.82it/s] 10%|█         | 40345/400000 [00:05<00:45, 7854.94it/s] 10%|█         | 41135/400000 [00:05<00:45, 7865.93it/s] 10%|█         | 41935/400000 [00:05<00:45, 7903.12it/s] 11%|█         | 42741/400000 [00:05<00:44, 7947.36it/s] 11%|█         | 43536/400000 [00:05<00:45, 7912.72it/s] 11%|█         | 44328/400000 [00:05<00:45, 7881.36it/s] 11%|█▏        | 45117/400000 [00:05<00:45, 7731.78it/s] 11%|█▏        | 45891/400000 [00:05<00:47, 7520.92it/s] 12%|█▏        | 46645/400000 [00:05<00:47, 7428.75it/s] 12%|█▏        | 47390/400000 [00:06<00:47, 7368.81it/s] 12%|█▏        | 48129/400000 [00:06<00:48, 7255.90it/s] 12%|█▏        | 48924/400000 [00:06<00:47, 7449.88it/s] 12%|█▏        | 49672/400000 [00:06<00:46, 7457.06it/s] 13%|█▎        | 50420/400000 [00:06<00:47, 7291.87it/s] 13%|█▎        | 51152/400000 [00:06<00:48, 7224.59it/s] 13%|█▎        | 51876/400000 [00:06<00:48, 7110.78it/s] 13%|█▎        | 52667/400000 [00:06<00:47, 7333.03it/s] 13%|█▎        | 53466/400000 [00:06<00:46, 7517.03it/s] 14%|█▎        | 54249/400000 [00:07<00:45, 7605.98it/s] 14%|█▍        | 55017/400000 [00:07<00:45, 7625.92it/s] 14%|█▍        | 55782/400000 [00:07<00:46, 7345.49it/s] 14%|█▍        | 56520/400000 [00:07<00:47, 7303.86it/s] 14%|█▍        | 57314/400000 [00:07<00:45, 7483.54it/s] 15%|█▍        | 58066/400000 [00:07<00:46, 7404.22it/s] 15%|█▍        | 58809/400000 [00:07<00:46, 7287.60it/s] 15%|█▍        | 59540/400000 [00:07<00:47, 7123.82it/s] 15%|█▌        | 60270/400000 [00:07<00:47, 7175.12it/s] 15%|█▌        | 61036/400000 [00:07<00:46, 7312.79it/s] 15%|█▌        | 61845/400000 [00:08<00:44, 7528.81it/s] 16%|█▌        | 62636/400000 [00:08<00:44, 7637.70it/s] 16%|█▌        | 63407/400000 [00:08<00:43, 7657.69it/s] 16%|█▌        | 64193/400000 [00:08<00:43, 7715.08it/s] 16%|█▌        | 64966/400000 [00:08<00:43, 7657.39it/s] 16%|█▋        | 65759/400000 [00:08<00:43, 7735.35it/s] 17%|█▋        | 66534/400000 [00:08<00:45, 7392.52it/s] 17%|█▋        | 67278/400000 [00:08<00:46, 7186.38it/s] 17%|█▋        | 68001/400000 [00:08<00:46, 7071.36it/s] 17%|█▋        | 68795/400000 [00:08<00:45, 7311.28it/s] 17%|█▋        | 69602/400000 [00:09<00:43, 7522.56it/s] 18%|█▊        | 70406/400000 [00:09<00:42, 7668.81it/s] 18%|█▊        | 71188/400000 [00:09<00:42, 7710.67it/s] 18%|█▊        | 71998/400000 [00:09<00:41, 7820.93it/s] 18%|█▊        | 72783/400000 [00:09<00:42, 7778.38it/s] 18%|█▊        | 73563/400000 [00:09<00:42, 7750.20it/s] 19%|█▊        | 74357/400000 [00:09<00:41, 7804.89it/s] 19%|█▉        | 75139/400000 [00:09<00:41, 7757.82it/s] 19%|█▉        | 75916/400000 [00:09<00:41, 7751.68it/s] 19%|█▉        | 76702/400000 [00:09<00:41, 7781.39it/s] 19%|█▉        | 77481/400000 [00:10<00:41, 7750.06it/s] 20%|█▉        | 78257/400000 [00:10<00:42, 7620.50it/s] 20%|█▉        | 79020/400000 [00:10<00:42, 7623.17it/s] 20%|█▉        | 79812/400000 [00:10<00:41, 7708.46it/s] 20%|██        | 80605/400000 [00:10<00:41, 7771.68it/s] 20%|██        | 81394/400000 [00:10<00:40, 7805.09it/s] 21%|██        | 82175/400000 [00:10<00:40, 7795.50it/s] 21%|██        | 82955/400000 [00:10<00:40, 7792.28it/s] 21%|██        | 83769/400000 [00:10<00:40, 7892.53it/s] 21%|██        | 84594/400000 [00:10<00:39, 7995.18it/s] 21%|██▏       | 85397/400000 [00:11<00:39, 8003.85it/s] 22%|██▏       | 86198/400000 [00:11<00:39, 7882.85it/s] 22%|██▏       | 86988/400000 [00:11<00:39, 7858.64it/s] 22%|██▏       | 87799/400000 [00:11<00:39, 7932.39it/s] 22%|██▏       | 88593/400000 [00:11<00:39, 7837.19it/s] 22%|██▏       | 89378/400000 [00:11<00:40, 7716.95it/s] 23%|██▎       | 90190/400000 [00:11<00:39, 7832.46it/s] 23%|██▎       | 90975/400000 [00:11<00:40, 7710.96it/s] 23%|██▎       | 91748/400000 [00:11<00:40, 7547.86it/s] 23%|██▎       | 92505/400000 [00:12<00:41, 7440.78it/s] 23%|██▎       | 93251/400000 [00:12<00:41, 7401.70it/s] 23%|██▎       | 93993/400000 [00:12<00:42, 7278.16it/s] 24%|██▎       | 94765/400000 [00:12<00:41, 7404.47it/s] 24%|██▍       | 95572/400000 [00:12<00:40, 7591.65it/s] 24%|██▍       | 96370/400000 [00:12<00:39, 7701.69it/s] 24%|██▍       | 97167/400000 [00:12<00:38, 7779.06it/s] 24%|██▍       | 97961/400000 [00:12<00:38, 7826.24it/s] 25%|██▍       | 98750/400000 [00:12<00:38, 7844.58it/s] 25%|██▍       | 99557/400000 [00:12<00:37, 7907.45it/s] 25%|██▌       | 100363/400000 [00:13<00:37, 7951.17it/s] 25%|██▌       | 101175/400000 [00:13<00:37, 7998.82it/s] 25%|██▌       | 101976/400000 [00:13<00:37, 7987.43it/s] 26%|██▌       | 102776/400000 [00:13<00:37, 7945.75it/s] 26%|██▌       | 103571/400000 [00:13<00:37, 7921.88it/s] 26%|██▌       | 104388/400000 [00:13<00:36, 7993.29it/s] 26%|██▋       | 105188/400000 [00:13<00:37, 7767.10it/s] 26%|██▋       | 105967/400000 [00:13<00:38, 7587.59it/s] 27%|██▋       | 106728/400000 [00:13<00:39, 7377.80it/s] 27%|██▋       | 107519/400000 [00:13<00:38, 7529.29it/s] 27%|██▋       | 108344/400000 [00:14<00:37, 7731.45it/s] 27%|██▋       | 109143/400000 [00:14<00:37, 7804.23it/s] 27%|██▋       | 109939/400000 [00:14<00:36, 7848.93it/s] 28%|██▊       | 110726/400000 [00:14<00:36, 7835.49it/s] 28%|██▊       | 111531/400000 [00:14<00:36, 7898.11it/s] 28%|██▊       | 112329/400000 [00:14<00:36, 7922.30it/s] 28%|██▊       | 113122/400000 [00:14<00:36, 7883.92it/s] 28%|██▊       | 113928/400000 [00:14<00:36, 7933.30it/s] 29%|██▊       | 114722/400000 [00:14<00:36, 7862.76it/s] 29%|██▉       | 115546/400000 [00:14<00:35, 7970.11it/s] 29%|██▉       | 116357/400000 [00:15<00:35, 8011.47it/s] 29%|██▉       | 117159/400000 [00:15<00:35, 7934.95it/s] 29%|██▉       | 117965/400000 [00:15<00:35, 7970.73it/s] 30%|██▉       | 118763/400000 [00:15<00:36, 7678.60it/s] 30%|██▉       | 119534/400000 [00:15<00:36, 7584.96it/s] 30%|███       | 120309/400000 [00:15<00:36, 7631.45it/s] 30%|███       | 121114/400000 [00:15<00:35, 7751.18it/s] 30%|███       | 121919/400000 [00:15<00:35, 7836.18it/s] 31%|███       | 122704/400000 [00:15<00:35, 7830.21it/s] 31%|███       | 123504/400000 [00:15<00:35, 7880.27it/s] 31%|███       | 124314/400000 [00:16<00:34, 7944.07it/s] 31%|███▏      | 125128/400000 [00:16<00:34, 8000.06it/s] 31%|███▏      | 125933/400000 [00:16<00:34, 8013.71it/s] 32%|███▏      | 126735/400000 [00:16<00:34, 7956.42it/s] 32%|███▏      | 127531/400000 [00:16<00:34, 7908.58it/s] 32%|███▏      | 128323/400000 [00:16<00:34, 7890.28it/s] 32%|███▏      | 129117/400000 [00:16<00:34, 7904.51it/s] 32%|███▏      | 129927/400000 [00:16<00:33, 7960.06it/s] 33%|███▎      | 130724/400000 [00:16<00:33, 7923.41it/s] 33%|███▎      | 131517/400000 [00:17<00:34, 7760.08it/s] 33%|███▎      | 132294/400000 [00:17<00:35, 7586.26it/s] 33%|███▎      | 133055/400000 [00:17<00:36, 7332.23it/s] 33%|███▎      | 133842/400000 [00:17<00:35, 7484.03it/s] 34%|███▎      | 134619/400000 [00:17<00:35, 7567.15it/s] 34%|███▍      | 135422/400000 [00:17<00:34, 7700.18it/s] 34%|███▍      | 136236/400000 [00:17<00:33, 7826.64it/s] 34%|███▍      | 137058/400000 [00:17<00:33, 7939.24it/s] 34%|███▍      | 137859/400000 [00:17<00:32, 7959.20it/s] 35%|███▍      | 138657/400000 [00:17<00:33, 7784.12it/s] 35%|███▍      | 139468/400000 [00:18<00:33, 7879.02it/s] 35%|███▌      | 140263/400000 [00:18<00:32, 7898.73it/s] 35%|███▌      | 141073/400000 [00:18<00:32, 7955.50it/s] 35%|███▌      | 141875/400000 [00:18<00:32, 7973.03it/s] 36%|███▌      | 142673/400000 [00:18<00:33, 7646.76it/s] 36%|███▌      | 143472/400000 [00:18<00:33, 7744.46it/s] 36%|███▌      | 144262/400000 [00:18<00:32, 7789.33it/s] 36%|███▋      | 145070/400000 [00:18<00:32, 7874.05it/s] 36%|███▋      | 145868/400000 [00:18<00:32, 7904.08it/s] 37%|███▋      | 146660/400000 [00:18<00:32, 7832.03it/s] 37%|███▋      | 147447/400000 [00:19<00:32, 7842.77it/s] 37%|███▋      | 148254/400000 [00:19<00:31, 7906.76it/s] 37%|███▋      | 149046/400000 [00:19<00:32, 7617.25it/s] 37%|███▋      | 149811/400000 [00:19<00:34, 7330.19it/s] 38%|███▊      | 150549/400000 [00:19<00:33, 7338.83it/s] 38%|███▊      | 151360/400000 [00:19<00:32, 7552.41it/s] 38%|███▊      | 152165/400000 [00:19<00:32, 7692.87it/s] 38%|███▊      | 152956/400000 [00:19<00:31, 7755.13it/s] 38%|███▊      | 153758/400000 [00:19<00:31, 7831.65it/s] 39%|███▊      | 154543/400000 [00:19<00:31, 7830.08it/s] 39%|███▉      | 155339/400000 [00:20<00:31, 7867.62it/s] 39%|███▉      | 156139/400000 [00:20<00:30, 7906.49it/s] 39%|███▉      | 156955/400000 [00:20<00:30, 7980.54it/s] 39%|███▉      | 157754/400000 [00:20<00:30, 7968.74it/s] 40%|███▉      | 158552/400000 [00:20<00:30, 7949.66it/s] 40%|███▉      | 159348/400000 [00:20<00:30, 7934.03it/s] 40%|████      | 160142/400000 [00:20<00:30, 7918.67it/s] 40%|████      | 160950/400000 [00:20<00:30, 7964.18it/s] 40%|████      | 161747/400000 [00:20<00:29, 7960.75it/s] 41%|████      | 162544/400000 [00:20<00:30, 7855.38it/s] 41%|████      | 163335/400000 [00:21<00:30, 7870.71it/s] 41%|████      | 164123/400000 [00:21<00:30, 7838.43it/s] 41%|████      | 164908/400000 [00:21<00:30, 7835.27it/s] 41%|████▏     | 165718/400000 [00:21<00:29, 7909.78it/s] 42%|████▏     | 166510/400000 [00:21<00:29, 7798.29it/s] 42%|████▏     | 167294/400000 [00:21<00:29, 7810.15it/s] 42%|████▏     | 168083/400000 [00:21<00:29, 7830.01it/s] 42%|████▏     | 168867/400000 [00:21<00:29, 7819.31it/s] 42%|████▏     | 169655/400000 [00:21<00:29, 7837.36it/s] 43%|████▎     | 170439/400000 [00:21<00:29, 7764.47it/s] 43%|████▎     | 171224/400000 [00:22<00:29, 7789.57it/s] 43%|████▎     | 172004/400000 [00:22<00:29, 7776.81it/s] 43%|████▎     | 172782/400000 [00:22<00:29, 7713.39it/s] 43%|████▎     | 173554/400000 [00:22<00:30, 7390.18it/s] 44%|████▎     | 174327/400000 [00:22<00:30, 7486.98it/s] 44%|████▍     | 175133/400000 [00:22<00:29, 7648.18it/s] 44%|████▍     | 175933/400000 [00:22<00:28, 7749.80it/s] 44%|████▍     | 176737/400000 [00:22<00:28, 7832.76it/s] 44%|████▍     | 177522/400000 [00:22<00:28, 7815.27it/s] 45%|████▍     | 178305/400000 [00:23<00:28, 7806.62it/s] 45%|████▍     | 179097/400000 [00:23<00:28, 7839.65it/s] 45%|████▍     | 179882/400000 [00:23<00:28, 7720.35it/s] 45%|████▌     | 180675/400000 [00:23<00:28, 7780.27it/s] 45%|████▌     | 181476/400000 [00:23<00:27, 7845.96it/s] 46%|████▌     | 182262/400000 [00:23<00:27, 7840.88it/s] 46%|████▌     | 183047/400000 [00:23<00:27, 7840.73it/s] 46%|████▌     | 183855/400000 [00:23<00:27, 7909.63it/s] 46%|████▌     | 184659/400000 [00:23<00:27, 7946.91it/s] 46%|████▋     | 185460/400000 [00:23<00:26, 7964.94it/s] 47%|████▋     | 186257/400000 [00:24<00:27, 7915.90it/s] 47%|████▋     | 187058/400000 [00:24<00:26, 7943.63it/s] 47%|████▋     | 187853/400000 [00:24<00:26, 7879.00it/s] 47%|████▋     | 188642/400000 [00:24<00:27, 7659.90it/s] 47%|████▋     | 189410/400000 [00:24<00:28, 7455.09it/s] 48%|████▊     | 190158/400000 [00:24<00:28, 7345.82it/s] 48%|████▊     | 190895/400000 [00:24<00:28, 7272.94it/s] 48%|████▊     | 191624/400000 [00:24<00:29, 7166.66it/s] 48%|████▊     | 192401/400000 [00:24<00:28, 7336.76it/s] 48%|████▊     | 193195/400000 [00:24<00:27, 7505.73it/s] 48%|████▊     | 193977/400000 [00:25<00:27, 7596.27it/s] 49%|████▊     | 194791/400000 [00:25<00:26, 7751.26it/s] 49%|████▉     | 195587/400000 [00:25<00:26, 7811.63it/s] 49%|████▉     | 196399/400000 [00:25<00:25, 7901.00it/s] 49%|████▉     | 197208/400000 [00:25<00:25, 7955.11it/s] 50%|████▉     | 198005/400000 [00:25<00:25, 7927.63it/s] 50%|████▉     | 198799/400000 [00:25<00:25, 7896.61it/s] 50%|████▉     | 199602/400000 [00:25<00:25, 7933.72it/s] 50%|█████     | 200407/400000 [00:25<00:25, 7967.53it/s] 50%|█████     | 201205/400000 [00:25<00:24, 7965.92it/s] 51%|█████     | 202003/400000 [00:26<00:24, 7967.11it/s] 51%|█████     | 202800/400000 [00:26<00:24, 7958.73it/s] 51%|█████     | 203602/400000 [00:26<00:24, 7975.45it/s] 51%|█████     | 204400/400000 [00:26<00:24, 7971.19it/s] 51%|█████▏    | 205204/400000 [00:26<00:24, 7989.04it/s] 52%|█████▏    | 206003/400000 [00:26<00:24, 7786.25it/s] 52%|█████▏    | 206783/400000 [00:26<00:25, 7510.17it/s] 52%|█████▏    | 207537/400000 [00:26<00:26, 7375.48it/s] 52%|█████▏    | 208278/400000 [00:26<00:26, 7273.14it/s] 52%|█████▏    | 209090/400000 [00:26<00:25, 7506.55it/s] 52%|█████▏    | 209857/400000 [00:27<00:25, 7554.40it/s] 53%|█████▎    | 210629/400000 [00:27<00:24, 7600.95it/s] 53%|█████▎    | 211391/400000 [00:27<00:25, 7431.05it/s] 53%|█████▎    | 212178/400000 [00:27<00:24, 7556.17it/s] 53%|█████▎    | 212982/400000 [00:27<00:24, 7693.79it/s] 53%|█████▎    | 213760/400000 [00:27<00:24, 7716.35it/s] 54%|█████▎    | 214563/400000 [00:27<00:23, 7806.53it/s] 54%|█████▍    | 215384/400000 [00:27<00:23, 7920.76it/s] 54%|█████▍    | 216178/400000 [00:27<00:23, 7910.28it/s] 54%|█████▍    | 216985/400000 [00:28<00:23, 7956.88it/s] 54%|█████▍    | 217782/400000 [00:28<00:23, 7883.62it/s] 55%|█████▍    | 218571/400000 [00:28<00:23, 7669.47it/s] 55%|█████▍    | 219340/400000 [00:28<00:24, 7398.53it/s] 55%|█████▌    | 220084/400000 [00:28<00:24, 7247.13it/s] 55%|█████▌    | 220812/400000 [00:28<00:25, 7121.21it/s] 55%|█████▌    | 221527/400000 [00:28<00:25, 7067.52it/s] 56%|█████▌    | 222236/400000 [00:28<00:25, 7021.48it/s] 56%|█████▌    | 222950/400000 [00:28<00:25, 7055.05it/s] 56%|█████▌    | 223657/400000 [00:28<00:25, 6950.87it/s] 56%|█████▌    | 224423/400000 [00:29<00:24, 7148.76it/s] 56%|█████▋    | 225211/400000 [00:29<00:23, 7353.07it/s] 56%|█████▋    | 226000/400000 [00:29<00:23, 7504.41it/s] 57%|█████▋    | 226803/400000 [00:29<00:22, 7653.48it/s] 57%|█████▋    | 227595/400000 [00:29<00:22, 7731.51it/s] 57%|█████▋    | 228410/400000 [00:29<00:21, 7852.06it/s] 57%|█████▋    | 229198/400000 [00:29<00:22, 7749.90it/s] 57%|█████▋    | 229993/400000 [00:29<00:21, 7806.09it/s] 58%|█████▊    | 230789/400000 [00:29<00:21, 7851.48it/s] 58%|█████▊    | 231581/400000 [00:29<00:21, 7871.13it/s] 58%|█████▊    | 232369/400000 [00:30<00:21, 7820.62it/s] 58%|█████▊    | 233152/400000 [00:30<00:22, 7470.83it/s] 58%|█████▊    | 233903/400000 [00:30<00:22, 7333.60it/s] 59%|█████▊    | 234640/400000 [00:30<00:22, 7272.53it/s] 59%|█████▉    | 235370/400000 [00:30<00:22, 7194.22it/s] 59%|█████▉    | 236092/400000 [00:30<00:23, 7107.90it/s] 59%|█████▉    | 236820/400000 [00:30<00:22, 7155.55it/s] 59%|█████▉    | 237574/400000 [00:30<00:22, 7263.77it/s] 60%|█████▉    | 238379/400000 [00:30<00:21, 7481.48it/s] 60%|█████▉    | 239130/400000 [00:31<00:21, 7324.46it/s] 60%|█████▉    | 239865/400000 [00:31<00:21, 7302.59it/s] 60%|██████    | 240619/400000 [00:31<00:21, 7355.65it/s] 60%|██████    | 241356/400000 [00:31<00:21, 7259.64it/s] 61%|██████    | 242084/400000 [00:31<00:22, 7161.45it/s] 61%|██████    | 242802/400000 [00:31<00:22, 7138.32it/s] 61%|██████    | 243517/400000 [00:31<00:22, 7107.81it/s] 61%|██████    | 244231/400000 [00:31<00:21, 7117.21it/s] 61%|██████    | 244944/400000 [00:31<00:21, 7074.54it/s] 61%|██████▏   | 245652/400000 [00:31<00:21, 7075.03it/s] 62%|██████▏   | 246373/400000 [00:32<00:21, 7111.87it/s] 62%|██████▏   | 247085/400000 [00:32<00:21, 7059.62it/s] 62%|██████▏   | 247826/400000 [00:32<00:21, 7161.20it/s] 62%|██████▏   | 248543/400000 [00:32<00:21, 6947.77it/s] 62%|██████▏   | 249240/400000 [00:32<00:22, 6652.79it/s] 62%|██████▏   | 249971/400000 [00:32<00:21, 6835.74it/s] 63%|██████▎   | 250750/400000 [00:32<00:21, 7094.84it/s] 63%|██████▎   | 251522/400000 [00:32<00:20, 7270.29it/s] 63%|██████▎   | 252319/400000 [00:32<00:19, 7465.27it/s] 63%|██████▎   | 253090/400000 [00:32<00:19, 7535.68it/s] 63%|██████▎   | 253877/400000 [00:33<00:19, 7632.57it/s] 64%|██████▎   | 254659/400000 [00:33<00:18, 7687.70it/s] 64%|██████▍   | 255441/400000 [00:33<00:18, 7724.04it/s] 64%|██████▍   | 256230/400000 [00:33<00:18, 7772.86it/s] 64%|██████▍   | 257009/400000 [00:33<00:18, 7758.89it/s] 64%|██████▍   | 257814/400000 [00:33<00:18, 7842.13it/s] 65%|██████▍   | 258599/400000 [00:33<00:18, 7837.42it/s] 65%|██████▍   | 259394/400000 [00:33<00:17, 7868.72it/s] 65%|██████▌   | 260182/400000 [00:33<00:17, 7849.35it/s] 65%|██████▌   | 260968/400000 [00:33<00:17, 7804.01it/s] 65%|██████▌   | 261749/400000 [00:34<00:18, 7452.34it/s] 66%|██████▌   | 262498/400000 [00:34<00:18, 7417.99it/s] 66%|██████▌   | 263286/400000 [00:34<00:18, 7547.34it/s] 66%|██████▌   | 264052/400000 [00:34<00:17, 7578.80it/s] 66%|██████▌   | 264812/400000 [00:34<00:18, 7366.18it/s] 66%|██████▋   | 265552/400000 [00:34<00:18, 7224.04it/s] 67%|██████▋   | 266289/400000 [00:34<00:18, 7264.84it/s] 67%|██████▋   | 267018/400000 [00:34<00:18, 7220.59it/s] 67%|██████▋   | 267742/400000 [00:34<00:18, 7054.68it/s] 67%|██████▋   | 268450/400000 [00:35<00:19, 6911.51it/s] 67%|██████▋   | 269143/400000 [00:35<00:18, 6887.41it/s] 67%|██████▋   | 269834/400000 [00:35<00:19, 6735.83it/s] 68%|██████▊   | 270510/400000 [00:35<00:19, 6701.60it/s] 68%|██████▊   | 271304/400000 [00:35<00:18, 7028.63it/s] 68%|██████▊   | 272100/400000 [00:35<00:17, 7282.92it/s] 68%|██████▊   | 272869/400000 [00:35<00:17, 7399.83it/s] 68%|██████▊   | 273614/400000 [00:35<00:17, 7360.62it/s] 69%|██████▊   | 274383/400000 [00:35<00:16, 7454.59it/s] 69%|██████▉   | 275178/400000 [00:35<00:16, 7592.85it/s] 69%|██████▉   | 275968/400000 [00:36<00:16, 7682.12it/s] 69%|██████▉   | 276768/400000 [00:36<00:15, 7773.77it/s] 69%|██████▉   | 277548/400000 [00:36<00:15, 7731.24it/s] 70%|██████▉   | 278323/400000 [00:36<00:15, 7728.02it/s] 70%|██████▉   | 279097/400000 [00:36<00:15, 7714.88it/s] 70%|██████▉   | 279887/400000 [00:36<00:15, 7766.55it/s] 70%|███████   | 280665/400000 [00:36<00:15, 7661.20it/s] 70%|███████   | 281432/400000 [00:36<00:15, 7426.97it/s] 71%|███████   | 282177/400000 [00:36<00:16, 7330.80it/s] 71%|███████   | 282912/400000 [00:36<00:16, 7256.82it/s] 71%|███████   | 283640/400000 [00:37<00:16, 7261.74it/s] 71%|███████   | 284368/400000 [00:37<00:16, 7217.38it/s] 71%|███████▏  | 285093/400000 [00:37<00:15, 7226.07it/s] 71%|███████▏  | 285869/400000 [00:37<00:15, 7375.63it/s] 72%|███████▏  | 286633/400000 [00:37<00:15, 7452.89it/s] 72%|███████▏  | 287400/400000 [00:37<00:14, 7516.06it/s] 72%|███████▏  | 288153/400000 [00:37<00:14, 7499.43it/s] 72%|███████▏  | 288904/400000 [00:37<00:15, 7398.00it/s] 72%|███████▏  | 289645/400000 [00:37<00:15, 7177.13it/s] 73%|███████▎  | 290436/400000 [00:37<00:14, 7381.65it/s] 73%|███████▎  | 291177/400000 [00:38<00:14, 7377.06it/s] 73%|███████▎  | 291917/400000 [00:38<00:14, 7352.36it/s] 73%|███████▎  | 292654/400000 [00:38<00:14, 7202.35it/s] 73%|███████▎  | 293376/400000 [00:38<00:14, 7167.48it/s] 74%|███████▎  | 294167/400000 [00:38<00:14, 7374.53it/s] 74%|███████▎  | 294972/400000 [00:38<00:13, 7562.90it/s] 74%|███████▍  | 295732/400000 [00:38<00:14, 7376.67it/s] 74%|███████▍  | 296499/400000 [00:38<00:13, 7459.80it/s] 74%|███████▍  | 297284/400000 [00:38<00:13, 7570.57it/s] 75%|███████▍  | 298092/400000 [00:38<00:13, 7714.91it/s] 75%|███████▍  | 298890/400000 [00:39<00:12, 7789.69it/s] 75%|███████▍  | 299696/400000 [00:39<00:12, 7866.38it/s] 75%|███████▌  | 300484/400000 [00:39<00:12, 7712.93it/s] 75%|███████▌  | 301257/400000 [00:39<00:13, 7468.38it/s] 76%|███████▌  | 302056/400000 [00:39<00:12, 7617.53it/s] 76%|███████▌  | 302857/400000 [00:39<00:12, 7729.33it/s] 76%|███████▌  | 303655/400000 [00:39<00:12, 7801.43it/s] 76%|███████▌  | 304462/400000 [00:39<00:12, 7879.73it/s] 76%|███████▋  | 305252/400000 [00:39<00:12, 7861.36it/s] 77%|███████▋  | 306051/400000 [00:40<00:11, 7898.86it/s] 77%|███████▋  | 306842/400000 [00:40<00:11, 7886.04it/s] 77%|███████▋  | 307655/400000 [00:40<00:11, 7957.31it/s] 77%|███████▋  | 308457/400000 [00:40<00:11, 7973.18it/s] 77%|███████▋  | 309255/400000 [00:40<00:11, 7849.22it/s] 78%|███████▊  | 310049/400000 [00:40<00:11, 7875.34it/s] 78%|███████▊  | 310862/400000 [00:40<00:11, 7948.43it/s] 78%|███████▊  | 311658/400000 [00:40<00:11, 7700.14it/s] 78%|███████▊  | 312431/400000 [00:40<00:11, 7493.64it/s] 78%|███████▊  | 313217/400000 [00:40<00:11, 7598.88it/s] 78%|███████▊  | 313980/400000 [00:41<00:11, 7595.72it/s] 79%|███████▊  | 314742/400000 [00:41<00:11, 7331.08it/s] 79%|███████▉  | 315542/400000 [00:41<00:11, 7518.74it/s] 79%|███████▉  | 316325/400000 [00:41<00:10, 7608.47it/s] 79%|███████▉  | 317110/400000 [00:41<00:10, 7678.00it/s] 79%|███████▉  | 317886/400000 [00:41<00:10, 7701.28it/s] 80%|███████▉  | 318658/400000 [00:41<00:10, 7706.50it/s] 80%|███████▉  | 319448/400000 [00:41<00:10, 7761.74it/s] 80%|████████  | 320225/400000 [00:41<00:10, 7703.93it/s] 80%|████████  | 320999/400000 [00:41<00:10, 7713.31it/s] 80%|████████  | 321807/400000 [00:42<00:10, 7817.99it/s] 81%|████████  | 322631/400000 [00:42<00:09, 7937.92it/s] 81%|████████  | 323432/400000 [00:42<00:09, 7956.98it/s] 81%|████████  | 324229/400000 [00:42<00:09, 7858.74it/s] 81%|████████▏ | 325016/400000 [00:42<00:09, 7774.07it/s] 81%|████████▏ | 325805/400000 [00:42<00:09, 7806.58it/s] 82%|████████▏ | 326604/400000 [00:42<00:09, 7858.52it/s] 82%|████████▏ | 327391/400000 [00:42<00:09, 7828.30it/s] 82%|████████▏ | 328203/400000 [00:42<00:09, 7911.45it/s] 82%|████████▏ | 328995/400000 [00:42<00:09, 7857.27it/s] 82%|████████▏ | 329803/400000 [00:43<00:08, 7922.54it/s] 83%|████████▎ | 330617/400000 [00:43<00:08, 7984.46it/s] 83%|████████▎ | 331416/400000 [00:43<00:08, 7622.11it/s] 83%|████████▎ | 332207/400000 [00:43<00:08, 7705.32it/s] 83%|████████▎ | 332981/400000 [00:43<00:08, 7550.35it/s] 83%|████████▎ | 333739/400000 [00:43<00:08, 7428.24it/s] 84%|████████▎ | 334485/400000 [00:43<00:09, 7258.52it/s] 84%|████████▍ | 335214/400000 [00:43<00:09, 7064.04it/s] 84%|████████▍ | 335968/400000 [00:43<00:08, 7199.02it/s] 84%|████████▍ | 336691/400000 [00:44<00:08, 7067.05it/s] 84%|████████▍ | 337401/400000 [00:44<00:08, 7005.77it/s] 85%|████████▍ | 338179/400000 [00:44<00:08, 7219.16it/s] 85%|████████▍ | 338955/400000 [00:44<00:08, 7372.12it/s] 85%|████████▍ | 339705/400000 [00:44<00:08, 7409.39it/s] 85%|████████▌ | 340483/400000 [00:44<00:07, 7516.81it/s] 85%|████████▌ | 341281/400000 [00:44<00:07, 7649.49it/s] 86%|████████▌ | 342075/400000 [00:44<00:07, 7734.04it/s] 86%|████████▌ | 342864/400000 [00:44<00:07, 7778.60it/s] 86%|████████▌ | 343667/400000 [00:44<00:07, 7851.29it/s] 86%|████████▌ | 344454/400000 [00:45<00:07, 7408.70it/s] 86%|████████▋ | 345201/400000 [00:45<00:07, 7354.01it/s] 86%|████████▋ | 345973/400000 [00:45<00:07, 7457.89it/s] 87%|████████▋ | 346770/400000 [00:45<00:07, 7600.58it/s] 87%|████████▋ | 347553/400000 [00:45<00:06, 7665.27it/s] 87%|████████▋ | 348356/400000 [00:45<00:06, 7767.39it/s] 87%|████████▋ | 349165/400000 [00:45<00:06, 7859.23it/s] 87%|████████▋ | 349953/400000 [00:45<00:06, 7553.76it/s] 88%|████████▊ | 350713/400000 [00:45<00:06, 7501.03it/s] 88%|████████▊ | 351480/400000 [00:45<00:06, 7549.23it/s] 88%|████████▊ | 352267/400000 [00:46<00:06, 7638.08it/s] 88%|████████▊ | 353062/400000 [00:46<00:06, 7728.26it/s] 88%|████████▊ | 353864/400000 [00:46<00:05, 7811.74it/s] 89%|████████▊ | 354670/400000 [00:46<00:05, 7884.29it/s] 89%|████████▉ | 355460/400000 [00:46<00:05, 7771.50it/s] 89%|████████▉ | 356239/400000 [00:46<00:05, 7551.62it/s] 89%|████████▉ | 356997/400000 [00:46<00:05, 7355.88it/s] 89%|████████▉ | 357751/400000 [00:46<00:05, 7409.38it/s] 90%|████████▉ | 358494/400000 [00:46<00:05, 7291.20it/s] 90%|████████▉ | 359244/400000 [00:46<00:05, 7349.77it/s] 90%|█████████ | 360041/400000 [00:47<00:05, 7524.65it/s] 90%|█████████ | 360844/400000 [00:47<00:05, 7667.98it/s] 90%|█████████ | 361613/400000 [00:47<00:05, 7620.10it/s] 91%|█████████ | 362404/400000 [00:47<00:04, 7701.08it/s] 91%|█████████ | 363176/400000 [00:47<00:04, 7634.64it/s] 91%|█████████ | 363977/400000 [00:47<00:04, 7742.95it/s] 91%|█████████ | 364781/400000 [00:47<00:04, 7828.26it/s] 91%|█████████▏| 365576/400000 [00:47<00:04, 7862.43it/s] 92%|█████████▏| 366389/400000 [00:47<00:04, 7937.83it/s] 92%|█████████▏| 367184/400000 [00:48<00:04, 7899.09it/s] 92%|█████████▏| 367975/400000 [00:48<00:04, 7875.44it/s] 92%|█████████▏| 368768/400000 [00:48<00:03, 7889.36it/s] 92%|█████████▏| 369569/400000 [00:48<00:03, 7924.50it/s] 93%|█████████▎| 370363/400000 [00:48<00:03, 7928.24it/s] 93%|█████████▎| 371156/400000 [00:48<00:03, 7844.18it/s] 93%|█████████▎| 371958/400000 [00:48<00:03, 7892.66it/s] 93%|█████████▎| 372753/400000 [00:48<00:03, 7909.77it/s] 93%|█████████▎| 373551/400000 [00:48<00:03, 7928.70it/s] 94%|█████████▎| 374358/400000 [00:48<00:03, 7968.51it/s] 94%|█████████▍| 375156/400000 [00:49<00:03, 7925.54it/s] 94%|█████████▍| 375949/400000 [00:49<00:03, 7888.75it/s] 94%|█████████▍| 376739/400000 [00:49<00:02, 7866.88it/s] 94%|█████████▍| 377535/400000 [00:49<00:02, 7892.93it/s] 95%|█████████▍| 378325/400000 [00:49<00:02, 7849.81it/s] 95%|█████████▍| 379111/400000 [00:49<00:02, 7747.41it/s] 95%|█████████▍| 379902/400000 [00:49<00:02, 7794.95it/s] 95%|█████████▌| 380713/400000 [00:49<00:02, 7884.55it/s] 95%|█████████▌| 381512/400000 [00:49<00:02, 7913.97it/s] 96%|█████████▌| 382304/400000 [00:49<00:02, 7647.63it/s] 96%|█████████▌| 383071/400000 [00:50<00:02, 7487.55it/s] 96%|█████████▌| 383823/400000 [00:50<00:02, 7179.27it/s] 96%|█████████▌| 384546/400000 [00:50<00:02, 7159.10it/s] 96%|█████████▋| 385265/400000 [00:50<00:02, 7147.17it/s] 96%|█████████▋| 385989/400000 [00:50<00:01, 7170.83it/s] 97%|█████████▋| 386746/400000 [00:50<00:01, 7284.05it/s] 97%|█████████▋| 387528/400000 [00:50<00:01, 7435.43it/s] 97%|█████████▋| 388307/400000 [00:50<00:01, 7536.44it/s] 97%|█████████▋| 389063/400000 [00:50<00:01, 7242.44it/s] 97%|█████████▋| 389846/400000 [00:50<00:01, 7407.31it/s] 98%|█████████▊| 390637/400000 [00:51<00:01, 7550.44it/s] 98%|█████████▊| 391410/400000 [00:51<00:01, 7602.46it/s] 98%|█████████▊| 392219/400000 [00:51<00:01, 7741.68it/s] 98%|█████████▊| 392996/400000 [00:51<00:00, 7466.06it/s] 98%|█████████▊| 393747/400000 [00:51<00:00, 7176.46it/s] 99%|█████████▊| 394501/400000 [00:51<00:00, 7270.72it/s] 99%|█████████▉| 395245/400000 [00:51<00:00, 7318.41it/s] 99%|█████████▉| 395980/400000 [00:51<00:00, 7237.76it/s] 99%|█████████▉| 396706/400000 [00:51<00:00, 7233.12it/s] 99%|█████████▉| 397431/400000 [00:52<00:00, 7164.64it/s]100%|█████████▉| 398153/400000 [00:52<00:00, 7179.05it/s]100%|█████████▉| 398872/400000 [00:52<00:00, 7131.70it/s]100%|█████████▉| 399651/400000 [00:52<00:00, 7315.74it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7640.63it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd58c0d21d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011100155004274862 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010881848757880987 	 Accuracy: 71

  model saves at 71% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15838 out of table with 15834 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15838 out of table with 15834 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

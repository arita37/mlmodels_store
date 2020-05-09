
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7faf75bd84a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 13:15:16.509300
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 13:15:16.513005
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 13:15:16.516175
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 13:15:16.519411
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7faf6df28438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351173.6875
Epoch 2/10

1/1 [==============================] - 0s 123ms/step - loss: 222166.8125
Epoch 3/10

1/1 [==============================] - 0s 110ms/step - loss: 119761.2500
Epoch 4/10

1/1 [==============================] - 0s 107ms/step - loss: 62817.7539
Epoch 5/10

1/1 [==============================] - 0s 89ms/step - loss: 35595.7578
Epoch 6/10

1/1 [==============================] - 0s 86ms/step - loss: 22107.8418
Epoch 7/10

1/1 [==============================] - 0s 104ms/step - loss: 14858.4707
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 10658.1621
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 8066.5317
Epoch 10/10

1/1 [==============================] - 0s 107ms/step - loss: 6395.9219

  #### Inference Need return ypred, ytrue ######################### 
[[-0.424497    2.1317182   0.23398474 -0.18195571  1.1141825  -0.16377223
   1.1242807   0.05873728  0.07928433 -0.23796579 -0.53653103 -1.1538969
  -0.02618778 -0.06509101 -0.66154414  0.39356846 -0.83643514  0.32357383
   1.590251   -1.4164798   0.22561726 -1.3674436   0.87390214  0.09876597
   0.726413    0.6964911   0.13812454  0.48480976 -0.04530111 -0.6295235
  -1.5271742  -1.3598161  -0.08790241  1.2849205  -0.06460536 -0.01347779
  -1.0300752   0.54218423 -1.3273144  -0.65929127  0.4503823  -0.7010678
   0.2190102   0.37371635 -1.1204338   0.31993967 -0.26553115  0.4975534
   0.49461868  1.4125617  -0.46710312 -0.27624288 -0.78217536  0.33332726
  -0.08951926 -1.3202845  -0.965938   -0.85785943 -0.91262543  1.107065
  -0.11268803  7.508422    6.914527    8.055507    8.254479    6.7328343
   8.591788    8.187757    7.1377926   7.2048583   7.438396    8.944205
   7.03716     7.688295    8.425355    8.08253     8.444535    7.6775436
   9.284559    6.6587453   9.263884    8.146535    8.467499    9.387907
   9.948697    8.108941    8.409734    8.379645    6.464115    7.9662147
   9.3901205   7.280655    8.428511    6.800807    8.068748    7.2652206
   8.309189    8.38172     7.9433208   9.846738    8.686881    8.145063
   8.718311    8.734902    9.144945    8.646095    9.253902    7.3919754
   8.073059    6.834606    8.203177    6.156643    7.942217    7.891334
   8.378976    9.6634445   8.314143    7.7986274   9.543877    8.489574
  -1.1174418  -1.1373775  -0.12948631  0.4634019  -1.079721   -1.1772039
   0.797289    0.34139562  2.0435398   0.4634755   0.590563    0.7589817
  -0.4799842  -0.06342429 -1.0507939  -0.23572169 -0.35040483  0.93655705
   1.4316794   0.57862437  0.13325033  1.1095057   0.41424483 -1.542379
  -0.6327854  -0.61436534  1.0261202  -0.13494581 -0.37790295 -1.1155173
  -0.2486456   1.0955637   0.3583119  -1.0884862   0.48303258  1.0934649
   0.20179582  0.41125932 -0.8524498  -0.41187024 -0.4699182  -1.5905628
   0.17772028  1.1268847  -0.09048507 -1.5103512   0.48420292 -1.2011492
  -1.7680167  -0.29334125 -0.28167766 -0.68472767 -1.553246    1.2441982
  -0.85784996  0.84924585 -1.1665789   0.45803902 -0.72765505 -0.07771924
   0.54167646  0.5687768   1.1196053   1.0265248   1.2027715   0.18069243
   1.784975    0.69548476  0.21306646  2.066127    1.3135555   0.7086494
   0.2311148   0.324535    2.5826292   0.3145619   1.1435273   1.0044073
   0.20308745  0.4871167   0.72406256  2.716786    0.37282836  1.1496757
   0.47019702  2.13313     3.2490983   1.6326611   0.5739162   1.3160294
   2.9649837   2.4457026   1.2685564   1.568344    0.36334246  0.79069614
   1.1740513   0.38951015  1.2391162   2.5926957   0.5999857   1.9845316
   1.2685833   1.6083189   2.1499257   0.8643911   0.25921673  2.6673427
   2.728006    1.7244706   0.7909775   1.8366935   0.30004674  0.27100027
   2.3916383   0.41850203  0.14189988  1.5283501   1.4456329   1.9279549
   0.15908313  8.422585    8.1156645   8.314281    6.9587893   9.159702
   6.89763     7.9707108   8.143124   10.152181    9.902157    9.869291
   9.340578    7.460681    9.138035    9.923794    9.082549    8.0736
   8.6229      7.498623    8.389194    6.9700217   8.354138    8.850611
   7.2391186   7.436878    8.718646    8.838581    7.024658    7.4976864
   7.580689    8.236322    9.457667    7.6734824   9.639261    7.9639316
   8.775597    7.7061777   8.921095    9.219093    7.971058    8.173566
   8.851674    8.502726    9.124088    8.831628    7.32114     7.777373
   7.8276315   9.081987    8.865314    8.198078    9.028771    7.3468866
   9.268446    8.583165    8.2468      9.617458    8.654949    8.082738
   0.5875563   0.61868954  0.12826061  0.39163494  1.4877481   0.61327064
   0.23785442  1.0818714   0.62540925  0.5465594   1.737062    0.9711021
   3.017363    0.4100814   0.8044426   1.0538045   1.5110607   1.6945206
   0.6279263   0.6387788   2.2671776   0.5191393   1.5712669   0.7583332
   1.1013087   0.4642297   0.32861745  0.40814102  0.46326935  0.61329186
   3.0859447   2.3984075   0.8360764   0.68221146  0.5717208   2.2151976
   1.5174761   2.5966105   1.413312    1.6311032   1.6276432   1.3286496
   0.1179738   0.9316548   1.1567669   1.4383097   0.79012716  0.99602455
   2.1739445   0.5350131   0.09372246  0.91694903  0.5460822   1.2601717
   1.8161271   0.8855859   0.44886923  0.19260263  0.96629083  0.23652196
  -6.799655    7.9695935  -4.6095295 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 13:15:27.302259
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.327
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 13:15:27.306603
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8921.93
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 13:15:27.310188
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.4915
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 13:15:27.313660
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -798.022
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140390882459776
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140390209553408
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140390209553912
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140390209554416
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140390209554920
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140390209555424

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7faf6038bb38> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.579936
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.547504
grad_step = 000002, loss = 0.526367
grad_step = 000003, loss = 0.505562
grad_step = 000004, loss = 0.483274
grad_step = 000005, loss = 0.462318
grad_step = 000006, loss = 0.448383
grad_step = 000007, loss = 0.437531
grad_step = 000008, loss = 0.421336
grad_step = 000009, loss = 0.401238
grad_step = 000010, loss = 0.383102
grad_step = 000011, loss = 0.365758
grad_step = 000012, loss = 0.346532
grad_step = 000013, loss = 0.326532
grad_step = 000014, loss = 0.306812
grad_step = 000015, loss = 0.288570
grad_step = 000016, loss = 0.271632
grad_step = 000017, loss = 0.256198
grad_step = 000018, loss = 0.242757
grad_step = 000019, loss = 0.231378
grad_step = 000020, loss = 0.221156
grad_step = 000021, loss = 0.211955
grad_step = 000022, loss = 0.203150
grad_step = 000023, loss = 0.193943
grad_step = 000024, loss = 0.184501
grad_step = 000025, loss = 0.175212
grad_step = 000026, loss = 0.166049
grad_step = 000027, loss = 0.157253
grad_step = 000028, loss = 0.149053
grad_step = 000029, loss = 0.141450
grad_step = 000030, loss = 0.134602
grad_step = 000031, loss = 0.127981
grad_step = 000032, loss = 0.120953
grad_step = 000033, loss = 0.113947
grad_step = 000034, loss = 0.107433
grad_step = 000035, loss = 0.101418
grad_step = 000036, loss = 0.095819
grad_step = 000037, loss = 0.090479
grad_step = 000038, loss = 0.085461
grad_step = 000039, loss = 0.080755
grad_step = 000040, loss = 0.076220
grad_step = 000041, loss = 0.071855
grad_step = 000042, loss = 0.067631
grad_step = 000043, loss = 0.063223
grad_step = 000044, loss = 0.058857
grad_step = 000045, loss = 0.054739
grad_step = 000046, loss = 0.050948
grad_step = 000047, loss = 0.047502
grad_step = 000048, loss = 0.044446
grad_step = 000049, loss = 0.041654
grad_step = 000050, loss = 0.038934
grad_step = 000051, loss = 0.036350
grad_step = 000052, loss = 0.033988
grad_step = 000053, loss = 0.031767
grad_step = 000054, loss = 0.029592
grad_step = 000055, loss = 0.027506
grad_step = 000056, loss = 0.025581
grad_step = 000057, loss = 0.023803
grad_step = 000058, loss = 0.022065
grad_step = 000059, loss = 0.020350
grad_step = 000060, loss = 0.018765
grad_step = 000061, loss = 0.017359
grad_step = 000062, loss = 0.016070
grad_step = 000063, loss = 0.014838
grad_step = 000064, loss = 0.013687
grad_step = 000065, loss = 0.012640
grad_step = 000066, loss = 0.011654
grad_step = 000067, loss = 0.010713
grad_step = 000068, loss = 0.009858
grad_step = 000069, loss = 0.009088
grad_step = 000070, loss = 0.008362
grad_step = 000071, loss = 0.007689
grad_step = 000072, loss = 0.007102
grad_step = 000073, loss = 0.006571
grad_step = 000074, loss = 0.006065
grad_step = 000075, loss = 0.005607
grad_step = 000076, loss = 0.005199
grad_step = 000077, loss = 0.004818
grad_step = 000078, loss = 0.004475
grad_step = 000079, loss = 0.004178
grad_step = 000080, loss = 0.003911
grad_step = 000081, loss = 0.003670
grad_step = 000082, loss = 0.003460
grad_step = 000083, loss = 0.003271
grad_step = 000084, loss = 0.003101
grad_step = 000085, loss = 0.002952
grad_step = 000086, loss = 0.002820
grad_step = 000087, loss = 0.002704
grad_step = 000088, loss = 0.002601
grad_step = 000089, loss = 0.002511
grad_step = 000090, loss = 0.002435
grad_step = 000091, loss = 0.002369
grad_step = 000092, loss = 0.002310
grad_step = 000093, loss = 0.002260
grad_step = 000094, loss = 0.002215
grad_step = 000095, loss = 0.002175
grad_step = 000096, loss = 0.002144
grad_step = 000097, loss = 0.002115
grad_step = 000098, loss = 0.002088
grad_step = 000099, loss = 0.002069
grad_step = 000100, loss = 0.002057
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002060
grad_step = 000102, loss = 0.002084
grad_step = 000103, loss = 0.002092
grad_step = 000104, loss = 0.002052
grad_step = 000105, loss = 0.001990
grad_step = 000106, loss = 0.001988
grad_step = 000107, loss = 0.002020
grad_step = 000108, loss = 0.002009
grad_step = 000109, loss = 0.001967
grad_step = 000110, loss = 0.001953
grad_step = 000111, loss = 0.001973
grad_step = 000112, loss = 0.001977
grad_step = 000113, loss = 0.001947
grad_step = 000114, loss = 0.001928
grad_step = 000115, loss = 0.001937
grad_step = 000116, loss = 0.001944
grad_step = 000117, loss = 0.001931
grad_step = 000118, loss = 0.001911
grad_step = 000119, loss = 0.001904
grad_step = 000120, loss = 0.001909
grad_step = 000121, loss = 0.001910
grad_step = 000122, loss = 0.001900
grad_step = 000123, loss = 0.001884
grad_step = 000124, loss = 0.001874
grad_step = 000125, loss = 0.001873
grad_step = 000126, loss = 0.001875
grad_step = 000127, loss = 0.001875
grad_step = 000128, loss = 0.001868
grad_step = 000129, loss = 0.001858
grad_step = 000130, loss = 0.001847
grad_step = 000131, loss = 0.001839
grad_step = 000132, loss = 0.001835
grad_step = 000133, loss = 0.001835
grad_step = 000134, loss = 0.001830
grad_step = 000135, loss = 0.001830
grad_step = 000136, loss = 0.001839
grad_step = 000137, loss = 0.001848
grad_step = 000138, loss = 0.001862
grad_step = 000139, loss = 0.001881
grad_step = 000140, loss = 0.001902
grad_step = 000141, loss = 0.001879
grad_step = 000142, loss = 0.001832
grad_step = 000143, loss = 0.001790
grad_step = 000144, loss = 0.001789
grad_step = 000145, loss = 0.001809
grad_step = 000146, loss = 0.001819
grad_step = 000147, loss = 0.001813
grad_step = 000148, loss = 0.001791
grad_step = 000149, loss = 0.001775
grad_step = 000150, loss = 0.001768
grad_step = 000151, loss = 0.001766
grad_step = 000152, loss = 0.001771
grad_step = 000153, loss = 0.001781
grad_step = 000154, loss = 0.001781
grad_step = 000155, loss = 0.001759
grad_step = 000156, loss = 0.001744
grad_step = 000157, loss = 0.001739
grad_step = 000158, loss = 0.001738
grad_step = 000159, loss = 0.001739
grad_step = 000160, loss = 0.001740
grad_step = 000161, loss = 0.001743
grad_step = 000162, loss = 0.001748
grad_step = 000163, loss = 0.001747
grad_step = 000164, loss = 0.001738
grad_step = 000165, loss = 0.001729
grad_step = 000166, loss = 0.001719
grad_step = 000167, loss = 0.001709
grad_step = 000168, loss = 0.001707
grad_step = 000169, loss = 0.001716
grad_step = 000170, loss = 0.001714
grad_step = 000171, loss = 0.001721
grad_step = 000172, loss = 0.001703
grad_step = 000173, loss = 0.001696
grad_step = 000174, loss = 0.001697
grad_step = 000175, loss = 0.001710
grad_step = 000176, loss = 0.001724
grad_step = 000177, loss = 0.001736
grad_step = 000178, loss = 0.001759
grad_step = 000179, loss = 0.001820
grad_step = 000180, loss = 0.001882
grad_step = 000181, loss = 0.001954
grad_step = 000182, loss = 0.001797
grad_step = 000183, loss = 0.001709
grad_step = 000184, loss = 0.001732
grad_step = 000185, loss = 0.001743
grad_step = 000186, loss = 0.001821
grad_step = 000187, loss = 0.001793
grad_step = 000188, loss = 0.001687
grad_step = 000189, loss = 0.001836
grad_step = 000190, loss = 0.001794
grad_step = 000191, loss = 0.001774
grad_step = 000192, loss = 0.001682
grad_step = 000193, loss = 0.001766
grad_step = 000194, loss = 0.001744
grad_step = 000195, loss = 0.001755
grad_step = 000196, loss = 0.001693
grad_step = 000197, loss = 0.001690
grad_step = 000198, loss = 0.001725
grad_step = 000199, loss = 0.001717
grad_step = 000200, loss = 0.001682
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001652
grad_step = 000202, loss = 0.001677
grad_step = 000203, loss = 0.001656
grad_step = 000204, loss = 0.001681
grad_step = 000205, loss = 0.001636
grad_step = 000206, loss = 0.001647
grad_step = 000207, loss = 0.001631
grad_step = 000208, loss = 0.001654
grad_step = 000209, loss = 0.001629
grad_step = 000210, loss = 0.001642
grad_step = 000211, loss = 0.001615
grad_step = 000212, loss = 0.001624
grad_step = 000213, loss = 0.001616
grad_step = 000214, loss = 0.001618
grad_step = 000215, loss = 0.001611
grad_step = 000216, loss = 0.001608
grad_step = 000217, loss = 0.001604
grad_step = 000218, loss = 0.001595
grad_step = 000219, loss = 0.001599
grad_step = 000220, loss = 0.001588
grad_step = 000221, loss = 0.001591
grad_step = 000222, loss = 0.001584
grad_step = 000223, loss = 0.001586
grad_step = 000224, loss = 0.001577
grad_step = 000225, loss = 0.001577
grad_step = 000226, loss = 0.001571
grad_step = 000227, loss = 0.001567
grad_step = 000228, loss = 0.001565
grad_step = 000229, loss = 0.001559
grad_step = 000230, loss = 0.001558
grad_step = 000231, loss = 0.001553
grad_step = 000232, loss = 0.001550
grad_step = 000233, loss = 0.001547
grad_step = 000234, loss = 0.001543
grad_step = 000235, loss = 0.001544
grad_step = 000236, loss = 0.001546
grad_step = 000237, loss = 0.001558
grad_step = 000238, loss = 0.001590
grad_step = 000239, loss = 0.001676
grad_step = 000240, loss = 0.001811
grad_step = 000241, loss = 0.002022
grad_step = 000242, loss = 0.001921
grad_step = 000243, loss = 0.001657
grad_step = 000244, loss = 0.001529
grad_step = 000245, loss = 0.001720
grad_step = 000246, loss = 0.001756
grad_step = 000247, loss = 0.001533
grad_step = 000248, loss = 0.001600
grad_step = 000249, loss = 0.001714
grad_step = 000250, loss = 0.001540
grad_step = 000251, loss = 0.001528
grad_step = 000252, loss = 0.001650
grad_step = 000253, loss = 0.001578
grad_step = 000254, loss = 0.001496
grad_step = 000255, loss = 0.001536
grad_step = 000256, loss = 0.001583
grad_step = 000257, loss = 0.001545
grad_step = 000258, loss = 0.001478
grad_step = 000259, loss = 0.001574
grad_step = 000260, loss = 0.001595
grad_step = 000261, loss = 0.001505
grad_step = 000262, loss = 0.001545
grad_step = 000263, loss = 0.001546
grad_step = 000264, loss = 0.001541
grad_step = 000265, loss = 0.001467
grad_step = 000266, loss = 0.001502
grad_step = 000267, loss = 0.001518
grad_step = 000268, loss = 0.001456
grad_step = 000269, loss = 0.001480
grad_step = 000270, loss = 0.001475
grad_step = 000271, loss = 0.001454
grad_step = 000272, loss = 0.001464
grad_step = 000273, loss = 0.001441
grad_step = 000274, loss = 0.001434
grad_step = 000275, loss = 0.001448
grad_step = 000276, loss = 0.001427
grad_step = 000277, loss = 0.001423
grad_step = 000278, loss = 0.001425
grad_step = 000279, loss = 0.001409
grad_step = 000280, loss = 0.001418
grad_step = 000281, loss = 0.001406
grad_step = 000282, loss = 0.001400
grad_step = 000283, loss = 0.001400
grad_step = 000284, loss = 0.001391
grad_step = 000285, loss = 0.001398
grad_step = 000286, loss = 0.001394
grad_step = 000287, loss = 0.001381
grad_step = 000288, loss = 0.001385
grad_step = 000289, loss = 0.001378
grad_step = 000290, loss = 0.001379
grad_step = 000291, loss = 0.001382
grad_step = 000292, loss = 0.001365
grad_step = 000293, loss = 0.001363
grad_step = 000294, loss = 0.001357
grad_step = 000295, loss = 0.001352
grad_step = 000296, loss = 0.001353
grad_step = 000297, loss = 0.001348
grad_step = 000298, loss = 0.001348
grad_step = 000299, loss = 0.001343
grad_step = 000300, loss = 0.001336
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001335
grad_step = 000302, loss = 0.001330
grad_step = 000303, loss = 0.001324
grad_step = 000304, loss = 0.001323
grad_step = 000305, loss = 0.001318
grad_step = 000306, loss = 0.001315
grad_step = 000307, loss = 0.001314
grad_step = 000308, loss = 0.001312
grad_step = 000309, loss = 0.001315
grad_step = 000310, loss = 0.001326
grad_step = 000311, loss = 0.001345
grad_step = 000312, loss = 0.001408
grad_step = 000313, loss = 0.001392
grad_step = 000314, loss = 0.001404
grad_step = 000315, loss = 0.001352
grad_step = 000316, loss = 0.001309
grad_step = 000317, loss = 0.001294
grad_step = 000318, loss = 0.001315
grad_step = 000319, loss = 0.001351
grad_step = 000320, loss = 0.001349
grad_step = 000321, loss = 0.001348
grad_step = 000322, loss = 0.001301
grad_step = 000323, loss = 0.001281
grad_step = 000324, loss = 0.001286
grad_step = 000325, loss = 0.001315
grad_step = 000326, loss = 0.001353
grad_step = 000327, loss = 0.001317
grad_step = 000328, loss = 0.001277
grad_step = 000329, loss = 0.001273
grad_step = 000330, loss = 0.001290
grad_step = 000331, loss = 0.001300
grad_step = 000332, loss = 0.001276
grad_step = 000333, loss = 0.001269
grad_step = 000334, loss = 0.001278
grad_step = 000335, loss = 0.001269
grad_step = 000336, loss = 0.001268
grad_step = 000337, loss = 0.001265
grad_step = 000338, loss = 0.001254
grad_step = 000339, loss = 0.001247
grad_step = 000340, loss = 0.001252
grad_step = 000341, loss = 0.001263
grad_step = 000342, loss = 0.001269
grad_step = 000343, loss = 0.001280
grad_step = 000344, loss = 0.001296
grad_step = 000345, loss = 0.001351
grad_step = 000346, loss = 0.001313
grad_step = 000347, loss = 0.001288
grad_step = 000348, loss = 0.001259
grad_step = 000349, loss = 0.001241
grad_step = 000350, loss = 0.001233
grad_step = 000351, loss = 0.001246
grad_step = 000352, loss = 0.001260
grad_step = 000353, loss = 0.001248
grad_step = 000354, loss = 0.001223
grad_step = 000355, loss = 0.001214
grad_step = 000356, loss = 0.001226
grad_step = 000357, loss = 0.001234
grad_step = 000358, loss = 0.001225
grad_step = 000359, loss = 0.001209
grad_step = 000360, loss = 0.001203
grad_step = 000361, loss = 0.001209
grad_step = 000362, loss = 0.001209
grad_step = 000363, loss = 0.001201
grad_step = 000364, loss = 0.001192
grad_step = 000365, loss = 0.001191
grad_step = 000366, loss = 0.001193
grad_step = 000367, loss = 0.001194
grad_step = 000368, loss = 0.001190
grad_step = 000369, loss = 0.001186
grad_step = 000370, loss = 0.001197
grad_step = 000371, loss = 0.001234
grad_step = 000372, loss = 0.001387
grad_step = 000373, loss = 0.001442
grad_step = 000374, loss = 0.001657
grad_step = 000375, loss = 0.001418
grad_step = 000376, loss = 0.001377
grad_step = 000377, loss = 0.001450
grad_step = 000378, loss = 0.001305
grad_step = 000379, loss = 0.001538
grad_step = 000380, loss = 0.001628
grad_step = 000381, loss = 0.001410
grad_step = 000382, loss = 0.001582
grad_step = 000383, loss = 0.001502
grad_step = 000384, loss = 0.001523
grad_step = 000385, loss = 0.001284
grad_step = 000386, loss = 0.001463
grad_step = 000387, loss = 0.001480
grad_step = 000388, loss = 0.001325
grad_step = 000389, loss = 0.001509
grad_step = 000390, loss = 0.001281
grad_step = 000391, loss = 0.001468
grad_step = 000392, loss = 0.001327
grad_step = 000393, loss = 0.001389
grad_step = 000394, loss = 0.001358
grad_step = 000395, loss = 0.001311
grad_step = 000396, loss = 0.001311
grad_step = 000397, loss = 0.001201
grad_step = 000398, loss = 0.001279
grad_step = 000399, loss = 0.001180
grad_step = 000400, loss = 0.001268
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001160
grad_step = 000402, loss = 0.001228
grad_step = 000403, loss = 0.001144
grad_step = 000404, loss = 0.001225
grad_step = 000405, loss = 0.001136
grad_step = 000406, loss = 0.001189
grad_step = 000407, loss = 0.001132
grad_step = 000408, loss = 0.001166
grad_step = 000409, loss = 0.001134
grad_step = 000410, loss = 0.001142
grad_step = 000411, loss = 0.001136
grad_step = 000412, loss = 0.001122
grad_step = 000413, loss = 0.001129
grad_step = 000414, loss = 0.001110
grad_step = 000415, loss = 0.001125
grad_step = 000416, loss = 0.001109
grad_step = 000417, loss = 0.001109
grad_step = 000418, loss = 0.001104
grad_step = 000419, loss = 0.001097
grad_step = 000420, loss = 0.001103
grad_step = 000421, loss = 0.001093
grad_step = 000422, loss = 0.001090
grad_step = 000423, loss = 0.001090
grad_step = 000424, loss = 0.001080
grad_step = 000425, loss = 0.001088
grad_step = 000426, loss = 0.001076
grad_step = 000427, loss = 0.001079
grad_step = 000428, loss = 0.001074
grad_step = 000429, loss = 0.001070
grad_step = 000430, loss = 0.001072
grad_step = 000431, loss = 0.001065
grad_step = 000432, loss = 0.001068
grad_step = 000433, loss = 0.001062
grad_step = 000434, loss = 0.001060
grad_step = 000435, loss = 0.001059
grad_step = 000436, loss = 0.001054
grad_step = 000437, loss = 0.001055
grad_step = 000438, loss = 0.001051
grad_step = 000439, loss = 0.001049
grad_step = 000440, loss = 0.001048
grad_step = 000441, loss = 0.001044
grad_step = 000442, loss = 0.001043
grad_step = 000443, loss = 0.001040
grad_step = 000444, loss = 0.001038
grad_step = 000445, loss = 0.001037
grad_step = 000446, loss = 0.001034
grad_step = 000447, loss = 0.001033
grad_step = 000448, loss = 0.001030
grad_step = 000449, loss = 0.001028
grad_step = 000450, loss = 0.001026
grad_step = 000451, loss = 0.001024
grad_step = 000452, loss = 0.001022
grad_step = 000453, loss = 0.001020
grad_step = 000454, loss = 0.001017
grad_step = 000455, loss = 0.001016
grad_step = 000456, loss = 0.001014
grad_step = 000457, loss = 0.001011
grad_step = 000458, loss = 0.001010
grad_step = 000459, loss = 0.001008
grad_step = 000460, loss = 0.001007
grad_step = 000461, loss = 0.001010
grad_step = 000462, loss = 0.001019
grad_step = 000463, loss = 0.001030
grad_step = 000464, loss = 0.001037
grad_step = 000465, loss = 0.001030
grad_step = 000466, loss = 0.001022
grad_step = 000467, loss = 0.001035
grad_step = 000468, loss = 0.001070
grad_step = 000469, loss = 0.001149
grad_step = 000470, loss = 0.001106
grad_step = 000471, loss = 0.001096
grad_step = 000472, loss = 0.001076
grad_step = 000473, loss = 0.001053
grad_step = 000474, loss = 0.001014
grad_step = 000475, loss = 0.000984
grad_step = 000476, loss = 0.001002
grad_step = 000477, loss = 0.001031
grad_step = 000478, loss = 0.001024
grad_step = 000479, loss = 0.001002
grad_step = 000480, loss = 0.000989
grad_step = 000481, loss = 0.000990
grad_step = 000482, loss = 0.000995
grad_step = 000483, loss = 0.000976
grad_step = 000484, loss = 0.000964
grad_step = 000485, loss = 0.000968
grad_step = 000486, loss = 0.000973
grad_step = 000487, loss = 0.000976
grad_step = 000488, loss = 0.000975
grad_step = 000489, loss = 0.000987
grad_step = 000490, loss = 0.001023
grad_step = 000491, loss = 0.001102
grad_step = 000492, loss = 0.001165
grad_step = 000493, loss = 0.001263
grad_step = 000494, loss = 0.001185
grad_step = 000495, loss = 0.001075
grad_step = 000496, loss = 0.000965
grad_step = 000497, loss = 0.000971
grad_step = 000498, loss = 0.001046
grad_step = 000499, loss = 0.001043
grad_step = 000500, loss = 0.000991
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000962
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

  date_run                              2020-05-09 13:15:48.601047
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.235471
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 13:15:48.606678
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.142769
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 13:15:48.613121
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.131375
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 13:15:48.618208
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.16943
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
0   2020-05-09 13:15:16.509300  ...    mean_absolute_error
1   2020-05-09 13:15:16.513005  ...     mean_squared_error
2   2020-05-09 13:15:16.516175  ...  median_absolute_error
3   2020-05-09 13:15:16.519411  ...               r2_score
4   2020-05-09 13:15:27.302259  ...    mean_absolute_error
5   2020-05-09 13:15:27.306603  ...     mean_squared_error
6   2020-05-09 13:15:27.310188  ...  median_absolute_error
7   2020-05-09 13:15:27.313660  ...               r2_score
8   2020-05-09 13:15:48.601047  ...    mean_absolute_error
9   2020-05-09 13:15:48.606678  ...     mean_squared_error
10  2020-05-09 13:15:48.613121  ...  median_absolute_error
11  2020-05-09 13:15:48.618208  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 312648.80it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 405414.36it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 561836.00it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 793812.70it/s] 77%|███████▋  | 7651328/9912422 [00:00<00:02, 1122618.50it/s]9920512it [00:00, 10771479.37it/s]                            
0it [00:00, ?it/s]32768it [00:00, 383110.44it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 313547.99it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 405783.28it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 561692.78it/s]1654784it [00:00, 2833457.36it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 54676.04it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed760a20> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa98aea8a20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed71ce48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa98aea8e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed765710> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed765f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa98aeaa080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed765f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa98aeaa080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed765f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ##### 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9ed765710> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3fc2f1f208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=ed3410d57c143b6b4df6ef25cf877744091c7b88d7eda0aed76cb1389e00fab2
  Stored in directory: /tmp/pip-ephem-wheel-cache-ujj24nt0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3fb92a9080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 44s
   57344/17464789 [..............................] - ETA: 38s
   90112/17464789 [..............................] - ETA: 36s
  196608/17464789 [..............................] - ETA: 22s
  385024/17464789 [..............................] - ETA: 14s
  786432/17464789 [>.............................] - ETA: 8s 
 1589248/17464789 [=>............................] - ETA: 4s
 3186688/17464789 [====>.........................] - ETA: 2s
 6234112/17464789 [=========>....................] - ETA: 1s
 9150464/17464789 [==============>...............] - ETA: 0s
12050432/17464789 [===================>..........] - ETA: 0s
14934016/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 13:17:19.919449: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 13:17:19.923757: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-09 13:17:19.923902: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557d8ec220d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 13:17:19.923917: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9733 - accuracy: 0.4800
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6973 - accuracy: 0.4980
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6114 - accuracy: 0.5036
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6381 - accuracy: 0.5019
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6225 - accuracy: 0.5029
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6308 - accuracy: 0.5023
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6406 - accuracy: 0.5017
11000/25000 [============>.................] - ETA: 4s - loss: 7.6067 - accuracy: 0.5039
12000/25000 [=============>................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6407 - accuracy: 0.5017
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6655 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6714 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6865 - accuracy: 0.4987
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6760 - accuracy: 0.4994
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6942 - accuracy: 0.4982
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6914 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6973 - accuracy: 0.4980
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 9s 357us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 13:17:35.933739
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 13:17:35.933739  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 13:17:42.480566: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 13:17:42.485889: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-09 13:17:42.486121: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5633e0548490 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 13:17:42.486140: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9e3472cd30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 991ms/step - loss: 1.7149 - crf_viterbi_accuracy: 0.1867 - val_loss: 1.6374 - val_crf_viterbi_accuracy: 0.1600

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9e29ad4f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7228 - accuracy: 0.4963
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6728 - accuracy: 0.4996
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6162 - accuracy: 0.5033
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 4s - loss: 7.6304 - accuracy: 0.5024
12000/25000 [=============>................] - ETA: 3s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6466 - accuracy: 0.5013
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6502 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 3s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6340 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6549 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6445 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6408 - accuracy: 0.5017
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6265 - accuracy: 0.5026
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6464 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6486 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 9s 357us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f9e281ea438> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<34:55:01, 6.86kB/s].vector_cache/glove.6B.zip:   0%|          | 369k/862M [00:01<24:27:21, 9.79kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.21M/862M [00:01<17:01:27, 14.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.9M/862M [00:01<11:47:50, 20.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:01<8:10:53, 28.5kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<5:40:38, 40.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:01<3:57:04, 58.2kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:01<2:44:10, 83.1kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<1:53:52, 119kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<1:19:36, 169kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<14:28:12, 15.5kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<10:07:57, 22.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<7:05:50, 31.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<4:58:27, 44.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<3:30:09, 63.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<2:27:51, 90.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:45:05, 126kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<1:14:54, 177kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<54:06, 243kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:11<38:45, 340kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<28:59, 452kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<21:19, 614kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:15<16:48, 776kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:15<12:26, 1.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<10:42, 1.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<08:36, 1.51MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:19<07:54, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:19<06:51, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:21<06:39, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<05:36, 2.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:21<04:01, 3.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:23<32:21, 394kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:23<23:36, 540kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<18:20, 691kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<13:32, 936kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<11:24, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<08:56, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<08:06, 1.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:55, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:38, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:22, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:15, 2.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<7:17:20, 28.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<5:05:12, 40.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<3:37:42, 56.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<2:33:35, 80.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:48:36, 114kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<1:17:54, 158kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<55:24, 222kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<40:25, 302kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<29:01, 421kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<22:07, 550kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<16:21, 743kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<13:12, 915kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<09:56, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<08:49, 1.36MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<06:55, 1.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:41, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<05:27, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:37, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<05:09, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:19, 2.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:54, 2.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:07, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:33, 2.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<03:16, 3.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<25:51, 451kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<18:43, 623kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<14:51, 781kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<11:03, 1.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<09:29, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<07:19, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:54, 1.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:32, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<05:38, 2.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:39, 2.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:59, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:11, 2.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:38, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:57, 2.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<03:05, 3.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<02:32, 4.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<12:02:04, 15.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<8:25:38, 22.1kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<5:53:48, 31.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<4:08:07, 44.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<2:54:27, 63.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:02:42, 89.9kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<1:27:06, 126kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<1:01:31, 178kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<44:32, 245kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<32:12, 338kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<22:40, 477kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<11:18:55, 15.9kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<7:55:23, 22.8kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<5:32:39, 32.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<3:53:48, 46.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<2:44:16, 65.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<1:55:38, 92.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<1:22:05, 129kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<57:57, 183kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<42:00, 251kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<30:01, 351kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<22:30, 466kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<16:20, 641kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<12:59, 801kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<09:38, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<08:19, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<06:25, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<04:33, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<5:36:18, 30.5kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<3:55:55, 43.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<2:45:42, 61.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<1:57:02, 87.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<1:22:53, 122kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<58:28, 173kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<42:17, 238kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<30:10, 333kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<22:34, 443kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<16:47, 595kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<13:07, 756kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<09:45, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<07:12, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<5:50:40, 28.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<4:04:22, 40.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<2:55:08, 56.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<2:03:21, 79.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<1:25:53, 113kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<2:03:08, 79.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:26:43, 112kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<1:01:49, 156kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<43:43, 221kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<31:58, 300kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<23:07, 415kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<17:30, 545kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<12:46, 746kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<10:21, 914kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<07:44, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:51, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:22, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:11, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<04:07, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:19, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:34, 2.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:54, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:25, 2.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:45, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:09, 2.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:37, 2.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:59, 3.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:30, 2.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<03:20, 2.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:37, 2.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:02, 2.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:29, 2.53MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<02:57, 2.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<02:24, 3.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<5:16:48, 27.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<3:40:41, 39.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<2:37:37, 55.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<1:50:57, 78.6kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<1:18:25, 110kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<55:21, 156kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<39:48, 216kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<28:20, 303kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<21:02, 405kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<15:34, 547kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<12:03, 701kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<08:53, 949kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<07:29, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<05:51, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:19, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:22, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:28, 2.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:42, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:17, 2.48MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:29, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:16, 2.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:36<02:21, 3.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<08:30, 946kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<06:47, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:38<04:47, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<22:27, 355kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<16:08, 493kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<11:23, 696kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<10:15, 771kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<08:18, 951kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<05:55, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:42<04:14, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<39:37, 198kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<28:23, 276kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<19:53, 392kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<15:26, 504kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<4:33:30, 28.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<3:11:18, 40.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<2:13:23, 58.0kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<1:37:19, 79.3kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<1:08:36, 112kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<47:55, 160kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<35:48, 214kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<25:39, 298kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<17:57, 423kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<15:53, 477kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<11:45, 644kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<08:16, 910kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<09:05, 826kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:55, 1.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<04:56, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<05:41, 1.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<04:35, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<03:17, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<05:41, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:35, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:19, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:20, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:44, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<02:42, 2.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:09, 1.74MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:27, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:31, 2.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:48, 1.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:40, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<02:39, 2.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:03, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:26, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:05<02:28, 2.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<05:19, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:43, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<03:23, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:16, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:33, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<02:33, 2.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:40, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:30, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<04:26, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<3:50:35, 29.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<2:41:01, 42.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<1:52:03, 60.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<2:07:14, 53.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<1:29:29, 75.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<1:02:25, 108kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<45:36, 147kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<32:26, 206kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<22:40, 294kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<18:03, 367kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<13:10, 503kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<09:13, 713kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<10:15, 640kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<07:46, 843kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<05:30, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<06:07, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<04:50, 1.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:26, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<05:01, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:03, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:53, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:53, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:09, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<02:58, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<04:19, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<03:31, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:32, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:36, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:02, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<02:11, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:35, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:43, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<02:39, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:57, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:57, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<02:49, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:42, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:57, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:50, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:01, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<3:11:19, 31.2kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<2:13:37, 44.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<1:33:58, 62.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<1:06:17, 88.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<46:06, 127kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<32:38, 179kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<5:22:43, 18.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<3:46:07, 25.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<2:37:10, 36.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<1:52:26, 51.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<1:19:04, 72.7kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<54:59, 104kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<41:20, 138kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<29:22, 193kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<20:27, 276kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<17:10, 327kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<12:26, 451kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<08:44, 638kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<07:44, 718kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<05:49, 951kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<04:08, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<04:40, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<03:38, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:36, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:35, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:57, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:07, 2.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:59, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:16, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:21, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<03:19, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:50, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<02:03, 2.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:53, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:27, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [03:59<01:45, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:42, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:01, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<02:10, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:18, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<01:55, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:05, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:50, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:05<02:02, 2.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:58, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:41, 1.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<01:56, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:49, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:25, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:44, 2.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<03:58, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<2:48:05, 28.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<1:57:22, 40.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<1:21:23, 58.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<1:08:49, 68.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<48:31, 97.6kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<33:46, 139kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<25:00, 187kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<17:51, 261kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<12:26, 372kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<10:58, 420kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<08:01, 573kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<05:38, 809kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<05:27, 831kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:26, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<03:08, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:43, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:14, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:19, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:49, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:28, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:47, 2.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:27, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:06, 2.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<01:31, 2.82MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:25, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:03, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:26<01:28, 2.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<03:10, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:47, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<01:59, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:32, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:06, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:30, 2.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:13, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:35, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<01:50, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:51, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<02:19, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:34<01:38, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:59, 783kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:49, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<02:42, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<03:04, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:27, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<01:45, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:39, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:11, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:40<01:32, 2.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<04:11, 884kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:27, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<02:26, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:52, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:18, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:44, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<2:02:40, 29.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<1:25:31, 41.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<59:39, 59.0kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<41:57, 83.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<28:57, 119kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<24:43, 140kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<17:33, 196kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<12:07, 279kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<14:18, 236kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<10:29, 322kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<07:17, 457kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<06:20, 523kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:41, 704kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<03:15, 997kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<08:45, 370kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<06:24, 506kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<04:26, 717kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<06:13, 510kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<04:35, 691kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<03:12, 976kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<03:31, 881kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<02:41, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [04:59<01:53, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<03:00, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:23, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:01<01:40, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:54, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:17, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:01, 1.44MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:38, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:17, 2.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:20, 1.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:50, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:07<01:17, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:30, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:58, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:09<01:23, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<02:29, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<02:01, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:11<01:24, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<02:51, 920kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<02:12, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<01:32, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:37, 974kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:05, 1.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:32, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<1:28:34, 28.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<1:01:26, 40.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<42:45, 57.0kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<30:06, 80.8kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<20:39, 115kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<14:37, 162kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<2:26:56, 16.1kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<1:42:29, 23.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<1:10:20, 32.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<49:16, 46.5kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<33:53, 65.7kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<23:50, 93.2kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<16:32, 131kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<11:44, 183kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<08:05, 261kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<06:30, 321kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<04:50, 432kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<03:20, 611kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<02:25, 835kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<2:01:37, 16.6kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<1:24:59, 23.6kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<57:58, 33.6kB/s]  .vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<40:43, 47.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<27:47, 68.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<20:25, 92.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<14:24, 130kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<09:46, 185kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<10:33, 171kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<07:30, 240kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<05:07, 342kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<04:35, 379kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:24, 509kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<02:19, 722kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<02:40, 624kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:01, 821kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:41<01:23, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<02:21, 680kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:46, 860kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:22, 1.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:08, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:56, 1.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:38, 2.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:33, 900kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<01:10, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:48, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:45, 757kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:19, 995kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:50<00:54, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<01:33, 805kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:10, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:58, 1.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:49, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:54<00:33, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:42, 656kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:19, 840kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:56<00:53, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<05:29, 192kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<03:55, 266kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:58<02:36, 379kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<03:13, 305kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<02:17, 424kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<01:34, 592kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<34:48, 26.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<23:40, 38.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<15:59, 54.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<11:16, 76.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<07:28, 109kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<05:09, 154kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<48:23, 16.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<33:39, 23.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<21:48, 33.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<15:13, 47.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<09:48, 66.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<06:51, 94.7kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<04:23, 135kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<03:50, 153kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<02:42, 215kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<01:46, 293kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:15, 404kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:50, 531kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:36, 728kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:25, 895kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:20, 1.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:12, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:31, 585kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:22, 803kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:15, 970kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:11, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:06, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:10, 991kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:07, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:03, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:12, 488kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:09, 639kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:25<00:02, 906kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:07, 289kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:04, 401kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 767/400000 [00:00<00:52, 7662.44it/s]  0%|          | 1564/400000 [00:00<00:51, 7750.50it/s]  1%|          | 2404/400000 [00:00<00:50, 7934.12it/s]  1%|          | 3193/400000 [00:00<00:50, 7919.53it/s]  1%|          | 4015/400000 [00:00<00:49, 8006.35it/s]  1%|          | 4688/400000 [00:00<00:52, 7539.64it/s]  1%|▏         | 5519/400000 [00:00<00:50, 7753.69it/s]  2%|▏         | 6333/400000 [00:00<00:50, 7863.40it/s]  2%|▏         | 7155/400000 [00:00<00:49, 7966.30it/s]  2%|▏         | 7999/400000 [00:01<00:48, 8100.87it/s]  2%|▏         | 8814/400000 [00:01<00:48, 8113.39it/s]  2%|▏         | 9620/400000 [00:01<00:48, 8095.28it/s]  3%|▎         | 10453/400000 [00:01<00:47, 8161.10it/s]  3%|▎         | 11263/400000 [00:01<00:47, 8122.44it/s]  3%|▎         | 12081/400000 [00:01<00:47, 8138.74it/s]  3%|▎         | 12892/400000 [00:01<00:51, 7514.70it/s]  3%|▎         | 13651/400000 [00:01<00:53, 7218.05it/s]  4%|▎         | 14381/400000 [00:01<00:54, 7031.00it/s]  4%|▍         | 15091/400000 [00:01<00:55, 6924.55it/s]  4%|▍         | 15909/400000 [00:02<00:52, 7257.41it/s]  4%|▍         | 16726/400000 [00:02<00:51, 7506.90it/s]  4%|▍         | 17502/400000 [00:02<00:50, 7579.96it/s]  5%|▍         | 18305/400000 [00:02<00:49, 7708.96it/s]  5%|▍         | 19081/400000 [00:02<00:49, 7702.48it/s]  5%|▍         | 19876/400000 [00:02<00:48, 7774.96it/s]  5%|▌         | 20656/400000 [00:02<00:49, 7704.02it/s]  5%|▌         | 21446/400000 [00:02<00:48, 7761.40it/s]  6%|▌         | 22261/400000 [00:02<00:47, 7872.94it/s]  6%|▌         | 23092/400000 [00:02<00:47, 7997.89it/s]  6%|▌         | 23911/400000 [00:03<00:46, 8053.50it/s]  6%|▌         | 24718/400000 [00:03<00:47, 7984.28it/s]  6%|▋         | 25518/400000 [00:03<00:47, 7908.70it/s]  7%|▋         | 26357/400000 [00:03<00:46, 8045.58it/s]  7%|▋         | 27166/400000 [00:03<00:46, 8057.16it/s]  7%|▋         | 28010/400000 [00:03<00:45, 8166.64it/s]  7%|▋         | 28831/400000 [00:03<00:45, 8178.75it/s]  7%|▋         | 29650/400000 [00:03<00:46, 8035.53it/s]  8%|▊         | 30455/400000 [00:03<00:46, 7969.27it/s]  8%|▊         | 31279/400000 [00:03<00:45, 8047.55it/s]  8%|▊         | 32136/400000 [00:04<00:44, 8195.79it/s]  8%|▊         | 32957/400000 [00:04<00:45, 8081.36it/s]  8%|▊         | 33767/400000 [00:04<00:45, 8007.74it/s]  9%|▊         | 34569/400000 [00:04<00:45, 8004.69it/s]  9%|▉         | 35371/400000 [00:04<00:45, 7930.52it/s]  9%|▉         | 36189/400000 [00:04<00:45, 8001.98it/s]  9%|▉         | 36990/400000 [00:04<00:45, 7952.64it/s]  9%|▉         | 37812/400000 [00:04<00:45, 8028.34it/s] 10%|▉         | 38616/400000 [00:04<00:45, 7969.19it/s] 10%|▉         | 39463/400000 [00:04<00:44, 8110.48it/s] 10%|█         | 40278/400000 [00:05<00:44, 8120.44it/s] 10%|█         | 41091/400000 [00:05<00:44, 8097.72it/s] 10%|█         | 41918/400000 [00:05<00:43, 8146.87it/s] 11%|█         | 42734/400000 [00:05<00:45, 7781.08it/s] 11%|█         | 43516/400000 [00:05<00:48, 7407.73it/s] 11%|█         | 44264/400000 [00:05<00:50, 7044.70it/s] 11%|█         | 44977/400000 [00:05<00:50, 7012.59it/s] 11%|█▏        | 45831/400000 [00:05<00:47, 7410.17it/s] 12%|█▏        | 46674/400000 [00:05<00:45, 7688.42it/s] 12%|█▏        | 47494/400000 [00:06<00:45, 7832.58it/s] 12%|█▏        | 48285/400000 [00:06<00:46, 7628.86it/s] 12%|█▏        | 49055/400000 [00:06<00:48, 7300.39it/s] 12%|█▏        | 49793/400000 [00:06<00:47, 7314.07it/s] 13%|█▎        | 50629/400000 [00:06<00:45, 7598.15it/s] 13%|█▎        | 51475/400000 [00:06<00:44, 7835.23it/s] 13%|█▎        | 52286/400000 [00:06<00:43, 7913.62it/s] 13%|█▎        | 53148/400000 [00:06<00:42, 8112.88it/s] 13%|█▎        | 53964/400000 [00:06<00:45, 7679.57it/s] 14%|█▎        | 54747/400000 [00:07<00:44, 7723.48it/s] 14%|█▍        | 55525/400000 [00:07<00:44, 7671.55it/s] 14%|█▍        | 56349/400000 [00:07<00:43, 7832.01it/s] 14%|█▍        | 57182/400000 [00:07<00:42, 7974.89it/s] 15%|█▍        | 58015/400000 [00:07<00:42, 8075.85it/s] 15%|█▍        | 58863/400000 [00:07<00:41, 8191.39it/s] 15%|█▍        | 59719/400000 [00:07<00:41, 8298.09it/s] 15%|█▌        | 60551/400000 [00:07<00:40, 8290.76it/s] 15%|█▌        | 61411/400000 [00:07<00:40, 8379.07it/s] 16%|█▌        | 62273/400000 [00:07<00:39, 8448.24it/s] 16%|█▌        | 63134/400000 [00:08<00:39, 8495.32it/s] 16%|█▌        | 63999/400000 [00:08<00:39, 8539.52it/s] 16%|█▌        | 64854/400000 [00:08<00:39, 8466.89it/s] 16%|█▋        | 65702/400000 [00:08<00:41, 8129.02it/s] 17%|█▋        | 66519/400000 [00:08<00:44, 7547.95it/s] 17%|█▋        | 67285/400000 [00:08<00:45, 7276.40it/s] 17%|█▋        | 68023/400000 [00:08<00:47, 7024.38it/s] 17%|█▋        | 68838/400000 [00:08<00:45, 7325.65it/s] 17%|█▋        | 69700/400000 [00:08<00:43, 7669.68it/s] 18%|█▊        | 70548/400000 [00:08<00:41, 7893.28it/s] 18%|█▊        | 71347/400000 [00:09<00:41, 7853.98it/s] 18%|█▊        | 72139/400000 [00:09<00:42, 7736.18it/s] 18%|█▊        | 72958/400000 [00:09<00:41, 7864.23it/s] 18%|█▊        | 73793/400000 [00:09<00:40, 8001.97it/s] 19%|█▊        | 74618/400000 [00:09<00:40, 8068.51it/s] 19%|█▉        | 75448/400000 [00:09<00:39, 8133.36it/s] 19%|█▉        | 76264/400000 [00:09<00:40, 7902.22it/s] 19%|█▉        | 77080/400000 [00:09<00:40, 7975.91it/s] 19%|█▉        | 77880/400000 [00:09<00:41, 7842.23it/s] 20%|█▉        | 78667/400000 [00:10<00:42, 7491.84it/s] 20%|█▉        | 79451/400000 [00:10<00:42, 7592.93it/s] 20%|██        | 80251/400000 [00:10<00:41, 7709.76it/s] 20%|██        | 81025/400000 [00:10<00:42, 7514.07it/s] 20%|██        | 81780/400000 [00:10<00:44, 7221.76it/s] 21%|██        | 82622/400000 [00:10<00:42, 7542.05it/s] 21%|██        | 83475/400000 [00:10<00:40, 7802.45it/s] 21%|██        | 84263/400000 [00:10<00:41, 7556.31it/s] 21%|██▏       | 85026/400000 [00:10<00:43, 7301.81it/s] 21%|██▏       | 85763/400000 [00:10<00:44, 7074.99it/s] 22%|██▏       | 86477/400000 [00:11<00:45, 6932.81it/s] 22%|██▏       | 87176/400000 [00:11<00:45, 6917.80it/s] 22%|██▏       | 87913/400000 [00:11<00:44, 7047.09it/s] 22%|██▏       | 88706/400000 [00:11<00:42, 7290.18it/s] 22%|██▏       | 89516/400000 [00:11<00:41, 7506.48it/s] 23%|██▎       | 90344/400000 [00:11<00:40, 7722.50it/s] 23%|██▎       | 91156/400000 [00:11<00:39, 7837.25it/s] 23%|██▎       | 91944/400000 [00:11<00:39, 7721.25it/s] 23%|██▎       | 92786/400000 [00:11<00:38, 7917.05it/s] 23%|██▎       | 93589/400000 [00:11<00:38, 7948.75it/s] 24%|██▎       | 94436/400000 [00:12<00:37, 8096.35it/s] 24%|██▍       | 95281/400000 [00:12<00:37, 8197.55it/s] 24%|██▍       | 96103/400000 [00:12<00:37, 8088.10it/s] 24%|██▍       | 96914/400000 [00:12<00:37, 8006.08it/s] 24%|██▍       | 97722/400000 [00:12<00:37, 8026.32it/s] 25%|██▍       | 98557/400000 [00:12<00:37, 8118.15it/s] 25%|██▍       | 99393/400000 [00:12<00:36, 8186.75it/s] 25%|██▌       | 100213/400000 [00:12<00:37, 8011.68it/s] 25%|██▌       | 101064/400000 [00:12<00:36, 8154.54it/s] 25%|██▌       | 101912/400000 [00:13<00:36, 8246.91it/s] 26%|██▌       | 102739/400000 [00:13<00:36, 8192.31it/s] 26%|██▌       | 103560/400000 [00:13<00:36, 8134.16it/s] 26%|██▌       | 104375/400000 [00:13<00:36, 8062.43it/s] 26%|██▋       | 105200/400000 [00:13<00:36, 8117.44it/s] 27%|██▋       | 106013/400000 [00:13<00:37, 7774.66it/s] 27%|██▋       | 106794/400000 [00:13<00:39, 7439.86it/s] 27%|██▋       | 107544/400000 [00:13<00:39, 7420.22it/s] 27%|██▋       | 108360/400000 [00:13<00:38, 7626.35it/s] 27%|██▋       | 109191/400000 [00:13<00:37, 7817.97it/s] 28%|██▊       | 110009/400000 [00:14<00:36, 7921.95it/s] 28%|██▊       | 110809/400000 [00:14<00:36, 7943.43it/s] 28%|██▊       | 111606/400000 [00:14<00:36, 7936.77it/s] 28%|██▊       | 112402/400000 [00:14<00:38, 7504.53it/s] 28%|██▊       | 113159/400000 [00:14<00:39, 7337.42it/s] 28%|██▊       | 113939/400000 [00:14<00:38, 7470.04it/s] 29%|██▊       | 114691/400000 [00:14<00:39, 7223.52it/s] 29%|██▉       | 115418/400000 [00:14<00:39, 7130.39it/s] 29%|██▉       | 116183/400000 [00:14<00:38, 7278.56it/s] 29%|██▉       | 117029/400000 [00:14<00:37, 7596.63it/s] 29%|██▉       | 117848/400000 [00:15<00:36, 7763.06it/s] 30%|██▉       | 118630/400000 [00:15<00:37, 7447.15it/s] 30%|██▉       | 119381/400000 [00:15<00:38, 7253.43it/s] 30%|███       | 120195/400000 [00:15<00:37, 7497.61it/s] 30%|███       | 120988/400000 [00:15<00:36, 7621.70it/s] 30%|███       | 121760/400000 [00:15<00:36, 7648.77it/s] 31%|███       | 122562/400000 [00:15<00:35, 7755.53it/s] 31%|███       | 123341/400000 [00:15<00:35, 7752.44it/s] 31%|███       | 124204/400000 [00:15<00:34, 7995.86it/s] 31%|███▏      | 125059/400000 [00:16<00:33, 8153.79it/s] 31%|███▏      | 125878/400000 [00:16<00:35, 7740.79it/s] 32%|███▏      | 126659/400000 [00:16<00:37, 7341.77it/s] 32%|███▏      | 127402/400000 [00:16<00:38, 7091.41it/s] 32%|███▏      | 128173/400000 [00:16<00:37, 7265.32it/s] 32%|███▏      | 129021/400000 [00:16<00:35, 7590.98it/s] 32%|███▏      | 129864/400000 [00:16<00:34, 7822.78it/s] 33%|███▎      | 130654/400000 [00:16<00:34, 7765.11it/s] 33%|███▎      | 131437/400000 [00:16<00:37, 7247.27it/s] 33%|███▎      | 132173/400000 [00:17<00:37, 7059.70it/s] 33%|███▎      | 132918/400000 [00:17<00:37, 7170.31it/s] 33%|███▎      | 133741/400000 [00:17<00:35, 7456.88it/s] 34%|███▎      | 134527/400000 [00:17<00:35, 7570.98it/s] 34%|███▍      | 135292/400000 [00:17<00:34, 7594.17it/s] 34%|███▍      | 136158/400000 [00:17<00:33, 7882.87it/s] 34%|███▍      | 137013/400000 [00:17<00:32, 8070.04it/s] 34%|███▍      | 137876/400000 [00:17<00:31, 8228.93it/s] 35%|███▍      | 138704/400000 [00:17<00:31, 8225.04it/s] 35%|███▍      | 139549/400000 [00:17<00:31, 8291.03it/s] 35%|███▌      | 140384/400000 [00:18<00:31, 8307.91it/s] 35%|███▌      | 141217/400000 [00:18<00:31, 8263.64it/s] 36%|███▌      | 142077/400000 [00:18<00:30, 8360.24it/s] 36%|███▌      | 142915/400000 [00:18<00:31, 8197.42it/s] 36%|███▌      | 143737/400000 [00:18<00:31, 8034.90it/s] 36%|███▌      | 144543/400000 [00:18<00:32, 7982.81it/s] 36%|███▋      | 145383/400000 [00:18<00:31, 8101.49it/s] 37%|███▋      | 146231/400000 [00:18<00:30, 8210.56it/s] 37%|███▋      | 147054/400000 [00:18<00:30, 8201.21it/s] 37%|███▋      | 147876/400000 [00:18<00:31, 8111.11it/s] 37%|███▋      | 148732/400000 [00:19<00:30, 8240.30it/s] 37%|███▋      | 149558/400000 [00:19<00:30, 8205.76it/s] 38%|███▊      | 150403/400000 [00:19<00:30, 8275.95it/s] 38%|███▊      | 151232/400000 [00:19<00:30, 8171.67it/s] 38%|███▊      | 152050/400000 [00:19<00:30, 8129.22it/s] 38%|███▊      | 152912/400000 [00:19<00:29, 8268.80it/s] 38%|███▊      | 153744/400000 [00:19<00:29, 8282.48it/s] 39%|███▊      | 154605/400000 [00:19<00:29, 8376.79it/s] 39%|███▉      | 155444/400000 [00:19<00:29, 8332.98it/s] 39%|███▉      | 156278/400000 [00:19<00:30, 7920.08it/s] 39%|███▉      | 157075/400000 [00:20<00:31, 7749.95it/s] 39%|███▉      | 157855/400000 [00:20<00:31, 7745.34it/s] 40%|███▉      | 158670/400000 [00:20<00:30, 7861.00it/s] 40%|███▉      | 159459/400000 [00:20<00:32, 7401.22it/s] 40%|████      | 160207/400000 [00:20<00:32, 7315.29it/s] 40%|████      | 161056/400000 [00:20<00:31, 7632.04it/s] 40%|████      | 161916/400000 [00:20<00:30, 7898.12it/s] 41%|████      | 162767/400000 [00:20<00:29, 8062.67it/s] 41%|████      | 163580/400000 [00:20<00:29, 8055.00it/s] 41%|████      | 164448/400000 [00:21<00:28, 8232.82it/s] 41%|████▏     | 165276/400000 [00:21<00:29, 7931.42it/s] 42%|████▏     | 166075/400000 [00:21<00:31, 7539.11it/s] 42%|████▏     | 166837/400000 [00:21<00:30, 7531.35it/s] 42%|████▏     | 167678/400000 [00:21<00:29, 7773.25it/s] 42%|████▏     | 168530/400000 [00:21<00:28, 7982.52it/s] 42%|████▏     | 169397/400000 [00:21<00:28, 8175.81it/s] 43%|████▎     | 170220/400000 [00:21<00:29, 7849.56it/s] 43%|████▎     | 171016/400000 [00:21<00:29, 7882.26it/s] 43%|████▎     | 171862/400000 [00:21<00:28, 8044.63it/s] 43%|████▎     | 172671/400000 [00:22<00:28, 7920.81it/s] 43%|████▎     | 173505/400000 [00:22<00:28, 8041.30it/s] 44%|████▎     | 174347/400000 [00:22<00:27, 8151.13it/s] 44%|████▍     | 175165/400000 [00:22<00:28, 7949.04it/s] 44%|████▍     | 175986/400000 [00:22<00:27, 8025.35it/s] 44%|████▍     | 176791/400000 [00:22<00:27, 8032.43it/s] 44%|████▍     | 177602/400000 [00:22<00:27, 8053.55it/s] 45%|████▍     | 178469/400000 [00:22<00:26, 8227.54it/s] 45%|████▍     | 179294/400000 [00:22<00:26, 8196.61it/s] 45%|████▌     | 180147/400000 [00:22<00:26, 8291.93it/s] 45%|████▌     | 181015/400000 [00:23<00:26, 8402.28it/s] 45%|████▌     | 181867/400000 [00:23<00:25, 8437.01it/s] 46%|████▌     | 182734/400000 [00:23<00:25, 8504.97it/s] 46%|████▌     | 183586/400000 [00:23<00:28, 7714.59it/s] 46%|████▌     | 184373/400000 [00:23<00:29, 7407.82it/s] 46%|████▋     | 185127/400000 [00:23<00:30, 7141.78it/s] 46%|████▋     | 185853/400000 [00:23<00:30, 7090.91it/s] 47%|████▋     | 186703/400000 [00:23<00:28, 7460.54it/s] 47%|████▋     | 187534/400000 [00:23<00:27, 7694.36it/s] 47%|████▋     | 188392/400000 [00:24<00:26, 7939.51it/s] 47%|████▋     | 189260/400000 [00:24<00:25, 8147.84it/s] 48%|████▊     | 190126/400000 [00:24<00:25, 8294.76it/s] 48%|████▊     | 190978/400000 [00:24<00:25, 8359.19it/s] 48%|████▊     | 191819/400000 [00:24<00:24, 8329.14it/s] 48%|████▊     | 192655/400000 [00:24<00:25, 8131.35it/s] 48%|████▊     | 193472/400000 [00:24<00:26, 7658.72it/s] 49%|████▊     | 194246/400000 [00:24<00:27, 7354.38it/s] 49%|████▊     | 194990/400000 [00:24<00:29, 6935.03it/s] 49%|████▉     | 195695/400000 [00:25<00:29, 6872.13it/s] 49%|████▉     | 196513/400000 [00:25<00:28, 7217.74it/s] 49%|████▉     | 197376/400000 [00:25<00:26, 7590.11it/s] 50%|████▉     | 198238/400000 [00:25<00:25, 7872.13it/s] 50%|████▉     | 199067/400000 [00:25<00:25, 7988.90it/s] 50%|████▉     | 199875/400000 [00:25<00:26, 7612.02it/s] 50%|█████     | 200692/400000 [00:25<00:25, 7770.73it/s] 50%|█████     | 201550/400000 [00:25<00:24, 7995.99it/s] 51%|█████     | 202413/400000 [00:25<00:24, 8174.86it/s] 51%|█████     | 203237/400000 [00:25<00:24, 8178.77it/s] 51%|█████     | 204095/400000 [00:26<00:23, 8294.37it/s] 51%|█████     | 204960/400000 [00:26<00:23, 8397.33it/s] 51%|█████▏    | 205817/400000 [00:26<00:22, 8448.21it/s] 52%|█████▏    | 206664/400000 [00:26<00:22, 8436.87it/s] 52%|█████▏    | 207509/400000 [00:26<00:23, 8316.53it/s] 52%|█████▏    | 208342/400000 [00:26<00:24, 7700.94it/s] 52%|█████▏    | 209123/400000 [00:26<00:25, 7453.33it/s] 52%|█████▏    | 209973/400000 [00:26<00:24, 7737.11it/s] 53%|█████▎    | 210820/400000 [00:26<00:23, 7941.57it/s] 53%|█████▎    | 211646/400000 [00:26<00:23, 8033.01it/s] 53%|█████▎    | 212503/400000 [00:27<00:22, 8184.96it/s] 53%|█████▎    | 213360/400000 [00:27<00:22, 8296.16it/s] 54%|█████▎    | 214217/400000 [00:27<00:22, 8375.45it/s] 54%|█████▍    | 215082/400000 [00:27<00:21, 8454.41it/s] 54%|█████▍    | 215930/400000 [00:27<00:22, 8010.44it/s] 54%|█████▍    | 216738/400000 [00:27<00:24, 7585.98it/s] 54%|█████▍    | 217506/400000 [00:27<00:24, 7536.78it/s] 55%|█████▍    | 218320/400000 [00:27<00:23, 7706.54it/s] 55%|█████▍    | 219173/400000 [00:27<00:22, 7935.99it/s] 55%|█████▌    | 220013/400000 [00:28<00:22, 8067.42it/s] 55%|█████▌    | 220855/400000 [00:28<00:21, 8169.64it/s] 55%|█████▌    | 221676/400000 [00:28<00:22, 7801.48it/s] 56%|█████▌    | 222537/400000 [00:28<00:22, 8026.17it/s] 56%|█████▌    | 223389/400000 [00:28<00:21, 8168.19it/s] 56%|█████▌    | 224225/400000 [00:28<00:21, 8224.66it/s] 56%|█████▋    | 225051/400000 [00:28<00:22, 7924.65it/s] 56%|█████▋    | 225849/400000 [00:28<00:23, 7543.27it/s] 57%|█████▋    | 226611/400000 [00:28<00:23, 7272.24it/s] 57%|█████▋    | 227346/400000 [00:28<00:24, 7078.43it/s] 57%|█████▋    | 228185/400000 [00:29<00:23, 7425.72it/s] 57%|█████▋    | 229025/400000 [00:29<00:22, 7692.31it/s] 57%|█████▋    | 229859/400000 [00:29<00:21, 7874.44it/s] 58%|█████▊    | 230716/400000 [00:29<00:20, 8069.83it/s] 58%|█████▊    | 231536/400000 [00:29<00:20, 8095.20it/s] 58%|█████▊    | 232370/400000 [00:29<00:20, 8166.46it/s] 58%|█████▊    | 233201/400000 [00:29<00:20, 8206.69it/s] 59%|█████▊    | 234024/400000 [00:29<00:20, 8024.41it/s] 59%|█████▊    | 234830/400000 [00:29<00:20, 8032.84it/s] 59%|█████▉    | 235653/400000 [00:29<00:20, 8090.61it/s] 59%|█████▉    | 236486/400000 [00:30<00:20, 8158.87it/s] 59%|█████▉    | 237322/400000 [00:30<00:19, 8215.53it/s] 60%|█████▉    | 238182/400000 [00:30<00:19, 8326.22it/s] 60%|█████▉    | 239041/400000 [00:30<00:19, 8403.59it/s] 60%|█████▉    | 239883/400000 [00:30<00:19, 8369.63it/s] 60%|██████    | 240721/400000 [00:30<00:20, 7832.06it/s] 60%|██████    | 241512/400000 [00:30<00:21, 7366.71it/s] 61%|██████    | 242260/400000 [00:30<00:22, 7122.81it/s] 61%|██████    | 242982/400000 [00:30<00:22, 6944.48it/s] 61%|██████    | 243813/400000 [00:31<00:21, 7302.97it/s] 61%|██████    | 244636/400000 [00:31<00:20, 7555.58it/s] 61%|██████▏   | 245444/400000 [00:31<00:20, 7705.35it/s] 62%|██████▏   | 246294/400000 [00:31<00:19, 7927.12it/s] 62%|██████▏   | 247094/400000 [00:31<00:20, 7609.06it/s] 62%|██████▏   | 247925/400000 [00:31<00:19, 7806.32it/s] 62%|██████▏   | 248787/400000 [00:31<00:18, 8032.66it/s] 62%|██████▏   | 249597/400000 [00:31<00:19, 7683.81it/s] 63%|██████▎   | 250440/400000 [00:31<00:18, 7891.28it/s] 63%|██████▎   | 251264/400000 [00:31<00:18, 7990.28it/s] 63%|██████▎   | 252103/400000 [00:32<00:18, 8105.78it/s] 63%|██████▎   | 252965/400000 [00:32<00:17, 8251.37it/s] 63%|██████▎   | 253827/400000 [00:32<00:17, 8357.97it/s] 64%|██████▎   | 254666/400000 [00:32<00:17, 8212.43it/s] 64%|██████▍   | 255506/400000 [00:32<00:17, 8266.50it/s] 64%|██████▍   | 256335/400000 [00:32<00:18, 7894.15it/s] 64%|██████▍   | 257130/400000 [00:32<00:19, 7487.66it/s] 64%|██████▍   | 257887/400000 [00:32<00:19, 7230.68it/s] 65%|██████▍   | 258618/400000 [00:32<00:20, 7066.72it/s] 65%|██████▍   | 259331/400000 [00:33<00:20, 6892.06it/s] 65%|██████▌   | 260170/400000 [00:33<00:19, 7281.23it/s] 65%|██████▌   | 261033/400000 [00:33<00:18, 7637.50it/s] 65%|██████▌   | 261885/400000 [00:33<00:17, 7881.19it/s] 66%|██████▌   | 262737/400000 [00:33<00:17, 8061.98it/s] 66%|██████▌   | 263573/400000 [00:33<00:16, 8146.76it/s] 66%|██████▌   | 264394/400000 [00:33<00:16, 8048.48it/s] 66%|██████▋   | 265240/400000 [00:33<00:16, 8165.89it/s] 67%|██████▋   | 266103/400000 [00:33<00:16, 8297.28it/s] 67%|██████▋   | 266954/400000 [00:33<00:15, 8358.93it/s] 67%|██████▋   | 267793/400000 [00:34<00:15, 8284.85it/s] 67%|██████▋   | 268630/400000 [00:34<00:15, 8308.01it/s] 67%|██████▋   | 269488/400000 [00:34<00:15, 8385.91it/s] 68%|██████▊   | 270350/400000 [00:34<00:15, 8451.92it/s] 68%|██████▊   | 271197/400000 [00:34<00:15, 8385.96it/s] 68%|██████▊   | 272037/400000 [00:34<00:15, 8314.30it/s] 68%|██████▊   | 272870/400000 [00:34<00:16, 7839.60it/s] 68%|██████▊   | 273660/400000 [00:34<00:16, 7848.59it/s] 69%|██████▊   | 274518/400000 [00:34<00:15, 8052.41it/s] 69%|██████▉   | 275380/400000 [00:35<00:15, 8213.06it/s] 69%|██████▉   | 276220/400000 [00:35<00:14, 8267.60it/s] 69%|██████▉   | 277053/400000 [00:35<00:14, 8283.83it/s] 69%|██████▉   | 277887/400000 [00:35<00:14, 8298.83it/s] 70%|██████▉   | 278736/400000 [00:35<00:14, 8355.24it/s] 70%|██████▉   | 279594/400000 [00:35<00:14, 8420.87it/s] 70%|███████   | 280437/400000 [00:35<00:14, 8417.65it/s] 70%|███████   | 281283/400000 [00:35<00:14, 8429.87it/s] 71%|███████   | 282132/400000 [00:35<00:13, 8443.69it/s] 71%|███████   | 282977/400000 [00:35<00:14, 8061.75it/s] 71%|███████   | 283788/400000 [00:36<00:14, 8043.44it/s] 71%|███████   | 284596/400000 [00:36<00:14, 7992.51it/s] 71%|███████▏  | 285398/400000 [00:36<00:14, 7933.86it/s] 72%|███████▏  | 286193/400000 [00:36<00:14, 7912.61it/s] 72%|███████▏  | 287033/400000 [00:36<00:14, 8052.19it/s] 72%|███████▏  | 287882/400000 [00:36<00:13, 8177.79it/s] 72%|███████▏  | 288706/400000 [00:36<00:13, 8195.66it/s] 72%|███████▏  | 289531/400000 [00:36<00:13, 8209.58it/s] 73%|███████▎  | 290374/400000 [00:36<00:13, 8273.83it/s] 73%|███████▎  | 291224/400000 [00:36<00:13, 8339.23it/s] 73%|███████▎  | 292075/400000 [00:37<00:12, 8389.64it/s] 73%|███████▎  | 292925/400000 [00:37<00:12, 8422.22it/s] 73%|███████▎  | 293768/400000 [00:37<00:12, 8362.71it/s] 74%|███████▎  | 294605/400000 [00:37<00:12, 8355.36it/s] 74%|███████▍  | 295441/400000 [00:37<00:12, 8209.86it/s] 74%|███████▍  | 296263/400000 [00:37<00:12, 8105.83it/s] 74%|███████▍  | 297075/400000 [00:37<00:13, 7730.08it/s] 74%|███████▍  | 297913/400000 [00:37<00:12, 7911.90it/s] 75%|███████▍  | 298759/400000 [00:37<00:12, 8067.09it/s] 75%|███████▍  | 299570/400000 [00:37<00:12, 8052.20it/s] 75%|███████▌  | 300418/400000 [00:38<00:12, 8174.14it/s] 75%|███████▌  | 301248/400000 [00:38<00:12, 8209.78it/s] 76%|███████▌  | 302109/400000 [00:38<00:11, 8325.04it/s] 76%|███████▌  | 302943/400000 [00:38<00:11, 8256.90it/s] 76%|███████▌  | 303787/400000 [00:38<00:11, 8309.15it/s] 76%|███████▌  | 304620/400000 [00:38<00:11, 8313.54it/s] 76%|███████▋  | 305452/400000 [00:38<00:11, 8079.90it/s] 77%|███████▋  | 306296/400000 [00:38<00:11, 8182.58it/s] 77%|███████▋  | 307158/400000 [00:38<00:11, 8306.93it/s] 77%|███████▋  | 307991/400000 [00:38<00:11, 8246.82it/s] 77%|███████▋  | 308817/400000 [00:39<00:11, 8250.48it/s] 77%|███████▋  | 309643/400000 [00:39<00:10, 8242.91it/s] 78%|███████▊  | 310490/400000 [00:39<00:10, 8308.68it/s] 78%|███████▊  | 311348/400000 [00:39<00:10, 8385.76it/s] 78%|███████▊  | 312209/400000 [00:39<00:10, 8450.78it/s] 78%|███████▊  | 313055/400000 [00:39<00:11, 7440.26it/s] 78%|███████▊  | 313822/400000 [00:39<00:12, 7071.90it/s] 79%|███████▊  | 314654/400000 [00:39<00:11, 7404.16it/s] 79%|███████▉  | 315492/400000 [00:39<00:11, 7670.63it/s] 79%|███████▉  | 316274/400000 [00:40<00:11, 7407.29it/s] 79%|███████▉  | 317028/400000 [00:40<00:11, 7125.40it/s] 79%|███████▉  | 317752/400000 [00:40<00:11, 6936.22it/s] 80%|███████▉  | 318458/400000 [00:40<00:11, 6972.34it/s] 80%|███████▉  | 319289/400000 [00:40<00:11, 7325.67it/s] 80%|████████  | 320109/400000 [00:40<00:10, 7566.84it/s] 80%|████████  | 320920/400000 [00:40<00:10, 7720.46it/s] 80%|████████  | 321764/400000 [00:40<00:09, 7921.58it/s] 81%|████████  | 322626/400000 [00:40<00:09, 8116.41it/s] 81%|████████  | 323443/400000 [00:40<00:09, 8090.71it/s] 81%|████████  | 324256/400000 [00:41<00:09, 7761.99it/s] 81%|████████▏ | 325038/400000 [00:41<00:10, 7276.79it/s] 81%|████████▏ | 325897/400000 [00:41<00:09, 7626.49it/s] 82%|████████▏ | 326749/400000 [00:41<00:09, 7873.91it/s] 82%|████████▏ | 327583/400000 [00:41<00:09, 8005.59it/s] 82%|████████▏ | 328440/400000 [00:41<00:08, 8166.28it/s] 82%|████████▏ | 329263/400000 [00:41<00:08, 8006.67it/s] 83%|████████▎ | 330106/400000 [00:41<00:08, 8128.18it/s] 83%|████████▎ | 330964/400000 [00:41<00:08, 8256.17it/s] 83%|████████▎ | 331824/400000 [00:42<00:08, 8353.87it/s] 83%|████████▎ | 332687/400000 [00:42<00:07, 8434.68it/s] 83%|████████▎ | 333533/400000 [00:42<00:07, 8384.38it/s] 84%|████████▎ | 334387/400000 [00:42<00:07, 8429.85it/s] 84%|████████▍ | 335232/400000 [00:42<00:07, 8390.80it/s] 84%|████████▍ | 336072/400000 [00:42<00:07, 8389.88it/s] 84%|████████▍ | 336912/400000 [00:42<00:08, 7841.28it/s] 84%|████████▍ | 337704/400000 [00:42<00:08, 7398.62it/s] 85%|████████▍ | 338455/400000 [00:42<00:08, 7108.91it/s] 85%|████████▍ | 339202/400000 [00:42<00:08, 7212.39it/s] 85%|████████▌ | 340056/400000 [00:43<00:07, 7552.91it/s] 85%|████████▌ | 340900/400000 [00:43<00:07, 7797.18it/s] 85%|████████▌ | 341689/400000 [00:43<00:07, 7535.42it/s] 86%|████████▌ | 342451/400000 [00:43<00:07, 7545.85it/s] 86%|████████▌ | 343320/400000 [00:43<00:07, 7853.86it/s] 86%|████████▌ | 344113/400000 [00:43<00:07, 7754.66it/s] 86%|████████▌ | 344939/400000 [00:43<00:06, 7899.53it/s] 86%|████████▋ | 345781/400000 [00:43<00:06, 8046.21it/s] 87%|████████▋ | 346638/400000 [00:43<00:06, 8196.07it/s] 87%|████████▋ | 347500/400000 [00:44<00:06, 8316.75it/s] 87%|████████▋ | 348355/400000 [00:44<00:06, 8384.33it/s] 87%|████████▋ | 349196/400000 [00:44<00:06, 8238.42it/s] 88%|████████▊ | 350022/400000 [00:44<00:06, 8040.11it/s] 88%|████████▊ | 350829/400000 [00:44<00:06, 7730.50it/s] 88%|████████▊ | 351625/400000 [00:44<00:06, 7797.72it/s] 88%|████████▊ | 352408/400000 [00:44<00:06, 7572.28it/s] 88%|████████▊ | 353205/400000 [00:44<00:06, 7685.39it/s] 89%|████████▊ | 354036/400000 [00:44<00:05, 7860.48it/s] 89%|████████▊ | 354892/400000 [00:44<00:05, 8055.81it/s] 89%|████████▉ | 355752/400000 [00:45<00:05, 8211.14it/s] 89%|████████▉ | 356611/400000 [00:45<00:05, 8319.51it/s] 89%|████████▉ | 357446/400000 [00:45<00:05, 8258.16it/s] 90%|████████▉ | 358277/400000 [00:45<00:05, 8270.72it/s] 90%|████████▉ | 359106/400000 [00:45<00:05, 7958.93it/s] 90%|████████▉ | 359906/400000 [00:45<00:05, 7492.44it/s] 90%|█████████ | 360664/400000 [00:45<00:05, 7203.81it/s] 90%|█████████ | 361393/400000 [00:45<00:05, 7177.27it/s] 91%|█████████ | 362177/400000 [00:45<00:05, 7362.98it/s] 91%|█████████ | 363021/400000 [00:46<00:04, 7653.72it/s] 91%|█████████ | 363826/400000 [00:46<00:04, 7766.81it/s] 91%|█████████ | 364636/400000 [00:46<00:04, 7861.73it/s] 91%|█████████▏| 365430/400000 [00:46<00:04, 7883.21it/s] 92%|█████████▏| 366260/400000 [00:46<00:04, 8002.51it/s] 92%|█████████▏| 367111/400000 [00:46<00:04, 8147.57it/s] 92%|█████████▏| 367962/400000 [00:46<00:03, 8252.64it/s] 92%|█████████▏| 368810/400000 [00:46<00:03, 8317.40it/s] 92%|█████████▏| 369644/400000 [00:46<00:03, 8279.01it/s] 93%|█████████▎| 370480/400000 [00:46<00:03, 8300.37it/s] 93%|█████████▎| 371341/400000 [00:47<00:03, 8390.89it/s] 93%|█████████▎| 372196/400000 [00:47<00:03, 8435.01it/s] 93%|█████████▎| 373045/400000 [00:47<00:03, 8449.20it/s] 93%|█████████▎| 373891/400000 [00:47<00:03, 8302.93it/s] 94%|█████████▎| 374723/400000 [00:47<00:03, 8165.40it/s] 94%|█████████▍| 375577/400000 [00:47<00:02, 8272.57it/s] 94%|█████████▍| 376406/400000 [00:47<00:02, 8076.83it/s] 94%|█████████▍| 377216/400000 [00:47<00:03, 7506.32it/s] 94%|█████████▍| 377976/400000 [00:47<00:03, 7217.69it/s] 95%|█████████▍| 378707/400000 [00:47<00:03, 7042.27it/s] 95%|█████████▍| 379419/400000 [00:48<00:02, 6964.64it/s] 95%|█████████▌| 380247/400000 [00:48<00:02, 7312.77it/s] 95%|█████████▌| 381080/400000 [00:48<00:02, 7590.28it/s] 95%|█████████▌| 381928/400000 [00:48<00:02, 7834.85it/s] 96%|█████████▌| 382775/400000 [00:48<00:02, 8015.01it/s] 96%|█████████▌| 383605/400000 [00:48<00:02, 8098.21it/s] 96%|█████████▌| 384441/400000 [00:48<00:01, 8174.89it/s] 96%|█████████▋| 385286/400000 [00:48<00:01, 8251.27it/s] 97%|█████████▋| 386114/400000 [00:48<00:01, 7801.25it/s] 97%|█████████▋| 386949/400000 [00:49<00:01, 7956.57it/s] 97%|█████████▋| 387751/400000 [00:49<00:01, 7801.66it/s] 97%|█████████▋| 388536/400000 [00:49<00:01, 7466.90it/s] 97%|█████████▋| 389289/400000 [00:49<00:01, 7130.85it/s] 98%|█████████▊| 390010/400000 [00:49<00:01, 7149.16it/s] 98%|█████████▊| 390849/400000 [00:49<00:01, 7479.17it/s] 98%|█████████▊| 391678/400000 [00:49<00:01, 7704.19it/s] 98%|█████████▊| 392539/400000 [00:49<00:00, 7955.13it/s] 98%|█████████▊| 393342/400000 [00:49<00:00, 7708.37it/s] 99%|█████████▊| 394120/400000 [00:49<00:00, 7423.68it/s] 99%|█████████▊| 394870/400000 [00:50<00:00, 7151.75it/s] 99%|█████████▉| 395593/400000 [00:50<00:00, 6966.53it/s] 99%|█████████▉| 396296/400000 [00:50<00:00, 6866.63it/s] 99%|█████████▉| 397013/400000 [00:50<00:00, 6954.54it/s] 99%|█████████▉| 397800/400000 [00:50<00:00, 7204.92it/s]100%|█████████▉| 398526/400000 [00:50<00:00, 7080.13it/s]100%|█████████▉| 399238/400000 [00:50<00:00, 6951.19it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7871.80it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9e5cdf7518> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011365097479758855 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.01085364818572998 	 Accuracy: 59

  model saves at 59% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  ### Calculate Metrics    ######################################## 

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


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

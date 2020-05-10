
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3310bdd90ac91975df813c6632e92d33a8bcfcdd', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '3310bdd90ac91975df813c6632e92d33a8bcfcdd', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3310bdd90ac91975df813c6632e92d33a8bcfcdd

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3310bdd90ac91975df813c6632e92d33a8bcfcdd

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6bdc4e84a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 06:13:34.814113
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 06:13:34.819669
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 06:13:34.824210
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 06:13:34.828962
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6bd4838400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 359398.5312
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 314086.0625
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 228616.6406
Epoch 4/10

1/1 [==============================] - 0s 107ms/step - loss: 153617.5469
Epoch 5/10

1/1 [==============================] - 0s 116ms/step - loss: 96929.2578
Epoch 6/10

1/1 [==============================] - 0s 120ms/step - loss: 59216.4609
Epoch 7/10

1/1 [==============================] - 0s 107ms/step - loss: 37162.4336
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 24580.2383
Epoch 9/10

1/1 [==============================] - 0s 117ms/step - loss: 17245.9688
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 12745.8262

  #### Inference Need return ypred, ytrue ######################### 
[[-1.5369169e-03  6.1899338e+00  5.4515429e+00  4.8735805e+00
   4.2159228e+00  5.3057213e+00  6.3104506e+00  5.5583801e+00
   5.0622921e+00  4.9869404e+00  6.0429339e+00  6.8085418e+00
   6.7023082e+00  5.0687842e+00  5.8372693e+00  4.7098012e+00
   5.0732903e+00  5.2726989e+00  4.5170345e+00  5.5932198e+00
   5.4243579e+00  6.2264905e+00  4.9213758e+00  4.8772550e+00
   5.0566845e+00  4.8488269e+00  6.1525130e+00  5.8426514e+00
   5.4013166e+00  4.3385429e+00  6.2071781e+00  5.9247260e+00
   5.8694930e+00  5.7412291e+00  4.9563193e+00  5.5148325e+00
   5.4364958e+00  4.6281786e+00  5.1926479e+00  5.4735188e+00
   5.9064507e+00  4.9157124e+00  5.7222261e+00  5.6042457e+00
   5.3339124e+00  6.2877407e+00  5.5631571e+00  5.5315695e+00
   6.4828534e+00  6.4440022e+00  4.7627683e+00  5.2411723e+00
   5.1202974e+00  5.7844162e+00  5.4375796e+00  5.7330441e+00
   4.6753550e+00  4.8334394e+00  6.2732668e+00  4.9255724e+00
   5.3805590e-01 -6.8767482e-01  2.8573561e-01 -8.9409865e-02
  -7.1482438e-01 -3.2366823e-02  8.1439853e-01  3.9914042e-01
  -5.0052309e-01 -4.3353617e-01  8.7119728e-02  3.4186623e-01
   6.1612308e-01 -2.6282936e-01 -3.1094275e-02 -8.4427214e-01
  -2.2657758e-01 -1.9514655e-01 -4.7382377e-02 -8.5720444e-01
  -4.6311682e-01  1.7976991e+00  6.4660943e-01  1.9214025e-01
   6.3619745e-01 -1.8165495e-01 -1.0986711e+00 -6.8820959e-01
  -2.8724721e-01  1.0933751e-01 -4.0398455e-01 -1.0546567e+00
   5.3335589e-01 -4.3076760e-01 -5.7054073e-01  2.6373413e-01
   1.5864109e-01 -1.0974910e+00  5.8314061e-01  1.9816735e-01
  -3.0816752e-01  4.6741813e-02 -5.7657689e-02  3.3598453e-02
  -3.0340886e-01 -2.9255874e-02  1.7855248e-01 -6.1314607e-01
  -8.8487029e-01  9.9915481e-01 -3.2283807e-01  6.6241705e-01
   1.9886109e-01  8.5695285e-01  7.2396713e-01 -4.5128006e-01
   2.1435139e-01  1.7917454e-03 -1.5406340e-01  6.1317092e-01
  -7.2415709e-02 -6.3027418e-01 -3.0676842e-02  1.8962622e-02
   1.6655643e-01 -6.6773558e-01  1.2264140e-02  4.6364745e-01
  -3.3008859e-02 -5.9351635e-01 -9.8212314e-01  1.2925912e+00
   5.3860945e-01 -1.3029897e+00  8.1288010e-02  2.4840042e-01
   2.1855104e-01 -8.7877408e-02  4.1323492e-01 -5.2666110e-01
   7.8173965e-02  3.9652982e-01  7.8494281e-01 -4.7976941e-01
   7.7835989e-01 -3.9121836e-01  3.8627332e-01  3.4391499e-01
  -1.3176810e+00 -3.6041802e-01 -9.4282717e-01  7.5402415e-01
  -2.6410124e-01  3.8096124e-01 -1.1003211e+00  5.8534098e-01
  -2.9445207e-01 -2.5403774e-01 -4.1387525e-01 -4.3508843e-01
  -5.2786076e-01  8.0890602e-01  1.3762074e+00 -1.0364503e-02
   4.1360259e-02  1.0293470e-01 -5.4974329e-01  1.8155870e-01
  -1.0388508e+00  1.6341585e-01  5.3216580e-02  1.2298111e+00
  -7.1872509e-01 -3.9830491e-01  3.0341679e-01  5.0127012e-01
   5.0032634e-01  8.5301018e-01  4.6287999e-01  8.4036291e-02
   4.3147027e-02  6.3323398e+00  6.1799998e+00  6.1219482e+00
   6.9941874e+00  6.2713938e+00  6.4983115e+00  6.2659268e+00
   5.8335567e+00  5.5310593e+00  5.4185920e+00  5.3377266e+00
   6.4746714e+00  6.2922206e+00  5.9272623e+00  5.5169878e+00
   5.8470778e+00  5.5282660e+00  5.7564368e+00  6.3811421e+00
   5.9371967e+00  5.7751131e+00  6.7016945e+00  5.5942936e+00
   6.9209991e+00  7.0363517e+00  5.3017507e+00  5.8083220e+00
   6.0206056e+00  7.0023961e+00  6.2411528e+00  5.7913184e+00
   6.6039915e+00  6.4953322e+00  5.7157793e+00  5.3946385e+00
   6.7032180e+00  6.1884308e+00  5.6647019e+00  6.2957458e+00
   5.9662619e+00  6.3083024e+00  6.5133243e+00  6.1645832e+00
   5.8088012e+00  5.4862514e+00  6.4899364e+00  6.7000184e+00
   5.4968677e+00  4.6225204e+00  5.6890354e+00  5.1468072e+00
   5.6183629e+00  6.9706697e+00  6.5117574e+00  6.1664314e+00
   5.3557024e+00  5.9868355e+00  5.5618916e+00  6.3752494e+00
   1.8160205e+00  1.0977845e+00  6.5319717e-01  1.0815215e+00
   1.2830169e+00  1.1801571e+00  1.3783097e+00  1.0864580e+00
   1.9610530e+00  8.4456694e-01  7.4396110e-01  1.3705195e+00
   3.3146429e-01  6.8598080e-01  1.6690432e+00  6.9022810e-01
   6.8472540e-01  1.9080586e+00  1.0749604e+00  1.0058014e+00
   1.2235246e+00  6.6510987e-01  7.5125080e-01  9.1499406e-01
   8.7844127e-01  5.8027625e-01  1.6525590e+00  1.0678276e+00
   5.9132385e-01  4.8272288e-01  4.5549619e-01  6.8672895e-01
   5.8100200e-01  3.8789892e-01  2.3002429e+00  2.7832234e-01
   9.9855071e-01  2.0809822e+00  4.5987231e-01  7.9704660e-01
   2.0845318e+00  7.0573658e-01  1.2100005e+00  4.6577615e-01
   1.5082483e+00  6.3993782e-01  1.2280272e+00  9.0855020e-01
   1.5308671e+00  9.1555375e-01  7.6265502e-01  1.0427582e+00
   2.7245742e-01  2.0403521e+00  3.9954305e-01  1.4749188e+00
   1.3526862e+00  7.5730568e-01  8.6877489e-01  7.4521172e-01
   6.8982375e-01  1.8590535e+00  5.8405310e-01  3.2161725e-01
   1.0335408e+00  1.6373622e+00  1.2337024e+00  2.1934752e+00
   6.6258800e-01  3.7193632e-01  1.1313320e+00  5.1755816e-01
   6.6081810e-01  9.5755374e-01  2.4326854e+00  9.4885969e-01
   9.5206845e-01  3.7814987e-01  6.1724448e-01  6.0251528e-01
   1.0141426e+00  4.1018283e-01  6.5529656e-01  3.4403682e-01
   1.8287706e+00  6.2312376e-01  6.7490435e-01  1.4315168e+00
   1.2550335e+00  3.3613753e-01  6.7215997e-01  1.5243435e+00
   4.9264425e-01  1.1857699e+00  1.8090651e+00  6.2422037e-01
   1.9640617e+00  6.1644483e-01  1.0324174e+00  1.4323769e+00
   1.0905852e+00  4.4369131e-01  1.9066010e+00  7.2934306e-01
   2.9732490e-01  7.3373425e-01  1.4993441e+00  9.5568544e-01
   5.0734943e-01  1.6228292e+00  2.6680207e-01  4.7501576e-01
   2.1064097e-01  1.1305492e+00  4.6770763e-01  1.1809874e+00
   1.4951638e+00  2.8274930e-01  1.3866258e+00  1.3574516e+00
   1.7422776e+00 -4.9949260e+00 -6.9622455e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 06:13:44.782200
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.6492
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 06:13:44.786515
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    9363.1
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 06:13:44.790161
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.6882
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 06:13:44.793854
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -837.532
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140100545429968
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140099604112272
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140099604112776
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140099604113280
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140099604113784
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140099604114288

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6bcff4b0f0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.482392
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.441216
grad_step = 000002, loss = 0.408711
grad_step = 000003, loss = 0.377324
grad_step = 000004, loss = 0.343714
grad_step = 000005, loss = 0.313426
grad_step = 000006, loss = 0.294618
grad_step = 000007, loss = 0.276223
grad_step = 000008, loss = 0.253727
grad_step = 000009, loss = 0.236903
grad_step = 000010, loss = 0.230426
grad_step = 000011, loss = 0.221059
grad_step = 000012, loss = 0.206710
grad_step = 000013, loss = 0.193724
grad_step = 000014, loss = 0.184134
grad_step = 000015, loss = 0.176050
grad_step = 000016, loss = 0.167507
grad_step = 000017, loss = 0.158250
grad_step = 000018, loss = 0.149737
grad_step = 000019, loss = 0.143075
grad_step = 000020, loss = 0.137440
grad_step = 000021, loss = 0.131829
grad_step = 000022, loss = 0.125751
grad_step = 000023, loss = 0.118257
grad_step = 000024, loss = 0.110604
grad_step = 000025, loss = 0.104548
grad_step = 000026, loss = 0.099707
grad_step = 000027, loss = 0.094786
grad_step = 000028, loss = 0.089373
grad_step = 000029, loss = 0.084079
grad_step = 000030, loss = 0.079517
grad_step = 000031, loss = 0.075487
grad_step = 000032, loss = 0.071489
grad_step = 000033, loss = 0.067389
grad_step = 000034, loss = 0.063300
grad_step = 000035, loss = 0.059565
grad_step = 000036, loss = 0.056313
grad_step = 000037, loss = 0.053228
grad_step = 000038, loss = 0.050125
grad_step = 000039, loss = 0.047203
grad_step = 000040, loss = 0.044550
grad_step = 000041, loss = 0.041954
grad_step = 000042, loss = 0.039456
grad_step = 000043, loss = 0.037210
grad_step = 000044, loss = 0.035028
grad_step = 000045, loss = 0.032863
grad_step = 000046, loss = 0.030907
grad_step = 000047, loss = 0.029093
grad_step = 000048, loss = 0.027313
grad_step = 000049, loss = 0.025653
grad_step = 000050, loss = 0.024141
grad_step = 000051, loss = 0.022682
grad_step = 000052, loss = 0.021246
grad_step = 000053, loss = 0.019873
grad_step = 000054, loss = 0.018631
grad_step = 000055, loss = 0.017479
grad_step = 000056, loss = 0.016342
grad_step = 000057, loss = 0.015290
grad_step = 000058, loss = 0.014299
grad_step = 000059, loss = 0.013347
grad_step = 000060, loss = 0.012485
grad_step = 000061, loss = 0.011667
grad_step = 000062, loss = 0.010888
grad_step = 000063, loss = 0.010195
grad_step = 000064, loss = 0.009545
grad_step = 000065, loss = 0.008928
grad_step = 000066, loss = 0.008348
grad_step = 000067, loss = 0.007814
grad_step = 000068, loss = 0.007320
grad_step = 000069, loss = 0.006858
grad_step = 000070, loss = 0.006445
grad_step = 000071, loss = 0.006057
grad_step = 000072, loss = 0.005705
grad_step = 000073, loss = 0.005390
grad_step = 000074, loss = 0.005090
grad_step = 000075, loss = 0.004819
grad_step = 000076, loss = 0.004565
grad_step = 000077, loss = 0.004333
grad_step = 000078, loss = 0.004121
grad_step = 000079, loss = 0.003927
grad_step = 000080, loss = 0.003748
grad_step = 000081, loss = 0.003590
grad_step = 000082, loss = 0.003445
grad_step = 000083, loss = 0.003310
grad_step = 000084, loss = 0.003190
grad_step = 000085, loss = 0.003079
grad_step = 000086, loss = 0.002977
grad_step = 000087, loss = 0.002884
grad_step = 000088, loss = 0.002800
grad_step = 000089, loss = 0.002724
grad_step = 000090, loss = 0.002655
grad_step = 000091, loss = 0.002592
grad_step = 000092, loss = 0.002536
grad_step = 000093, loss = 0.002485
grad_step = 000094, loss = 0.002438
grad_step = 000095, loss = 0.002396
grad_step = 000096, loss = 0.002358
grad_step = 000097, loss = 0.002323
grad_step = 000098, loss = 0.002292
grad_step = 000099, loss = 0.002264
grad_step = 000100, loss = 0.002239
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002216
grad_step = 000102, loss = 0.002196
grad_step = 000103, loss = 0.002178
grad_step = 000104, loss = 0.002162
grad_step = 000105, loss = 0.002147
grad_step = 000106, loss = 0.002134
grad_step = 000107, loss = 0.002123
grad_step = 000108, loss = 0.002113
grad_step = 000109, loss = 0.002103
grad_step = 000110, loss = 0.002095
grad_step = 000111, loss = 0.002088
grad_step = 000112, loss = 0.002082
grad_step = 000113, loss = 0.002076
grad_step = 000114, loss = 0.002071
grad_step = 000115, loss = 0.002066
grad_step = 000116, loss = 0.002061
grad_step = 000117, loss = 0.002057
grad_step = 000118, loss = 0.002053
grad_step = 000119, loss = 0.002049
grad_step = 000120, loss = 0.002045
grad_step = 000121, loss = 0.002042
grad_step = 000122, loss = 0.002039
grad_step = 000123, loss = 0.002035
grad_step = 000124, loss = 0.002032
grad_step = 000125, loss = 0.002028
grad_step = 000126, loss = 0.002024
grad_step = 000127, loss = 0.002021
grad_step = 000128, loss = 0.002017
grad_step = 000129, loss = 0.002013
grad_step = 000130, loss = 0.002009
grad_step = 000131, loss = 0.002005
grad_step = 000132, loss = 0.002001
grad_step = 000133, loss = 0.001997
grad_step = 000134, loss = 0.001993
grad_step = 000135, loss = 0.001988
grad_step = 000136, loss = 0.001984
grad_step = 000137, loss = 0.001979
grad_step = 000138, loss = 0.001975
grad_step = 000139, loss = 0.001970
grad_step = 000140, loss = 0.001965
grad_step = 000141, loss = 0.001960
grad_step = 000142, loss = 0.001955
grad_step = 000143, loss = 0.001950
grad_step = 000144, loss = 0.001945
grad_step = 000145, loss = 0.001939
grad_step = 000146, loss = 0.001934
grad_step = 000147, loss = 0.001929
grad_step = 000148, loss = 0.001925
grad_step = 000149, loss = 0.001923
grad_step = 000150, loss = 0.001916
grad_step = 000151, loss = 0.001910
grad_step = 000152, loss = 0.001906
grad_step = 000153, loss = 0.001901
grad_step = 000154, loss = 0.001896
grad_step = 000155, loss = 0.001890
grad_step = 000156, loss = 0.001887
grad_step = 000157, loss = 0.001881
grad_step = 000158, loss = 0.001877
grad_step = 000159, loss = 0.001872
grad_step = 000160, loss = 0.001867
grad_step = 000161, loss = 0.001863
grad_step = 000162, loss = 0.001858
grad_step = 000163, loss = 0.001854
grad_step = 000164, loss = 0.001849
grad_step = 000165, loss = 0.001845
grad_step = 000166, loss = 0.001841
grad_step = 000167, loss = 0.001836
grad_step = 000168, loss = 0.001831
grad_step = 000169, loss = 0.001828
grad_step = 000170, loss = 0.001824
grad_step = 000171, loss = 0.001821
grad_step = 000172, loss = 0.001822
grad_step = 000173, loss = 0.001830
grad_step = 000174, loss = 0.001847
grad_step = 000175, loss = 0.001870
grad_step = 000176, loss = 0.001863
grad_step = 000177, loss = 0.001833
grad_step = 000178, loss = 0.001797
grad_step = 000179, loss = 0.001795
grad_step = 000180, loss = 0.001818
grad_step = 000181, loss = 0.001828
grad_step = 000182, loss = 0.001819
grad_step = 000183, loss = 0.001791
grad_step = 000184, loss = 0.001776
grad_step = 000185, loss = 0.001780
grad_step = 000186, loss = 0.001792
grad_step = 000187, loss = 0.001804
grad_step = 000188, loss = 0.001798
grad_step = 000189, loss = 0.001786
grad_step = 000190, loss = 0.001769
grad_step = 000191, loss = 0.001758
grad_step = 000192, loss = 0.001756
grad_step = 000193, loss = 0.001758
grad_step = 000194, loss = 0.001766
grad_step = 000195, loss = 0.001775
grad_step = 000196, loss = 0.001791
grad_step = 000197, loss = 0.001805
grad_step = 000198, loss = 0.001824
grad_step = 000199, loss = 0.001821
grad_step = 000200, loss = 0.001812
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001777
grad_step = 000202, loss = 0.001746
grad_step = 000203, loss = 0.001732
grad_step = 000204, loss = 0.001740
grad_step = 000205, loss = 0.001758
grad_step = 000206, loss = 0.001772
grad_step = 000207, loss = 0.001781
grad_step = 000208, loss = 0.001772
grad_step = 000209, loss = 0.001759
grad_step = 000210, loss = 0.001740
grad_step = 000211, loss = 0.001725
grad_step = 000212, loss = 0.001716
grad_step = 000213, loss = 0.001716
grad_step = 000214, loss = 0.001720
grad_step = 000215, loss = 0.001728
grad_step = 000216, loss = 0.001738
grad_step = 000217, loss = 0.001748
grad_step = 000218, loss = 0.001764
grad_step = 000219, loss = 0.001771
grad_step = 000220, loss = 0.001781
grad_step = 000221, loss = 0.001771
grad_step = 000222, loss = 0.001757
grad_step = 000223, loss = 0.001730
grad_step = 000224, loss = 0.001708
grad_step = 000225, loss = 0.001697
grad_step = 000226, loss = 0.001699
grad_step = 000227, loss = 0.001710
grad_step = 000228, loss = 0.001723
grad_step = 000229, loss = 0.001738
grad_step = 000230, loss = 0.001747
grad_step = 000231, loss = 0.001761
grad_step = 000232, loss = 0.001759
grad_step = 000233, loss = 0.001759
grad_step = 000234, loss = 0.001739
grad_step = 000235, loss = 0.001717
grad_step = 000236, loss = 0.001693
grad_step = 000237, loss = 0.001680
grad_step = 000238, loss = 0.001680
grad_step = 000239, loss = 0.001687
grad_step = 000240, loss = 0.001699
grad_step = 000241, loss = 0.001705
grad_step = 000242, loss = 0.001709
grad_step = 000243, loss = 0.001704
grad_step = 000244, loss = 0.001698
grad_step = 000245, loss = 0.001688
grad_step = 000246, loss = 0.001679
grad_step = 000247, loss = 0.001672
grad_step = 000248, loss = 0.001666
grad_step = 000249, loss = 0.001662
grad_step = 000250, loss = 0.001660
grad_step = 000251, loss = 0.001659
grad_step = 000252, loss = 0.001658
grad_step = 000253, loss = 0.001657
grad_step = 000254, loss = 0.001659
grad_step = 000255, loss = 0.001666
grad_step = 000256, loss = 0.001683
grad_step = 000257, loss = 0.001722
grad_step = 000258, loss = 0.001788
grad_step = 000259, loss = 0.001922
grad_step = 000260, loss = 0.002023
grad_step = 000261, loss = 0.002108
grad_step = 000262, loss = 0.001926
grad_step = 000263, loss = 0.001713
grad_step = 000264, loss = 0.001652
grad_step = 000265, loss = 0.001769
grad_step = 000266, loss = 0.001867
grad_step = 000267, loss = 0.001762
grad_step = 000268, loss = 0.001648
grad_step = 000269, loss = 0.001677
grad_step = 000270, loss = 0.001757
grad_step = 000271, loss = 0.001748
grad_step = 000272, loss = 0.001656
grad_step = 000273, loss = 0.001639
grad_step = 000274, loss = 0.001698
grad_step = 000275, loss = 0.001720
grad_step = 000276, loss = 0.001681
grad_step = 000277, loss = 0.001630
grad_step = 000278, loss = 0.001637
grad_step = 000279, loss = 0.001676
grad_step = 000280, loss = 0.001683
grad_step = 000281, loss = 0.001651
grad_step = 000282, loss = 0.001620
grad_step = 000283, loss = 0.001626
grad_step = 000284, loss = 0.001649
grad_step = 000285, loss = 0.001655
grad_step = 000286, loss = 0.001637
grad_step = 000287, loss = 0.001615
grad_step = 000288, loss = 0.001611
grad_step = 000289, loss = 0.001624
grad_step = 000290, loss = 0.001632
grad_step = 000291, loss = 0.001626
grad_step = 000292, loss = 0.001612
grad_step = 000293, loss = 0.001602
grad_step = 000294, loss = 0.001604
grad_step = 000295, loss = 0.001610
grad_step = 000296, loss = 0.001614
grad_step = 000297, loss = 0.001610
grad_step = 000298, loss = 0.001601
grad_step = 000299, loss = 0.001594
grad_step = 000300, loss = 0.001592
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001593
grad_step = 000302, loss = 0.001595
grad_step = 000303, loss = 0.001597
grad_step = 000304, loss = 0.001595
grad_step = 000305, loss = 0.001592
grad_step = 000306, loss = 0.001588
grad_step = 000307, loss = 0.001583
grad_step = 000308, loss = 0.001580
grad_step = 000309, loss = 0.001578
grad_step = 000310, loss = 0.001576
grad_step = 000311, loss = 0.001575
grad_step = 000312, loss = 0.001575
grad_step = 000313, loss = 0.001575
grad_step = 000314, loss = 0.001576
grad_step = 000315, loss = 0.001578
grad_step = 000316, loss = 0.001582
grad_step = 000317, loss = 0.001588
grad_step = 000318, loss = 0.001602
grad_step = 000319, loss = 0.001624
grad_step = 000320, loss = 0.001670
grad_step = 000321, loss = 0.001734
grad_step = 000322, loss = 0.001849
grad_step = 000323, loss = 0.001940
grad_step = 000324, loss = 0.002040
grad_step = 000325, loss = 0.001928
grad_step = 000326, loss = 0.001757
grad_step = 000327, loss = 0.001584
grad_step = 000328, loss = 0.001565
grad_step = 000329, loss = 0.001668
grad_step = 000330, loss = 0.001730
grad_step = 000331, loss = 0.001686
grad_step = 000332, loss = 0.001575
grad_step = 000333, loss = 0.001555
grad_step = 000334, loss = 0.001621
grad_step = 000335, loss = 0.001652
grad_step = 000336, loss = 0.001608
grad_step = 000337, loss = 0.001547
grad_step = 000338, loss = 0.001554
grad_step = 000339, loss = 0.001600
grad_step = 000340, loss = 0.001606
grad_step = 000341, loss = 0.001570
grad_step = 000342, loss = 0.001535
grad_step = 000343, loss = 0.001542
grad_step = 000344, loss = 0.001572
grad_step = 000345, loss = 0.001578
grad_step = 000346, loss = 0.001558
grad_step = 000347, loss = 0.001531
grad_step = 000348, loss = 0.001525
grad_step = 000349, loss = 0.001539
grad_step = 000350, loss = 0.001552
grad_step = 000351, loss = 0.001550
grad_step = 000352, loss = 0.001534
grad_step = 000353, loss = 0.001519
grad_step = 000354, loss = 0.001515
grad_step = 000355, loss = 0.001520
grad_step = 000356, loss = 0.001528
grad_step = 000357, loss = 0.001530
grad_step = 000358, loss = 0.001526
grad_step = 000359, loss = 0.001517
grad_step = 000360, loss = 0.001509
grad_step = 000361, loss = 0.001503
grad_step = 000362, loss = 0.001501
grad_step = 000363, loss = 0.001502
grad_step = 000364, loss = 0.001505
grad_step = 000365, loss = 0.001507
grad_step = 000366, loss = 0.001508
grad_step = 000367, loss = 0.001508
grad_step = 000368, loss = 0.001506
grad_step = 000369, loss = 0.001504
grad_step = 000370, loss = 0.001501
grad_step = 000371, loss = 0.001498
grad_step = 000372, loss = 0.001496
grad_step = 000373, loss = 0.001493
grad_step = 000374, loss = 0.001491
grad_step = 000375, loss = 0.001491
grad_step = 000376, loss = 0.001491
grad_step = 000377, loss = 0.001494
grad_step = 000378, loss = 0.001498
grad_step = 000379, loss = 0.001510
grad_step = 000380, loss = 0.001530
grad_step = 000381, loss = 0.001572
grad_step = 000382, loss = 0.001632
grad_step = 000383, loss = 0.001756
grad_step = 000384, loss = 0.001876
grad_step = 000385, loss = 0.002049
grad_step = 000386, loss = 0.001957
grad_step = 000387, loss = 0.001738
grad_step = 000388, loss = 0.001502
grad_step = 000389, loss = 0.001514
grad_step = 000390, loss = 0.001683
grad_step = 000391, loss = 0.001718
grad_step = 000392, loss = 0.001578
grad_step = 000393, loss = 0.001469
grad_step = 000394, loss = 0.001534
grad_step = 000395, loss = 0.001637
grad_step = 000396, loss = 0.001604
grad_step = 000397, loss = 0.001499
grad_step = 000398, loss = 0.001461
grad_step = 000399, loss = 0.001522
grad_step = 000400, loss = 0.001582
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001550
grad_step = 000402, loss = 0.001479
grad_step = 000403, loss = 0.001448
grad_step = 000404, loss = 0.001479
grad_step = 000405, loss = 0.001530
grad_step = 000406, loss = 0.001535
grad_step = 000407, loss = 0.001502
grad_step = 000408, loss = 0.001456
grad_step = 000409, loss = 0.001438
grad_step = 000410, loss = 0.001451
grad_step = 000411, loss = 0.001475
grad_step = 000412, loss = 0.001495
grad_step = 000413, loss = 0.001490
grad_step = 000414, loss = 0.001473
grad_step = 000415, loss = 0.001447
grad_step = 000416, loss = 0.001430
grad_step = 000417, loss = 0.001426
grad_step = 000418, loss = 0.001432
grad_step = 000419, loss = 0.001446
grad_step = 000420, loss = 0.001454
grad_step = 000421, loss = 0.001458
grad_step = 000422, loss = 0.001450
grad_step = 000423, loss = 0.001438
grad_step = 000424, loss = 0.001424
grad_step = 000425, loss = 0.001415
grad_step = 000426, loss = 0.001412
grad_step = 000427, loss = 0.001414
grad_step = 000428, loss = 0.001419
grad_step = 000429, loss = 0.001422
grad_step = 000430, loss = 0.001425
grad_step = 000431, loss = 0.001424
grad_step = 000432, loss = 0.001422
grad_step = 000433, loss = 0.001417
grad_step = 000434, loss = 0.001414
grad_step = 000435, loss = 0.001410
grad_step = 000436, loss = 0.001407
grad_step = 000437, loss = 0.001405
grad_step = 000438, loss = 0.001404
grad_step = 000439, loss = 0.001404
grad_step = 000440, loss = 0.001407
grad_step = 000441, loss = 0.001412
grad_step = 000442, loss = 0.001423
grad_step = 000443, loss = 0.001438
grad_step = 000444, loss = 0.001471
grad_step = 000445, loss = 0.001512
grad_step = 000446, loss = 0.001588
grad_step = 000447, loss = 0.001659
grad_step = 000448, loss = 0.001762
grad_step = 000449, loss = 0.001734
grad_step = 000450, loss = 0.001637
grad_step = 000451, loss = 0.001473
grad_step = 000452, loss = 0.001383
grad_step = 000453, loss = 0.001411
grad_step = 000454, loss = 0.001494
grad_step = 000455, loss = 0.001538
grad_step = 000456, loss = 0.001488
grad_step = 000457, loss = 0.001408
grad_step = 000458, loss = 0.001371
grad_step = 000459, loss = 0.001398
grad_step = 000460, loss = 0.001449
grad_step = 000461, loss = 0.001471
grad_step = 000462, loss = 0.001451
grad_step = 000463, loss = 0.001403
grad_step = 000464, loss = 0.001366
grad_step = 000465, loss = 0.001358
grad_step = 000466, loss = 0.001376
grad_step = 000467, loss = 0.001407
grad_step = 000468, loss = 0.001439
grad_step = 000469, loss = 0.001478
grad_step = 000470, loss = 0.001503
grad_step = 000471, loss = 0.001536
grad_step = 000472, loss = 0.001534
grad_step = 000473, loss = 0.001532
grad_step = 000474, loss = 0.001486
grad_step = 000475, loss = 0.001436
grad_step = 000476, loss = 0.001379
grad_step = 000477, loss = 0.001344
grad_step = 000478, loss = 0.001341
grad_step = 000479, loss = 0.001361
grad_step = 000480, loss = 0.001388
grad_step = 000481, loss = 0.001400
grad_step = 000482, loss = 0.001396
grad_step = 000483, loss = 0.001375
grad_step = 000484, loss = 0.001351
grad_step = 000485, loss = 0.001332
grad_step = 000486, loss = 0.001326
grad_step = 000487, loss = 0.001330
grad_step = 000488, loss = 0.001340
grad_step = 000489, loss = 0.001352
grad_step = 000490, loss = 0.001360
grad_step = 000491, loss = 0.001365
grad_step = 000492, loss = 0.001366
grad_step = 000493, loss = 0.001366
grad_step = 000494, loss = 0.001363
grad_step = 000495, loss = 0.001360
grad_step = 000496, loss = 0.001354
grad_step = 000497, loss = 0.001349
grad_step = 000498, loss = 0.001341
grad_step = 000499, loss = 0.001334
grad_step = 000500, loss = 0.001327
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001321
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

  date_run                              2020-05-10 06:14:09.221398
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.21513
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 06:14:09.228185
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.107973
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 06:14:09.236369
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13336
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 06:14:09.242406
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.640685
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
0   2020-05-10 06:13:34.814113  ...    mean_absolute_error
1   2020-05-10 06:13:34.819669  ...     mean_squared_error
2   2020-05-10 06:13:34.824210  ...  median_absolute_error
3   2020-05-10 06:13:34.828962  ...               r2_score
4   2020-05-10 06:13:44.782200  ...    mean_absolute_error
5   2020-05-10 06:13:44.786515  ...     mean_squared_error
6   2020-05-10 06:13:44.790161  ...  median_absolute_error
7   2020-05-10 06:13:44.793854  ...               r2_score
8   2020-05-10 06:14:09.221398  ...    mean_absolute_error
9   2020-05-10 06:14:09.228185  ...     mean_squared_error
10  2020-05-10 06:14:09.236369  ...  median_absolute_error
11  2020-05-10 06:14:09.242406  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3088384/9912422 [00:00<00:00, 30879010.44it/s]9920512it [00:00, 34151361.48it/s]                             
0it [00:00, ?it/s]32768it [00:00, 392756.79it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463152.48it/s]1654784it [00:00, 11318082.81it/s]                         
0it [00:00, ?it/s]8192it [00:00, 198255.95it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81d4f4128> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b9b53c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81c3c7e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b9b53da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81c4106d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81c410f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b9b55080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81c410f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7b9b55080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81c410f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81c4106d8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6b50ea91d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=490e8dc32f4994ba37946edac3e3f72719a74be7f7b3d7ad68088fa09a37c7db
  Stored in directory: /tmp/pip-ephem-wheel-cache-rnyd8l_o/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6ae8a8e080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2416640/17464789 [===>..........................] - ETA: 0s
 6782976/17464789 [==========>...................] - ETA: 0s
11517952/17464789 [==================>...........] - ETA: 0s
16228352/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 06:15:38.474201: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 06:15:38.478420: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-10 06:15:38.479048: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ac34e41720 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 06:15:38.479070: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7203 - accuracy: 0.4965
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7382 - accuracy: 0.4953 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6728 - accuracy: 0.4996
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6956 - accuracy: 0.4981
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
11000/25000 [============>.................] - ETA: 4s - loss: 7.6875 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 4s - loss: 7.6781 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6772 - accuracy: 0.4993
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
15000/25000 [=================>............] - ETA: 3s - loss: 7.6820 - accuracy: 0.4990
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6935 - accuracy: 0.4983
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6720 - accuracy: 0.4996
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6768 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6725 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 06:15:56.248029
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 06:15:56.248029  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 06:16:03.191316: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 06:16:03.197150: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-10 06:16:03.197757: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561721ec82c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 06:16:03.198066: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f27c93bcb70> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0036 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.9351 - val_crf_viterbi_accuracy: 0.3467

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f27d0b1a0b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.2680 - accuracy: 0.5260
 2000/25000 [=>............................] - ETA: 11s - loss: 7.3446 - accuracy: 0.5210
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.3906 - accuracy: 0.5180 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.3868 - accuracy: 0.5182
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5133 - accuracy: 0.5100
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5951 - accuracy: 0.5047
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6494 - accuracy: 0.5011
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6452 - accuracy: 0.5014
11000/25000 [============>.................] - ETA: 4s - loss: 7.6290 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 4s - loss: 7.6142 - accuracy: 0.5034
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5864 - accuracy: 0.5052
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6031 - accuracy: 0.5041
15000/25000 [=================>............] - ETA: 3s - loss: 7.6073 - accuracy: 0.5039
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6225 - accuracy: 0.5029
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6432 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6476 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6715 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6788 - accuracy: 0.4992
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f2760089d68> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<45:41:06, 5.24kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<32:12:34, 7.44kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<22:35:51, 10.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:49:11, 15.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:02:31, 21.6kB/s].vector_cache/glove.6B.zip:   1%|          | 7.49M/862M [00:02<7:41:48, 30.8kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:02<5:21:35, 44.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:02<3:43:59, 62.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:02<2:36:08, 89.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:02<2:00:57, 116kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:02<1:24:12, 165kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.2M/862M [00:02<58:46, 236kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:03<41:02, 336kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:03<28:39, 478kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.1M/862M [00:03<20:01, 680kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:03<14:00, 965kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:03<11:10, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<09:42, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<08:18, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:06<06:10, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:06<04:34, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<08:40, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<07:50, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:08<05:51, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<04:15, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<32:22, 411kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<24:13, 549kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:10<17:15, 768kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<12:14, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<23:16, 568kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<17:42, 746kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:12<12:40, 1.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<11:45, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<11:08, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:14<08:29, 1.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:06, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<24:49, 527kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<18:32, 705kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:16<13:15, 985kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<09:26, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:17<21:25, 607kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<17:20, 750kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:18<12:51, 1.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:18<09:17, 1.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<09:42, 1.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:19<08:11, 1.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:20<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<07:03, 1.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<06:20, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:22<04:49, 2.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:23<06:10, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:23<05:27, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:24<04:10, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:25<05:44, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:25<05:24, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:26<04:07, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:42, 2.22MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:40, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:15, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<03:56, 3.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:01, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:34, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:12, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<06:45, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:21, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<03:54, 3.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<20:07, 619kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<15:24, 808kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<11:03, 1.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<10:28, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<10:03, 1.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:37, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:29, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<08:59, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:37, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<05:39, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:40, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<07:14, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<04:08, 2.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<07:32, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<06:21, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:01, 2.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<03:38, 3.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<14:22, 842kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<12:40, 955kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<09:31, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:47, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<22:20, 539kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<16:57, 709kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<12:09, 988kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<11:07, 1.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<10:27, 1.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<07:51, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:45, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<07:04, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:15, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:41, 2.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:53, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:37, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<05:16, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<03:48, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<20:27, 575kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<15:23, 764kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<11:12, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<07:59, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<18:25, 634kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<14:07, 827kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<10:10, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<09:42, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<09:21, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<07:10, 1.62MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<05:09, 2.24MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<21:48, 530kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<16:28, 701kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<11:49, 975kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<10:49, 1.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<09:59, 1.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:36, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<05:27, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<21:48, 523kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<16:17, 700kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<11:38, 978kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<08:17, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<18:28, 614kB/s] .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<15:20, 739kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<11:14, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:02, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<09:29, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<07:51, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:47, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<06:32, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<05:36, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:09, 2.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<03:06, 3.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<14:27, 770kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<11:19, 982kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<08:12, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<08:12, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<08:04, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:09, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:26, 2.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<08:40, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<07:14, 1.52MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:18, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<06:11, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<06:39, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<05:13, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:46, 2.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<28:35, 380kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<21:11, 512kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<15:02, 720kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<12:53, 837kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<11:17, 955kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:23, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:01, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<07:55, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<06:42, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<05:50, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<06:20, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:33, 2.98MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<21:58, 482kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<16:30, 641kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<11:45, 897kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<10:36, 991kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<09:45, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<07:18, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<07:14, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<06:00, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:28, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:26, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<06:01, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:41, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:24, 3.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<07:43, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<06:28, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:44, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<05:40, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:45, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:27, 2.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<07:16, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<06:09, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:33, 2.22MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<05:30, 1.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:55, 2.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<03:42, 2.72MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<04:53, 2.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<04:30, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<03:25, 2.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<04:35, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<05:19, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<04:10, 2.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:01, 3.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<09:31, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<07:45, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:38, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:03, 2.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<3:20:03, 49.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<2:22:10, 69.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<1:39:51, 98.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:09:46, 140kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<53:16, 183kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<38:10, 255kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<26:51, 362kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<18:55, 512kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<24:05, 402kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<18:56, 511kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<13:42, 706kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:40, 995kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<11:54, 807kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<09:21, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:47, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<06:55, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<06:58, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<05:18, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<03:50, 2.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:54, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<05:49, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:16, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:09, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<03:26, 2.72MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:27, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:03<04:08, 2.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<03:06, 2.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<04:13, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:05<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<03:56, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<02:53, 3.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:20, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<04:42, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<03:32, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:31, 2.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:09<05:05, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<03:58, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<02:54, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:40, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<04:56, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<03:39, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:32, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:13<05:07, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:03, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<02:56, 3.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<15:43, 567kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<11:59, 743kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<08:36, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<08:01, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<07:34, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<05:42, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:05, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<07:07, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:56, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:20, 2.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:57, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:18, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<04:06, 2.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<02:59, 2.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:28, 1.58MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:45, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:32, 2.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:47, 1.79MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<03:44, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<02:45, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:54, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:21, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:14, 2.62MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:06, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:35, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:34, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<02:40, 3.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:16, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:55, 2.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<03:53, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:28, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:30, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<02:34, 3.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<05:06, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:28, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:18, 2.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:07, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:36, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:34, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:38, 3.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:26, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<03:59, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<02:58, 2.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:50, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:33, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:41, 2.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:38, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<04:13, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:23, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:28, 3.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<13:11, 597kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<10:05, 781kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:13, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<06:46, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:34, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:06, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:34, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:46, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:40, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:42, 2.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<04:19, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:51, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<02:53, 2.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:42, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:24, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:35, 2.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:28, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<04:01, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:19, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<12:53, 579kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<09:40, 772kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:56, 1.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<06:32, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<06:13, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:41, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:22, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<05:33, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<04:33, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:22, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<03:57, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<04:23, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:24, 2.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:28, 2.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<04:39, 1.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<04:00, 1.79MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:58, 2.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:43, 1.92MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:22, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:30, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:18, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<02:59, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<02:11, 3.19MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:50, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:34, 2.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<03:19, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<03:51, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<03:01, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:11, 3.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<05:33, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:35, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:22, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:55, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<04:10, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:13, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:21, 2.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<04:04, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:34, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:39, 2.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:18, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<02:57, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:08, 3.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<11:34, 567kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<08:49, 744kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:19, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<05:53, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<05:30, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<04:11, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:59, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<22:30, 286kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<16:26, 391kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<11:36, 551kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<09:30, 669kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<07:56, 801kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<05:52, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:08, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<12:50, 490kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<09:40, 649kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<06:53, 908kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<06:10, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<05:38, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:12, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:01, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:39, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<03:55, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:53, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:22, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:38, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:48, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:02, 2.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<04:06, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:30, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:35, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:09, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:31, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:44, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:00, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:13, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:53, 2.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<02:08, 2.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:47, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:10, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:31, 2.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:50, 3.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<10:19, 555kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<07:50, 730kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<05:37, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<05:10, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<04:48, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<03:37, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:35, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<04:33, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:47, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:47, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<03:12, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:30, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<01:51, 2.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:46, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<03:05, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:26, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<01:44, 3.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<07:18, 737kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<05:41, 945kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:05, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:01, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:59, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<03:01, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:12, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:00, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:40, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:00, 2.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:32, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:50, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<02:12, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:36, 3.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<04:22, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:35, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:38, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:59, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:09, 1.60MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:06<02:25, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:45, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:55, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:17, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<02:25, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:46, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:26, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<01:50, 2.66MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:24, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:41, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:06, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<01:33, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:32, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:13, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:40, 2.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:13, 3.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<05:30, 853kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<04:51, 967kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:36, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:34, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<03:36, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:01, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:14, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:33, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:45, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:08, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:32, 2.91MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<02:58, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:34, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:56, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:24, 3.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<07:24, 597kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<05:19, 827kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:46, 1.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<04:13, 1.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:52, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:54, 1.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:05, 2.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<03:01, 1.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:34, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:53, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:14, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:25, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:54, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:21, 3.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<22:45, 182kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<16:20, 253kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<11:28, 358kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<08:52, 459kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<07:04, 576kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:07, 793kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:36, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<04:06, 974kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<03:13, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:21, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:41, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<09:52, 399kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<07:46, 506kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<05:36, 699kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:57, 984kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<04:10, 927kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<03:20, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:25, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:31, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:36, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:59, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:25, 2.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<02:51, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:24, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:45, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:01, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:40, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:11, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<05:14, 684kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<04:02, 888kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:53, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<02:48, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<02:19, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:42, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:58, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:45, 1.96MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:18, 2.61MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:39, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:53, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:30, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:04, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<05:49, 568kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<04:25, 748kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:09, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<02:55, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<02:23, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:43, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<01:56, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:41, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:15, 2.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:33, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:47, 1.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:24, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:00, 3.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<05:29, 553kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<04:06, 738kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:55, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:04, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<04:38, 639kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<03:54, 760kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<02:51, 1.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:00, 1.45MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<02:55, 993kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<02:18, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:40, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<01:47, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:49, 1.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:23, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:00, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<01:47, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:32, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:08, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:23, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:32, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:11, 2.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<00:52, 3.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:26, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:17, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<00:57, 2.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:13, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:08, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<00:50, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:07, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:18, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:01, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:45, 3.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:15, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:07, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:50, 2.85MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:06, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:16, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<01:00, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:43, 3.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<07:21, 310kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<05:22, 422kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<03:46, 595kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:06, 711kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<02:24, 915kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:43, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:39, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:36, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:13, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:51, 2.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:10, 955kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:44, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:14, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:18, 1.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:19, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:00, 1.96MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:43, 2.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:25, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:11, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:52, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:01, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:08, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:52, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:37, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:15, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:04, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:47, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:55, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:00, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:46, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:33, 2.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:57, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:50, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:36, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:46, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:54, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:41, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:30, 3.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:55, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<00:48, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:36, 2.48MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:43, 1.98MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:39, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:29, 2.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:44, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:35, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:25, 3.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<02:15, 578kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:41, 770kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:11, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:05, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:01, 1.20MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:46, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:32, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<03:54, 300kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<02:50, 410kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:58, 576kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<01:35, 696kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<01:20, 822kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:58, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:40, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:48, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:40, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:32, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:27, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:19, 2.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:14, 3.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<03:30, 255kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<02:38, 338kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:51, 472kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:16, 667kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<01:11, 689kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:54, 906kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:38, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:35, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:34, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:26, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:17, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<01:15, 546kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:56, 720kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:39, 1.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:34, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:31, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:23, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:15, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<01:21, 405kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:59, 545kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:40, 764kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:32, 878kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:29, 981kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:21, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:13, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:46, 529kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:34, 705kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:23, 981kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:19, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:17, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:10, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:09, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:41, 196kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:28, 272kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:16, 384kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:07, 490kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:06, 609kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:03, 832kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.17MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 740/400000 [00:00<00:53, 7396.73it/s]  0%|          | 1474/400000 [00:00<00:54, 7379.22it/s]  1%|          | 2214/400000 [00:00<00:53, 7384.00it/s]  1%|          | 2934/400000 [00:00<00:54, 7325.36it/s]  1%|          | 3670/400000 [00:00<00:54, 7334.23it/s]  1%|          | 4392/400000 [00:00<00:54, 7299.34it/s]  1%|▏         | 5129/400000 [00:00<00:53, 7318.05it/s]  1%|▏         | 5870/400000 [00:00<00:53, 7344.46it/s]  2%|▏         | 6620/400000 [00:00<00:53, 7390.17it/s]  2%|▏         | 7354/400000 [00:01<00:53, 7374.15it/s]  2%|▏         | 8076/400000 [00:01<00:53, 7326.18it/s]  2%|▏         | 8804/400000 [00:01<00:53, 7310.54it/s]  2%|▏         | 9525/400000 [00:01<00:53, 7270.13it/s]  3%|▎         | 10252/400000 [00:01<00:53, 7268.24it/s]  3%|▎         | 10976/400000 [00:01<00:53, 7257.80it/s]  3%|▎         | 11738/400000 [00:01<00:52, 7362.76it/s]  3%|▎         | 12487/400000 [00:01<00:52, 7399.24it/s]  3%|▎         | 13240/400000 [00:01<00:52, 7434.71it/s]  3%|▎         | 13988/400000 [00:01<00:51, 7446.58it/s]  4%|▎         | 14733/400000 [00:02<00:51, 7430.41it/s]  4%|▍         | 15489/400000 [00:02<00:51, 7468.71it/s]  4%|▍         | 16239/400000 [00:02<00:51, 7476.86it/s]  4%|▍         | 17001/400000 [00:02<00:50, 7516.79it/s]  4%|▍         | 17756/400000 [00:02<00:50, 7525.49it/s]  5%|▍         | 18509/400000 [00:02<00:50, 7481.44it/s]  5%|▍         | 19268/400000 [00:02<00:50, 7511.21it/s]  5%|▌         | 20020/400000 [00:02<00:50, 7506.26it/s]  5%|▌         | 20783/400000 [00:02<00:50, 7542.53it/s]  5%|▌         | 21545/400000 [00:02<00:50, 7565.11it/s]  6%|▌         | 22302/400000 [00:03<00:50, 7453.67it/s]  6%|▌         | 23048/400000 [00:03<00:50, 7429.96it/s]  6%|▌         | 23792/400000 [00:03<00:50, 7425.45it/s]  6%|▌         | 24535/400000 [00:03<00:51, 7356.50it/s]  6%|▋         | 25271/400000 [00:03<00:51, 7321.36it/s]  7%|▋         | 26004/400000 [00:03<00:51, 7308.26it/s]  7%|▋         | 26736/400000 [00:03<00:51, 7301.77it/s]  7%|▋         | 27467/400000 [00:03<00:51, 7301.96it/s]  7%|▋         | 28198/400000 [00:03<00:51, 7256.38it/s]  7%|▋         | 28924/400000 [00:03<00:51, 7231.31it/s]  7%|▋         | 29648/400000 [00:04<00:51, 7197.28it/s]  8%|▊         | 30376/400000 [00:04<00:51, 7219.96it/s]  8%|▊         | 31122/400000 [00:04<00:50, 7287.34it/s]  8%|▊         | 31869/400000 [00:04<00:50, 7337.48it/s]  8%|▊         | 32603/400000 [00:04<00:50, 7332.51it/s]  8%|▊         | 33337/400000 [00:04<00:50, 7283.27it/s]  9%|▊         | 34079/400000 [00:04<00:49, 7322.88it/s]  9%|▊         | 34812/400000 [00:04<00:49, 7306.04it/s]  9%|▉         | 35557/400000 [00:04<00:49, 7346.19it/s]  9%|▉         | 36309/400000 [00:04<00:49, 7394.81it/s]  9%|▉         | 37049/400000 [00:05<00:49, 7321.19it/s]  9%|▉         | 37782/400000 [00:05<00:49, 7313.44it/s] 10%|▉         | 38514/400000 [00:05<00:49, 7291.14it/s] 10%|▉         | 39244/400000 [00:05<00:49, 7259.77it/s] 10%|▉         | 39971/400000 [00:05<00:49, 7235.20it/s] 10%|█         | 40695/400000 [00:05<00:49, 7210.88it/s] 10%|█         | 41431/400000 [00:05<00:49, 7252.60it/s] 11%|█         | 42158/400000 [00:05<00:49, 7255.67it/s] 11%|█         | 42908/400000 [00:05<00:48, 7325.46it/s] 11%|█         | 43660/400000 [00:05<00:48, 7380.73it/s] 11%|█         | 44399/400000 [00:06<00:48, 7333.06it/s] 11%|█▏        | 45147/400000 [00:06<00:48, 7375.98it/s] 11%|█▏        | 45885/400000 [00:06<00:48, 7372.51it/s] 12%|█▏        | 46628/400000 [00:06<00:47, 7388.05it/s] 12%|█▏        | 47369/400000 [00:06<00:47, 7391.42it/s] 12%|█▏        | 48109/400000 [00:06<00:48, 7319.94it/s] 12%|█▏        | 48842/400000 [00:06<00:48, 7309.85it/s] 12%|█▏        | 49574/400000 [00:06<00:48, 7287.06it/s] 13%|█▎        | 50317/400000 [00:06<00:47, 7326.62it/s] 13%|█▎        | 51050/400000 [00:06<00:47, 7305.86it/s] 13%|█▎        | 51781/400000 [00:07<00:48, 7237.86it/s] 13%|█▎        | 52507/400000 [00:07<00:47, 7242.91it/s] 13%|█▎        | 53232/400000 [00:07<00:48, 7197.17it/s] 13%|█▎        | 53969/400000 [00:07<00:47, 7248.15it/s] 14%|█▎        | 54702/400000 [00:07<00:47, 7271.56it/s] 14%|█▍        | 55430/400000 [00:07<00:47, 7248.05it/s] 14%|█▍        | 56175/400000 [00:07<00:47, 7305.20it/s] 14%|█▍        | 56906/400000 [00:07<00:47, 7269.51it/s] 14%|█▍        | 57653/400000 [00:07<00:46, 7328.11it/s] 15%|█▍        | 58387/400000 [00:07<00:46, 7329.15it/s] 15%|█▍        | 59121/400000 [00:08<00:46, 7309.28it/s] 15%|█▍        | 59862/400000 [00:08<00:46, 7337.33it/s] 15%|█▌        | 60597/400000 [00:08<00:46, 7339.41it/s] 15%|█▌        | 61351/400000 [00:08<00:45, 7396.62it/s] 16%|█▌        | 62105/400000 [00:08<00:45, 7436.30it/s] 16%|█▌        | 62849/400000 [00:08<00:45, 7406.41it/s] 16%|█▌        | 63590/400000 [00:08<00:45, 7389.21it/s] 16%|█▌        | 64330/400000 [00:08<00:45, 7367.49it/s] 16%|█▋        | 65071/400000 [00:08<00:45, 7379.88it/s] 16%|█▋        | 65810/400000 [00:08<00:45, 7373.27it/s] 17%|█▋        | 66548/400000 [00:09<00:45, 7302.67it/s] 17%|█▋        | 67288/400000 [00:09<00:45, 7329.79it/s] 17%|█▋        | 68022/400000 [00:09<00:45, 7330.67it/s] 17%|█▋        | 68756/400000 [00:09<00:45, 7277.41it/s] 17%|█▋        | 69494/400000 [00:09<00:45, 7305.47it/s] 18%|█▊        | 70225/400000 [00:09<00:45, 7232.27it/s] 18%|█▊        | 70955/400000 [00:09<00:45, 7251.56it/s] 18%|█▊        | 71681/400000 [00:09<00:45, 7242.72it/s] 18%|█▊        | 72406/400000 [00:09<00:45, 7239.73it/s] 18%|█▊        | 73131/400000 [00:09<00:45, 7232.55it/s] 18%|█▊        | 73855/400000 [00:10<00:45, 7123.79it/s] 19%|█▊        | 74582/400000 [00:10<00:45, 7166.39it/s] 19%|█▉        | 75314/400000 [00:10<00:45, 7209.39it/s] 19%|█▉        | 76056/400000 [00:10<00:44, 7269.34it/s] 19%|█▉        | 76793/400000 [00:10<00:44, 7296.73it/s] 19%|█▉        | 77523/400000 [00:10<00:44, 7271.60it/s] 20%|█▉        | 78266/400000 [00:10<00:43, 7316.79it/s] 20%|█▉        | 78998/400000 [00:10<00:44, 7256.37it/s] 20%|█▉        | 79733/400000 [00:10<00:43, 7281.45it/s] 20%|██        | 80462/400000 [00:10<00:43, 7278.67it/s] 20%|██        | 81191/400000 [00:11<00:44, 7226.27it/s] 20%|██        | 81920/400000 [00:11<00:43, 7242.80it/s] 21%|██        | 82645/400000 [00:11<00:44, 7144.35it/s] 21%|██        | 83361/400000 [00:11<00:44, 7147.74it/s] 21%|██        | 84115/400000 [00:11<00:43, 7259.79it/s] 21%|██        | 84842/400000 [00:11<00:44, 7090.76it/s] 21%|██▏       | 85577/400000 [00:11<00:43, 7165.04it/s] 22%|██▏       | 86334/400000 [00:11<00:43, 7279.35it/s] 22%|██▏       | 87080/400000 [00:11<00:42, 7332.60it/s] 22%|██▏       | 87827/400000 [00:11<00:42, 7372.35it/s] 22%|██▏       | 88565/400000 [00:12<00:42, 7322.01it/s] 22%|██▏       | 89314/400000 [00:12<00:42, 7371.15it/s] 23%|██▎       | 90052/400000 [00:12<00:42, 7252.54it/s] 23%|██▎       | 90779/400000 [00:12<00:43, 7172.66it/s] 23%|██▎       | 91523/400000 [00:12<00:42, 7248.66it/s] 23%|██▎       | 92249/400000 [00:12<00:43, 7082.93it/s] 23%|██▎       | 92988/400000 [00:12<00:42, 7171.16it/s] 23%|██▎       | 93727/400000 [00:12<00:42, 7234.51it/s] 24%|██▎       | 94475/400000 [00:12<00:41, 7305.54it/s] 24%|██▍       | 95207/400000 [00:13<00:41, 7285.22it/s] 24%|██▍       | 95937/400000 [00:13<00:42, 7220.08it/s] 24%|██▍       | 96678/400000 [00:13<00:41, 7274.86it/s] 24%|██▍       | 97407/400000 [00:13<00:41, 7277.92it/s] 25%|██▍       | 98144/400000 [00:13<00:41, 7303.91it/s] 25%|██▍       | 98888/400000 [00:13<00:41, 7341.76it/s] 25%|██▍       | 99639/400000 [00:13<00:40, 7390.33it/s] 25%|██▌       | 100388/400000 [00:13<00:40, 7417.86it/s] 25%|██▌       | 101149/400000 [00:13<00:39, 7472.04it/s] 25%|██▌       | 101917/400000 [00:13<00:39, 7532.34it/s] 26%|██▌       | 102675/400000 [00:14<00:39, 7543.77it/s] 26%|██▌       | 103430/400000 [00:14<00:39, 7497.99it/s] 26%|██▌       | 104181/400000 [00:14<00:39, 7492.39it/s] 26%|██▌       | 104931/400000 [00:14<00:39, 7467.58it/s] 26%|██▋       | 105678/400000 [00:14<00:39, 7432.99it/s] 27%|██▋       | 106426/400000 [00:14<00:39, 7444.26it/s] 27%|██▋       | 107173/400000 [00:14<00:39, 7450.45it/s] 27%|██▋       | 107919/400000 [00:14<00:39, 7425.96it/s] 27%|██▋       | 108662/400000 [00:14<00:39, 7378.82it/s] 27%|██▋       | 109400/400000 [00:14<00:39, 7326.24it/s] 28%|██▊       | 110137/400000 [00:15<00:39, 7337.60it/s] 28%|██▊       | 110874/400000 [00:15<00:39, 7343.18it/s] 28%|██▊       | 111609/400000 [00:15<00:39, 7317.89it/s] 28%|██▊       | 112341/400000 [00:15<00:39, 7236.04it/s] 28%|██▊       | 113080/400000 [00:15<00:39, 7277.77it/s] 28%|██▊       | 113817/400000 [00:15<00:39, 7304.51it/s] 29%|██▊       | 114562/400000 [00:15<00:38, 7346.78it/s] 29%|██▉       | 115297/400000 [00:15<00:39, 7207.28it/s] 29%|██▉       | 116035/400000 [00:15<00:39, 7256.67it/s] 29%|██▉       | 116762/400000 [00:15<00:39, 7114.34it/s] 29%|██▉       | 117486/400000 [00:16<00:39, 7150.54it/s] 30%|██▉       | 118227/400000 [00:16<00:39, 7223.67it/s] 30%|██▉       | 118957/400000 [00:16<00:38, 7244.40it/s] 30%|██▉       | 119690/400000 [00:16<00:38, 7268.47it/s] 30%|███       | 120420/400000 [00:16<00:38, 7275.66it/s] 30%|███       | 121155/400000 [00:16<00:38, 7294.11it/s] 30%|███       | 121889/400000 [00:16<00:38, 7305.29it/s] 31%|███       | 122620/400000 [00:16<00:38, 7293.62it/s] 31%|███       | 123362/400000 [00:16<00:37, 7329.68it/s] 31%|███       | 124110/400000 [00:16<00:37, 7371.31it/s] 31%|███       | 124848/400000 [00:17<00:37, 7335.82it/s] 31%|███▏      | 125582/400000 [00:17<00:37, 7299.57it/s] 32%|███▏      | 126313/400000 [00:17<00:37, 7225.50it/s] 32%|███▏      | 127040/400000 [00:17<00:37, 7235.61it/s] 32%|███▏      | 127772/400000 [00:17<00:37, 7257.73it/s] 32%|███▏      | 128516/400000 [00:17<00:37, 7310.08it/s] 32%|███▏      | 129248/400000 [00:17<00:37, 7310.64it/s] 32%|███▏      | 129980/400000 [00:17<00:37, 7271.60it/s] 33%|███▎      | 130708/400000 [00:17<00:37, 7095.75it/s] 33%|███▎      | 131465/400000 [00:17<00:37, 7231.65it/s] 33%|███▎      | 132190/400000 [00:18<00:37, 7226.41it/s] 33%|███▎      | 132925/400000 [00:18<00:36, 7261.77it/s] 33%|███▎      | 133653/400000 [00:18<00:36, 7267.18it/s] 34%|███▎      | 134404/400000 [00:18<00:36, 7338.17it/s] 34%|███▍      | 135139/400000 [00:18<00:36, 7325.58it/s] 34%|███▍      | 135881/400000 [00:18<00:35, 7351.30it/s] 34%|███▍      | 136630/400000 [00:18<00:35, 7389.47it/s] 34%|███▍      | 137370/400000 [00:18<00:35, 7370.03it/s] 35%|███▍      | 138120/400000 [00:18<00:35, 7408.17it/s] 35%|███▍      | 138861/400000 [00:18<00:35, 7408.57it/s] 35%|███▍      | 139602/400000 [00:19<00:35, 7400.45it/s] 35%|███▌      | 140343/400000 [00:19<00:35, 7383.45it/s] 35%|███▌      | 141082/400000 [00:19<00:35, 7364.45it/s] 35%|███▌      | 141819/400000 [00:19<00:35, 7352.70it/s] 36%|███▌      | 142555/400000 [00:19<00:35, 7341.95it/s] 36%|███▌      | 143299/400000 [00:19<00:34, 7368.85it/s] 36%|███▌      | 144042/400000 [00:19<00:34, 7383.74it/s] 36%|███▌      | 144781/400000 [00:19<00:34, 7347.39it/s] 36%|███▋      | 145522/400000 [00:19<00:34, 7364.53it/s] 37%|███▋      | 146271/400000 [00:19<00:34, 7399.14it/s] 37%|███▋      | 147011/400000 [00:20<00:34, 7340.81it/s] 37%|███▋      | 147754/400000 [00:20<00:34, 7366.55it/s] 37%|███▋      | 148491/400000 [00:20<00:34, 7360.37it/s] 37%|███▋      | 149228/400000 [00:20<00:34, 7354.17it/s] 37%|███▋      | 149964/400000 [00:20<00:34, 7345.66it/s] 38%|███▊      | 150702/400000 [00:20<00:33, 7353.88it/s] 38%|███▊      | 151457/400000 [00:20<00:33, 7411.41it/s] 38%|███▊      | 152199/400000 [00:20<00:33, 7401.78it/s] 38%|███▊      | 152942/400000 [00:20<00:33, 7408.90it/s] 38%|███▊      | 153683/400000 [00:20<00:33, 7285.11it/s] 39%|███▊      | 154418/400000 [00:21<00:33, 7304.22it/s] 39%|███▉      | 155153/400000 [00:21<00:33, 7316.26it/s] 39%|███▉      | 155885/400000 [00:21<00:33, 7303.09it/s] 39%|███▉      | 156616/400000 [00:21<00:33, 7242.84it/s] 39%|███▉      | 157341/400000 [00:21<00:33, 7143.19it/s] 40%|███▉      | 158078/400000 [00:21<00:33, 7205.71it/s] 40%|███▉      | 158825/400000 [00:21<00:33, 7281.61it/s] 40%|███▉      | 159554/400000 [00:21<00:33, 7242.14it/s] 40%|████      | 160298/400000 [00:21<00:32, 7298.85it/s] 40%|████      | 161029/400000 [00:22<00:33, 7218.51it/s] 40%|████      | 161760/400000 [00:22<00:32, 7244.70it/s] 41%|████      | 162503/400000 [00:22<00:32, 7296.85it/s] 41%|████      | 163244/400000 [00:22<00:32, 7329.37it/s] 41%|████      | 163978/400000 [00:22<00:32, 7316.08it/s] 41%|████      | 164710/400000 [00:22<00:32, 7252.11it/s] 41%|████▏     | 165436/400000 [00:22<00:32, 7248.90it/s] 42%|████▏     | 166173/400000 [00:22<00:32, 7281.40it/s] 42%|████▏     | 166918/400000 [00:22<00:31, 7330.28it/s] 42%|████▏     | 167674/400000 [00:22<00:31, 7395.80it/s] 42%|████▏     | 168430/400000 [00:23<00:31, 7442.45it/s] 42%|████▏     | 169183/400000 [00:23<00:30, 7466.60it/s] 42%|████▏     | 169930/400000 [00:23<00:30, 7447.00it/s] 43%|████▎     | 170675/400000 [00:23<00:30, 7423.34it/s] 43%|████▎     | 171425/400000 [00:23<00:30, 7444.21it/s] 43%|████▎     | 172170/400000 [00:23<00:30, 7441.32it/s] 43%|████▎     | 172915/400000 [00:23<00:30, 7377.10it/s] 43%|████▎     | 173653/400000 [00:23<00:30, 7342.77it/s] 44%|████▎     | 174388/400000 [00:23<00:30, 7302.56it/s] 44%|████▍     | 175144/400000 [00:23<00:30, 7377.84it/s] 44%|████▍     | 175907/400000 [00:24<00:30, 7450.41it/s] 44%|████▍     | 176656/400000 [00:24<00:29, 7460.62it/s] 44%|████▍     | 177403/400000 [00:24<00:29, 7457.64it/s] 45%|████▍     | 178149/400000 [00:24<00:29, 7427.09it/s] 45%|████▍     | 178903/400000 [00:24<00:29, 7459.34it/s] 45%|████▍     | 179650/400000 [00:24<00:29, 7403.41it/s] 45%|████▌     | 180391/400000 [00:24<00:29, 7396.52it/s] 45%|████▌     | 181131/400000 [00:24<00:29, 7339.39it/s] 45%|████▌     | 181866/400000 [00:24<00:30, 7097.73it/s] 46%|████▌     | 182594/400000 [00:24<00:30, 7150.44it/s] 46%|████▌     | 183339/400000 [00:25<00:29, 7236.64it/s] 46%|████▌     | 184079/400000 [00:25<00:29, 7283.76it/s] 46%|████▌     | 184823/400000 [00:25<00:29, 7327.38it/s] 46%|████▋     | 185558/400000 [00:25<00:29, 7331.98it/s] 47%|████▋     | 186297/400000 [00:25<00:29, 7347.26it/s] 47%|████▋     | 187055/400000 [00:25<00:28, 7413.09it/s] 47%|████▋     | 187797/400000 [00:25<00:28, 7366.98it/s] 47%|████▋     | 188536/400000 [00:25<00:28, 7371.51it/s] 47%|████▋     | 189274/400000 [00:25<00:28, 7323.00it/s] 48%|████▊     | 190021/400000 [00:25<00:28, 7365.62it/s] 48%|████▊     | 190790/400000 [00:26<00:28, 7457.29it/s] 48%|████▊     | 191544/400000 [00:26<00:27, 7480.74it/s] 48%|████▊     | 192293/400000 [00:26<00:27, 7475.89it/s] 48%|████▊     | 193041/400000 [00:26<00:27, 7439.39it/s] 48%|████▊     | 193789/400000 [00:26<00:27, 7450.68it/s] 49%|████▊     | 194535/400000 [00:26<00:27, 7432.88it/s] 49%|████▉     | 195279/400000 [00:26<00:27, 7401.26it/s] 49%|████▉     | 196020/400000 [00:26<00:27, 7350.32it/s] 49%|████▉     | 196756/400000 [00:26<00:27, 7310.66it/s] 49%|████▉     | 197510/400000 [00:26<00:27, 7376.38it/s] 50%|████▉     | 198262/400000 [00:27<00:27, 7416.23it/s] 50%|████▉     | 199010/400000 [00:27<00:27, 7435.15it/s] 50%|████▉     | 199754/400000 [00:27<00:26, 7432.55it/s] 50%|█████     | 200504/400000 [00:27<00:26, 7450.83it/s] 50%|█████     | 201250/400000 [00:27<00:26, 7433.56it/s] 50%|█████     | 201998/400000 [00:27<00:26, 7445.48it/s] 51%|█████     | 202743/400000 [00:27<00:26, 7430.42it/s] 51%|█████     | 203487/400000 [00:27<00:26, 7398.51it/s] 51%|█████     | 204228/400000 [00:27<00:26, 7399.18it/s] 51%|█████     | 204968/400000 [00:27<00:26, 7364.15it/s] 51%|█████▏    | 205716/400000 [00:28<00:26, 7398.14it/s] 52%|█████▏    | 206456/400000 [00:28<00:26, 7394.84it/s] 52%|█████▏    | 207199/400000 [00:28<00:26, 7404.03it/s] 52%|█████▏    | 207951/400000 [00:28<00:25, 7435.64it/s] 52%|█████▏    | 208705/400000 [00:28<00:25, 7466.49it/s] 52%|█████▏    | 209452/400000 [00:28<00:25, 7441.99it/s] 53%|█████▎    | 210197/400000 [00:28<00:25, 7431.69it/s] 53%|█████▎    | 210941/400000 [00:28<00:25, 7414.23it/s] 53%|█████▎    | 211683/400000 [00:28<00:25, 7380.44it/s] 53%|█████▎    | 212422/400000 [00:28<00:25, 7361.08it/s] 53%|█████▎    | 213189/400000 [00:29<00:25, 7450.84it/s] 53%|█████▎    | 213939/400000 [00:29<00:24, 7464.42it/s] 54%|█████▎    | 214689/400000 [00:29<00:24, 7472.63it/s] 54%|█████▍    | 215437/400000 [00:29<00:24, 7450.79it/s] 54%|█████▍    | 216183/400000 [00:29<00:24, 7399.41it/s] 54%|█████▍    | 216939/400000 [00:29<00:24, 7444.30it/s] 54%|█████▍    | 217684/400000 [00:29<00:24, 7420.39it/s] 55%|█████▍    | 218434/400000 [00:29<00:24, 7442.03it/s] 55%|█████▍    | 219180/400000 [00:29<00:24, 7445.02it/s] 55%|█████▍    | 219925/400000 [00:29<00:24, 7437.07it/s] 55%|█████▌    | 220691/400000 [00:30<00:23, 7501.00it/s] 55%|█████▌    | 221442/400000 [00:30<00:23, 7455.99it/s] 56%|█████▌    | 222188/400000 [00:30<00:23, 7424.64it/s] 56%|█████▌    | 222931/400000 [00:30<00:23, 7414.72it/s] 56%|█████▌    | 223673/400000 [00:30<00:23, 7356.36it/s] 56%|█████▌    | 224435/400000 [00:30<00:23, 7430.92it/s] 56%|█████▋    | 225191/400000 [00:30<00:23, 7468.23it/s] 56%|█████▋    | 225946/400000 [00:30<00:23, 7491.86it/s] 57%|█████▋    | 226706/400000 [00:30<00:23, 7522.49it/s] 57%|█████▋    | 227459/400000 [00:30<00:22, 7508.17it/s] 57%|█████▋    | 228233/400000 [00:31<00:22, 7574.35it/s] 57%|█████▋    | 228999/400000 [00:31<00:22, 7597.53it/s] 57%|█████▋    | 229759/400000 [00:31<00:22, 7569.35it/s] 58%|█████▊    | 230517/400000 [00:31<00:22, 7525.64it/s] 58%|█████▊    | 231270/400000 [00:31<00:22, 7451.46it/s] 58%|█████▊    | 232024/400000 [00:31<00:22, 7476.05it/s] 58%|█████▊    | 232772/400000 [00:31<00:22, 7476.12it/s] 58%|█████▊    | 233520/400000 [00:31<00:22, 7382.31it/s] 59%|█████▊    | 234259/400000 [00:31<00:22, 7351.72it/s] 59%|█████▊    | 234995/400000 [00:31<00:22, 7326.74it/s] 59%|█████▉    | 235740/400000 [00:32<00:22, 7360.98it/s] 59%|█████▉    | 236477/400000 [00:32<00:22, 7291.01it/s] 59%|█████▉    | 237207/400000 [00:32<00:22, 7272.02it/s] 59%|█████▉    | 237935/400000 [00:32<00:22, 7248.82it/s] 60%|█████▉    | 238661/400000 [00:32<00:22, 7160.30it/s] 60%|█████▉    | 239380/400000 [00:32<00:22, 7167.86it/s] 60%|██████    | 240098/400000 [00:32<00:22, 7152.16it/s] 60%|██████    | 240834/400000 [00:32<00:22, 7213.05it/s] 60%|██████    | 241565/400000 [00:32<00:21, 7241.46it/s] 61%|██████    | 242295/400000 [00:33<00:21, 7256.43it/s] 61%|██████    | 243047/400000 [00:33<00:21, 7329.68it/s] 61%|██████    | 243781/400000 [00:33<00:21, 7330.56it/s] 61%|██████    | 244527/400000 [00:33<00:21, 7367.51it/s] 61%|██████▏   | 245264/400000 [00:33<00:21, 7366.56it/s] 62%|██████▏   | 246001/400000 [00:33<00:21, 7319.08it/s] 62%|██████▏   | 246755/400000 [00:33<00:20, 7382.84it/s] 62%|██████▏   | 247507/400000 [00:33<00:20, 7421.42it/s] 62%|██████▏   | 248250/400000 [00:33<00:20, 7419.51it/s] 62%|██████▏   | 249003/400000 [00:33<00:20, 7450.12it/s] 62%|██████▏   | 249749/400000 [00:34<00:20, 7425.41it/s] 63%|██████▎   | 250495/400000 [00:34<00:20, 7435.35it/s] 63%|██████▎   | 251239/400000 [00:34<00:20, 7422.13it/s] 63%|██████▎   | 251982/400000 [00:34<00:20, 7334.32it/s] 63%|██████▎   | 252736/400000 [00:34<00:19, 7394.67it/s] 63%|██████▎   | 253476/400000 [00:34<00:19, 7385.88it/s] 64%|██████▎   | 254224/400000 [00:34<00:19, 7409.63it/s] 64%|██████▎   | 254979/400000 [00:34<00:19, 7449.09it/s] 64%|██████▍   | 255725/400000 [00:34<00:19, 7411.26it/s] 64%|██████▍   | 256467/400000 [00:34<00:19, 7364.91it/s] 64%|██████▍   | 257204/400000 [00:35<00:19, 7354.81it/s] 64%|██████▍   | 257949/400000 [00:35<00:19, 7380.33it/s] 65%|██████▍   | 258691/400000 [00:35<00:19, 7390.32it/s] 65%|██████▍   | 259431/400000 [00:35<00:19, 7388.46it/s] 65%|██████▌   | 260170/400000 [00:35<00:18, 7368.20it/s] 65%|██████▌   | 260907/400000 [00:35<00:18, 7364.55it/s] 65%|██████▌   | 261644/400000 [00:35<00:18, 7344.59it/s] 66%|██████▌   | 262379/400000 [00:35<00:18, 7303.92it/s] 66%|██████▌   | 263131/400000 [00:35<00:18, 7366.10it/s] 66%|██████▌   | 263869/400000 [00:35<00:18, 7367.67it/s] 66%|██████▌   | 264606/400000 [00:36<00:18, 7325.74it/s] 66%|██████▋   | 265361/400000 [00:36<00:18, 7387.81it/s] 67%|██████▋   | 266112/400000 [00:36<00:18, 7421.28it/s] 67%|██████▋   | 266855/400000 [00:36<00:18, 7371.57it/s] 67%|██████▋   | 267593/400000 [00:36<00:17, 7364.49it/s] 67%|██████▋   | 268330/400000 [00:36<00:18, 7168.94it/s] 67%|██████▋   | 269049/400000 [00:36<00:18, 7144.45it/s] 67%|██████▋   | 269765/400000 [00:36<00:18, 7011.69it/s] 68%|██████▊   | 270507/400000 [00:36<00:18, 7128.56it/s] 68%|██████▊   | 271239/400000 [00:36<00:17, 7183.15it/s] 68%|██████▊   | 271987/400000 [00:37<00:17, 7269.40it/s] 68%|██████▊   | 272745/400000 [00:37<00:17, 7359.42it/s] 68%|██████▊   | 273488/400000 [00:37<00:17, 7380.44it/s] 69%|██████▊   | 274227/400000 [00:37<00:17, 7243.32it/s] 69%|██████▊   | 274954/400000 [00:37<00:17, 7250.37it/s] 69%|██████▉   | 275680/400000 [00:37<00:17, 7241.67it/s] 69%|██████▉   | 276413/400000 [00:37<00:17, 7266.59it/s] 69%|██████▉   | 277162/400000 [00:37<00:16, 7329.33it/s] 69%|██████▉   | 277913/400000 [00:37<00:16, 7379.95it/s] 70%|██████▉   | 278654/400000 [00:37<00:16, 7387.09it/s] 70%|██████▉   | 279393/400000 [00:38<00:16, 7350.20it/s] 70%|███████   | 280143/400000 [00:38<00:16, 7394.32it/s] 70%|███████   | 280884/400000 [00:38<00:16, 7398.27it/s] 70%|███████   | 281632/400000 [00:38<00:15, 7420.64it/s] 71%|███████   | 282375/400000 [00:38<00:15, 7414.29it/s] 71%|███████   | 283117/400000 [00:38<00:15, 7360.08it/s] 71%|███████   | 283868/400000 [00:38<00:15, 7401.85it/s] 71%|███████   | 284609/400000 [00:38<00:15, 7388.51it/s] 71%|███████▏  | 285348/400000 [00:38<00:15, 7376.41it/s] 72%|███████▏  | 286087/400000 [00:38<00:15, 7378.90it/s] 72%|███████▏  | 286825/400000 [00:39<00:15, 7340.96it/s] 72%|███████▏  | 287591/400000 [00:39<00:15, 7433.51it/s] 72%|███████▏  | 288343/400000 [00:39<00:14, 7458.62it/s] 72%|███████▏  | 289090/400000 [00:39<00:14, 7401.74it/s] 72%|███████▏  | 289832/400000 [00:39<00:14, 7406.22it/s] 73%|███████▎  | 290573/400000 [00:39<00:14, 7364.50it/s] 73%|███████▎  | 291310/400000 [00:39<00:14, 7323.91it/s] 73%|███████▎  | 292053/400000 [00:39<00:14, 7352.97it/s] 73%|███████▎  | 292795/400000 [00:39<00:14, 7371.56it/s] 73%|███████▎  | 293533/400000 [00:39<00:14, 7371.74it/s] 74%|███████▎  | 294271/400000 [00:40<00:14, 7329.66it/s] 74%|███████▍  | 295005/400000 [00:40<00:14, 7332.43it/s] 74%|███████▍  | 295755/400000 [00:40<00:14, 7381.50it/s] 74%|███████▍  | 296501/400000 [00:40<00:13, 7404.23it/s] 74%|███████▍  | 297257/400000 [00:40<00:13, 7448.53it/s] 75%|███████▍  | 298002/400000 [00:40<00:13, 7368.26it/s] 75%|███████▍  | 298743/400000 [00:40<00:13, 7378.17it/s] 75%|███████▍  | 299490/400000 [00:40<00:13, 7405.14it/s] 75%|███████▌  | 300242/400000 [00:40<00:13, 7436.97it/s] 75%|███████▌  | 300986/400000 [00:40<00:13, 7424.57it/s] 75%|███████▌  | 301729/400000 [00:41<00:13, 7385.58it/s] 76%|███████▌  | 302468/400000 [00:41<00:13, 7354.44it/s] 76%|███████▌  | 303204/400000 [00:41<00:13, 7321.20it/s] 76%|███████▌  | 303940/400000 [00:41<00:13, 7332.78it/s] 76%|███████▌  | 304692/400000 [00:41<00:12, 7386.22it/s] 76%|███████▋  | 305432/400000 [00:41<00:12, 7388.00it/s] 77%|███████▋  | 306171/400000 [00:41<00:12, 7352.44it/s] 77%|███████▋  | 306907/400000 [00:41<00:12, 7341.21it/s] 77%|███████▋  | 307658/400000 [00:41<00:12, 7390.01it/s] 77%|███████▋  | 308404/400000 [00:41<00:12, 7409.31it/s] 77%|███████▋  | 309161/400000 [00:42<00:12, 7454.55it/s] 77%|███████▋  | 309907/400000 [00:42<00:12, 7416.82it/s] 78%|███████▊  | 310652/400000 [00:42<00:12, 7424.41it/s] 78%|███████▊  | 311395/400000 [00:42<00:11, 7406.07it/s] 78%|███████▊  | 312136/400000 [00:42<00:11, 7395.03it/s] 78%|███████▊  | 312876/400000 [00:42<00:11, 7320.89it/s] 78%|███████▊  | 313609/400000 [00:42<00:11, 7240.41it/s] 79%|███████▊  | 314349/400000 [00:42<00:11, 7285.74it/s] 79%|███████▉  | 315078/400000 [00:42<00:11, 7213.18it/s] 79%|███████▉  | 315808/400000 [00:42<00:11, 7238.96it/s] 79%|███████▉  | 316540/400000 [00:43<00:11, 7262.49it/s] 79%|███████▉  | 317269/400000 [00:43<00:11, 7269.56it/s] 79%|███████▉  | 317997/400000 [00:43<00:11, 7062.31it/s] 80%|███████▉  | 318705/400000 [00:43<00:11, 7026.61it/s] 80%|███████▉  | 319434/400000 [00:43<00:11, 7101.92it/s] 80%|████████  | 320159/400000 [00:43<00:11, 7144.64it/s] 80%|████████  | 320880/400000 [00:43<00:11, 7163.15it/s] 80%|████████  | 321635/400000 [00:43<00:10, 7274.95it/s] 81%|████████  | 322381/400000 [00:43<00:10, 7329.02it/s] 81%|████████  | 323117/400000 [00:44<00:10, 7337.03it/s] 81%|████████  | 323852/400000 [00:44<00:10, 7302.97it/s] 81%|████████  | 324583/400000 [00:44<00:10, 7239.32it/s] 81%|████████▏ | 325334/400000 [00:44<00:10, 7317.58it/s] 82%|████████▏ | 326071/400000 [00:44<00:10, 7332.32it/s] 82%|████████▏ | 326813/400000 [00:44<00:09, 7356.66it/s] 82%|████████▏ | 327549/400000 [00:44<00:09, 7337.14it/s] 82%|████████▏ | 328283/400000 [00:44<00:09, 7279.82it/s] 82%|████████▏ | 329018/400000 [00:44<00:09, 7300.55it/s] 82%|████████▏ | 329753/400000 [00:44<00:09, 7313.45it/s] 83%|████████▎ | 330485/400000 [00:45<00:09, 7303.34it/s] 83%|████████▎ | 331216/400000 [00:45<00:09, 7300.46it/s] 83%|████████▎ | 331947/400000 [00:45<00:09, 7283.28it/s] 83%|████████▎ | 332685/400000 [00:45<00:09, 7311.49it/s] 83%|████████▎ | 333417/400000 [00:45<00:09, 7299.57it/s] 84%|████████▎ | 334148/400000 [00:45<00:09, 7285.79it/s] 84%|████████▎ | 334879/400000 [00:45<00:08, 7290.42it/s] 84%|████████▍ | 335609/400000 [00:45<00:08, 7264.22it/s] 84%|████████▍ | 336336/400000 [00:45<00:08, 7241.72it/s] 84%|████████▍ | 337061/400000 [00:45<00:08, 7239.22it/s] 84%|████████▍ | 337796/400000 [00:46<00:08, 7269.82it/s] 85%|████████▍ | 338544/400000 [00:46<00:08, 7331.59it/s] 85%|████████▍ | 339287/400000 [00:46<00:08, 7358.51it/s] 85%|████████▌ | 340048/400000 [00:46<00:08, 7430.58it/s] 85%|████████▌ | 340792/400000 [00:46<00:08, 7338.60it/s] 85%|████████▌ | 341527/400000 [00:46<00:07, 7336.99it/s] 86%|████████▌ | 342262/400000 [00:46<00:07, 7332.69it/s] 86%|████████▌ | 342996/400000 [00:46<00:07, 7216.11it/s] 86%|████████▌ | 343719/400000 [00:46<00:07, 7192.98it/s] 86%|████████▌ | 344439/400000 [00:46<00:07, 7176.19it/s] 86%|████████▋ | 345174/400000 [00:47<00:07, 7226.45it/s] 86%|████████▋ | 345918/400000 [00:47<00:07, 7288.43it/s] 87%|████████▋ | 346648/400000 [00:47<00:07, 7238.20it/s] 87%|████████▋ | 347379/400000 [00:47<00:07, 7258.22it/s] 87%|████████▋ | 348109/400000 [00:47<00:07, 7268.21it/s] 87%|████████▋ | 348847/400000 [00:47<00:07, 7299.27it/s] 87%|████████▋ | 349586/400000 [00:47<00:06, 7324.47it/s] 88%|████████▊ | 350319/400000 [00:47<00:06, 7304.22it/s] 88%|████████▊ | 351055/400000 [00:47<00:06, 7318.63it/s] 88%|████████▊ | 351787/400000 [00:47<00:06, 7290.13it/s] 88%|████████▊ | 352517/400000 [00:48<00:06, 7269.75it/s] 88%|████████▊ | 353258/400000 [00:48<00:06, 7309.56it/s] 88%|████████▊ | 353993/400000 [00:48<00:06, 7321.47it/s] 89%|████████▊ | 354726/400000 [00:48<00:06, 7280.41it/s] 89%|████████▉ | 355456/400000 [00:48<00:06, 7284.38it/s] 89%|████████▉ | 356211/400000 [00:48<00:05, 7361.77it/s] 89%|████████▉ | 356956/400000 [00:48<00:05, 7386.57it/s] 89%|████████▉ | 357707/400000 [00:48<00:05, 7421.48it/s] 90%|████████▉ | 358450/400000 [00:48<00:05, 7392.51it/s] 90%|████████▉ | 359190/400000 [00:48<00:05, 7379.03it/s] 90%|████████▉ | 359946/400000 [00:49<00:05, 7429.90it/s] 90%|█████████ | 360702/400000 [00:49<00:05, 7466.71it/s] 90%|█████████ | 361459/400000 [00:49<00:05, 7497.04it/s] 91%|█████████ | 362209/400000 [00:49<00:05, 7403.67it/s] 91%|█████████ | 362963/400000 [00:49<00:04, 7443.02it/s] 91%|█████████ | 363717/400000 [00:49<00:04, 7470.31it/s] 91%|█████████ | 364465/400000 [00:49<00:04, 7468.98it/s] 91%|█████████▏| 365213/400000 [00:49<00:04, 7446.40it/s] 91%|█████████▏| 365958/400000 [00:49<00:04, 7350.19it/s] 92%|█████████▏| 366702/400000 [00:49<00:04, 7375.41it/s] 92%|█████████▏| 367440/400000 [00:50<00:04, 7317.44it/s] 92%|█████████▏| 368184/400000 [00:50<00:04, 7351.63it/s] 92%|█████████▏| 368941/400000 [00:50<00:04, 7413.02it/s] 92%|█████████▏| 369683/400000 [00:50<00:04, 7406.45it/s] 93%|█████████▎| 370432/400000 [00:50<00:03, 7428.75it/s] 93%|█████████▎| 371176/400000 [00:50<00:03, 7394.72it/s] 93%|█████████▎| 371916/400000 [00:50<00:03, 7385.05it/s] 93%|█████████▎| 372657/400000 [00:50<00:03, 7391.02it/s] 93%|█████████▎| 373397/400000 [00:50<00:03, 7313.07it/s] 94%|█████████▎| 374129/400000 [00:50<00:03, 7294.18it/s] 94%|█████████▎| 374859/400000 [00:51<00:03, 7284.08it/s] 94%|█████████▍| 375615/400000 [00:51<00:03, 7364.39it/s] 94%|█████████▍| 376358/400000 [00:51<00:03, 7383.15it/s] 94%|█████████▍| 377097/400000 [00:51<00:03, 7346.60it/s] 94%|█████████▍| 377832/400000 [00:51<00:03, 7336.47it/s] 95%|█████████▍| 378575/400000 [00:51<00:02, 7363.19it/s] 95%|█████████▍| 379312/400000 [00:51<00:02, 7355.98it/s] 95%|█████████▌| 380062/400000 [00:51<00:02, 7396.13it/s] 95%|█████████▌| 380802/400000 [00:51<00:02, 7314.86it/s] 95%|█████████▌| 381553/400000 [00:51<00:02, 7372.06it/s] 96%|█████████▌| 382302/400000 [00:52<00:02, 7406.97it/s] 96%|█████████▌| 383047/400000 [00:52<00:02, 7417.44it/s] 96%|█████████▌| 383802/400000 [00:52<00:02, 7454.16it/s] 96%|█████████▌| 384548/400000 [00:52<00:02, 7429.74it/s] 96%|█████████▋| 385292/400000 [00:52<00:01, 7422.79it/s] 97%|█████████▋| 386046/400000 [00:52<00:01, 7456.34it/s] 97%|█████████▋| 386792/400000 [00:52<00:01, 7432.40it/s] 97%|█████████▋| 387547/400000 [00:52<00:01, 7466.16it/s] 97%|█████████▋| 388294/400000 [00:52<00:01, 7412.85it/s] 97%|█████████▋| 389036/400000 [00:52<00:01, 7369.18it/s] 97%|█████████▋| 389774/400000 [00:53<00:01, 7360.72it/s] 98%|█████████▊| 390519/400000 [00:53<00:01, 7385.29it/s] 98%|█████████▊| 391262/400000 [00:53<00:01, 7396.95it/s] 98%|█████████▊| 392002/400000 [00:53<00:01, 7341.78it/s] 98%|█████████▊| 392737/400000 [00:53<00:00, 7333.80it/s] 98%|█████████▊| 393478/400000 [00:53<00:00, 7354.09it/s] 99%|█████████▊| 394222/400000 [00:53<00:00, 7377.86it/s] 99%|█████████▊| 394960/400000 [00:53<00:00, 7345.75it/s] 99%|█████████▉| 395695/400000 [00:53<00:00, 7277.63it/s] 99%|█████████▉| 396441/400000 [00:53<00:00, 7330.08it/s] 99%|█████████▉| 397175/400000 [00:54<00:00, 7211.84it/s] 99%|█████████▉| 397921/400000 [00:54<00:00, 7283.37it/s]100%|█████████▉| 398657/400000 [00:54<00:00, 7304.57it/s]100%|█████████▉| 399392/400000 [00:54<00:00, 7317.77it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7342.71it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f276d781dd8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011322997926130826 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.011469961807480623 	 Accuracy: 48

  model saves at 48% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15744 out of table with 15485 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15744 out of table with 15485 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

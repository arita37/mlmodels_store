
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7db47c7470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 07:11:55.771423
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 07:11:55.775068
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 07:11:55.777436
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 07:11:55.780229
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7dacb17438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355997.2188
Epoch 2/10

1/1 [==============================] - 0s 90ms/step - loss: 268841.9688
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 175335.1562
Epoch 4/10

1/1 [==============================] - 0s 86ms/step - loss: 95982.1875
Epoch 5/10

1/1 [==============================] - 0s 91ms/step - loss: 50820.3984
Epoch 6/10

1/1 [==============================] - 0s 81ms/step - loss: 28473.5410
Epoch 7/10

1/1 [==============================] - 0s 83ms/step - loss: 17267.2793
Epoch 8/10

1/1 [==============================] - 0s 87ms/step - loss: 11289.0938
Epoch 9/10

1/1 [==============================] - 0s 88ms/step - loss: 7925.0571
Epoch 10/10

1/1 [==============================] - 0s 89ms/step - loss: 5856.5200

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.93372178e-02  6.39035478e-02  1.33336568e+00 -1.34547722e+00
   1.92856729e-01  1.03353429e+00 -1.74807167e+00 -5.83928943e-01
   6.72385216e-01 -8.21029484e-01  1.10116458e+00 -2.29707107e-01
   4.91682410e-01  1.76181364e+00  5.05894065e-01  5.87940454e-01
   2.01993322e+00  2.77572870e-02 -2.01291847e+00 -3.86048943e-01
  -1.03159571e+00 -3.77673209e-01 -6.85135007e-01  6.97015285e-01
  -4.26409900e-01  2.27911890e-01 -7.16944218e-01 -2.36182642e+00
   2.33420563e+00  1.76695919e+00 -4.08486247e-01  6.51853204e-01
   5.05670846e-01  1.04576397e+00 -1.45892811e+00  4.96707320e-01
   1.74978554e-01 -1.21069121e+00  7.56412745e-04 -4.80016947e-01
  -6.23665869e-01  8.44831169e-02  3.54050547e-01  2.57001579e-01
  -1.10321701e+00 -9.57049489e-01  1.75932974e-01  3.71631444e-01
   8.37701976e-01 -6.34972811e-01 -1.44290698e+00  8.33405972e-01
  -2.74798959e-01 -4.10403982e-02  7.03103065e-01 -7.64875293e-01
   1.30307063e-01  2.32962221e-01  1.95733190e-01 -5.00935435e-01
   4.26697940e-01  1.00465651e+01  9.25915241e+00  8.90788174e+00
   9.63543606e+00  8.60517216e+00  1.03488283e+01  9.42594910e+00
   8.95230865e+00  9.12351131e+00  9.16764736e+00  7.29715776e+00
   9.25663567e+00  7.24623871e+00  6.34801865e+00  8.24915409e+00
   7.44110298e+00  9.74246025e+00  9.78294563e+00  8.49349308e+00
   9.17244434e+00  9.45356750e+00  8.60208321e+00  8.45018101e+00
   8.12363434e+00  9.38463116e+00  9.22287750e+00  7.60691023e+00
   9.13401604e+00  8.48246479e+00  1.04135828e+01  8.86125660e+00
   8.00486088e+00  8.29040051e+00  9.26902771e+00  1.02008400e+01
   9.13684273e+00  9.73699760e+00  9.70573330e+00  9.02858925e+00
   9.10920334e+00  9.85012722e+00  6.90806484e+00  7.61996412e+00
   7.86314201e+00  8.89185715e+00  8.42407131e+00  9.81364250e+00
   7.81423759e+00  7.99472809e+00  9.85791302e+00  1.02978411e+01
   8.50008106e+00  7.55133820e+00  8.14278698e+00  9.92330456e+00
   9.08840275e+00  9.16586685e+00  9.04939079e+00  8.85830212e+00
  -6.87149391e-02  1.16470861e+00  9.73762870e-01  7.40525901e-01
   1.86358070e+00 -1.93119907e+00  1.36255276e+00  4.39242899e-01
   8.61174226e-01  3.68421912e-01  2.71391869e-02 -9.95238781e-01
  -8.71770501e-01 -1.18142247e+00 -9.63650346e-02  3.59530568e-01
   9.53534782e-01  1.08112204e+00  1.56467223e+00 -1.34995222e+00
  -4.69675720e-01 -1.47189617e+00 -5.91054201e-01  3.69163454e-01
   2.25065410e-01  4.02874738e-01  4.90745634e-01  1.20438886e+00
  -1.41512036e+00  1.90644598e+00 -1.35593724e+00  4.90899295e-01
   4.36171055e-01  6.95449114e-01  1.14998114e+00 -1.51634738e-01
   3.98306489e-01 -1.54887557e+00  6.77579522e-01  6.42794549e-01
   6.65973127e-02  1.83831382e+00 -8.94615471e-01  7.11979270e-01
  -2.46573830e+00 -8.34814489e-01 -1.52894688e+00 -4.95629370e-01
  -5.74586272e-01 -1.62305760e+00 -1.00986826e+00 -1.94804162e-01
  -1.97880983e-01 -8.13398510e-03 -1.67261648e+00  6.36557221e-01
  -4.85934854e-01  1.98653400e-01 -1.02728975e+00  1.74868965e+00
   6.21139407e-01  1.82694864e+00  2.47177780e-01  2.61916542e+00
   1.50576401e+00  2.78130198e+00  1.52405858e+00  4.33144927e-01
   8.07653725e-01  9.45015728e-01  1.72576642e+00  5.79455614e-01
   8.83311689e-01  3.89911592e-01  3.68527174e-01  1.70460153e+00
   1.01317227e+00  1.11918378e+00  6.36449158e-01  1.44403720e+00
   5.62721491e-01  1.20090961e+00  8.05537105e-01  1.77391911e+00
   4.75195348e-01  6.11079335e-01  8.95994306e-01  9.95437860e-01
   1.69640183e-01  1.68449044e+00  2.05738664e+00  8.99824321e-01
   1.93671048e+00  2.52216768e+00  1.24935496e+00  2.28484333e-01
   5.30349314e-01  3.21836376e+00  2.43856490e-01  1.83541489e+00
   1.79664373e-01  6.36893630e-01  1.28426588e+00  4.58822608e-01
   2.80551016e-01  7.36729503e-01  9.75503385e-01  2.33510065e+00
   1.62873423e+00  1.88179755e+00  3.26437354e-01  5.60054958e-01
   2.25159287e-01  9.07893658e-01  1.50751233e+00  9.86217856e-01
   6.50499582e-01  2.16039300e-01  1.13669538e+00  1.79241371e+00
   5.34060597e-02  8.76502228e+00  9.71226120e+00  8.62801552e+00
   8.47793674e+00  9.35165691e+00  9.59500217e+00  9.40928364e+00
   9.63539600e+00  9.19243431e+00  9.60698795e+00  9.95801926e+00
   8.53074265e+00  9.61289310e+00  1.01744118e+01  9.78454971e+00
   8.51503849e+00  9.22613621e+00  9.48552799e+00  9.60210800e+00
   9.28737831e+00  9.12804794e+00  9.51996040e+00  9.25274467e+00
   8.42739201e+00  1.08833828e+01  7.26832867e+00  9.16740894e+00
   8.88291931e+00  8.79655552e+00  7.91926718e+00  8.80965805e+00
   1.01612310e+01  9.70858002e+00  1.01130495e+01  7.61775351e+00
   7.74300718e+00  8.46563530e+00  8.81640911e+00  9.20794868e+00
   9.88513756e+00  9.19489574e+00  8.16898823e+00  8.62942505e+00
   9.18831921e+00  9.29681206e+00  8.43529129e+00  8.71649551e+00
   1.11021557e+01  7.53710890e+00  9.74201202e+00  7.72770071e+00
   8.47098255e+00  9.03395367e+00  1.01676054e+01  9.54357338e+00
   8.50547123e+00  8.33404827e+00  7.79900932e+00  9.19806004e+00
   1.71622229e+00  1.60679507e+00  3.14516687e+00  1.62492585e+00
   2.16319418e+00  7.43206620e-01  4.41884696e-01  3.38272750e-01
   1.33228147e+00  9.39616382e-01  7.35206783e-01  2.63469696e-01
   5.35526693e-01  1.39358783e+00  1.04183841e+00  1.33436203e-01
   1.61164415e+00  2.24488974e+00  8.92525077e-01  2.28802109e+00
   8.96910727e-01  1.01700246e-01  1.45017540e+00  1.41091061e+00
   4.40548718e-01  4.16896820e-01  1.83928227e+00  1.02840567e+00
   1.45626926e+00  1.64707077e+00  4.02522445e-01  1.39141381e-01
   6.01548433e-01  1.26246107e+00  7.25588560e-01  7.17873752e-01
   2.81202722e+00  3.35708523e+00  3.45183849e-01  1.75599635e-01
   2.62502646e+00  2.37253428e-01  6.31071448e-01  2.38204598e-01
   2.60479331e+00  1.77891874e+00  2.59607613e-01  1.34884357e+00
   1.90210497e+00  8.90267909e-01  7.51446187e-01  2.41311491e-01
   7.17667162e-01  3.34163129e-01  2.34384656e-01  1.73232734e-01
   2.14131594e-01  1.75242233e+00  6.53145552e-01  1.38681066e+00
  -5.56260681e+00  1.05036936e+01 -8.45722294e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 07:12:03.487761
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.0218
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 07:12:03.492319
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8861.34
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 07:12:03.495629
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    93.743
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 07:12:03.498612
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -792.596
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140177203999352
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140174674228112
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140174674228616
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140174674229120
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140174674229624
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140174674230128

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7da82290f0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.657120
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.616560
grad_step = 000002, loss = 0.592991
grad_step = 000003, loss = 0.571189
grad_step = 000004, loss = 0.546352
grad_step = 000005, loss = 0.517829
grad_step = 000006, loss = 0.486165
grad_step = 000007, loss = 0.454079
grad_step = 000008, loss = 0.426537
grad_step = 000009, loss = 0.413880
grad_step = 000010, loss = 0.406686
grad_step = 000011, loss = 0.388576
grad_step = 000012, loss = 0.366491
grad_step = 000013, loss = 0.348207
grad_step = 000014, loss = 0.335502
grad_step = 000015, loss = 0.326691
grad_step = 000016, loss = 0.318464
grad_step = 000017, loss = 0.307335
grad_step = 000018, loss = 0.293998
grad_step = 000019, loss = 0.280306
grad_step = 000020, loss = 0.267306
grad_step = 000021, loss = 0.256043
grad_step = 000022, loss = 0.246234
grad_step = 000023, loss = 0.236624
grad_step = 000024, loss = 0.226532
grad_step = 000025, loss = 0.216066
grad_step = 000026, loss = 0.205596
grad_step = 000027, loss = 0.195617
grad_step = 000028, loss = 0.186105
grad_step = 000029, loss = 0.176902
grad_step = 000030, loss = 0.168258
grad_step = 000031, loss = 0.160201
grad_step = 000032, loss = 0.152187
grad_step = 000033, loss = 0.144114
grad_step = 000034, loss = 0.136339
grad_step = 000035, loss = 0.128938
grad_step = 000036, loss = 0.121976
grad_step = 000037, loss = 0.115414
grad_step = 000038, loss = 0.108866
grad_step = 000039, loss = 0.102294
grad_step = 000040, loss = 0.096088
grad_step = 000041, loss = 0.090276
grad_step = 000042, loss = 0.084728
grad_step = 000043, loss = 0.079519
grad_step = 000044, loss = 0.074555
grad_step = 000045, loss = 0.069847
grad_step = 000046, loss = 0.065445
grad_step = 000047, loss = 0.061158
grad_step = 000048, loss = 0.056980
grad_step = 000049, loss = 0.053073
grad_step = 000050, loss = 0.049421
grad_step = 000051, loss = 0.045990
grad_step = 000052, loss = 0.042751
grad_step = 000053, loss = 0.039628
grad_step = 000054, loss = 0.036738
grad_step = 000055, loss = 0.034073
grad_step = 000056, loss = 0.031507
grad_step = 000057, loss = 0.029098
grad_step = 000058, loss = 0.026862
grad_step = 000059, loss = 0.024783
grad_step = 000060, loss = 0.022836
grad_step = 000061, loss = 0.020984
grad_step = 000062, loss = 0.019284
grad_step = 000063, loss = 0.017733
grad_step = 000064, loss = 0.016265
grad_step = 000065, loss = 0.014906
grad_step = 000066, loss = 0.013665
grad_step = 000067, loss = 0.012529
grad_step = 000068, loss = 0.011463
grad_step = 000069, loss = 0.010485
grad_step = 000070, loss = 0.009607
grad_step = 000071, loss = 0.008799
grad_step = 000072, loss = 0.008059
grad_step = 000073, loss = 0.007397
grad_step = 000074, loss = 0.006802
grad_step = 000075, loss = 0.006262
grad_step = 000076, loss = 0.005777
grad_step = 000077, loss = 0.005354
grad_step = 000078, loss = 0.004975
grad_step = 000079, loss = 0.004637
grad_step = 000080, loss = 0.004340
grad_step = 000081, loss = 0.004075
grad_step = 000082, loss = 0.003840
grad_step = 000083, loss = 0.003631
grad_step = 000084, loss = 0.003451
grad_step = 000085, loss = 0.003291
grad_step = 000086, loss = 0.003149
grad_step = 000087, loss = 0.003024
grad_step = 000088, loss = 0.002913
grad_step = 000089, loss = 0.002812
grad_step = 000090, loss = 0.002722
grad_step = 000091, loss = 0.002644
grad_step = 000092, loss = 0.002571
grad_step = 000093, loss = 0.002507
grad_step = 000094, loss = 0.002450
grad_step = 000095, loss = 0.002399
grad_step = 000096, loss = 0.002352
grad_step = 000097, loss = 0.002310
grad_step = 000098, loss = 0.002271
grad_step = 000099, loss = 0.002236
grad_step = 000100, loss = 0.002205
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002178
grad_step = 000102, loss = 0.002152
grad_step = 000103, loss = 0.002130
grad_step = 000104, loss = 0.002109
grad_step = 000105, loss = 0.002091
grad_step = 000106, loss = 0.002075
grad_step = 000107, loss = 0.002060
grad_step = 000108, loss = 0.002047
grad_step = 000109, loss = 0.002035
grad_step = 000110, loss = 0.002024
grad_step = 000111, loss = 0.002015
grad_step = 000112, loss = 0.002006
grad_step = 000113, loss = 0.001999
grad_step = 000114, loss = 0.001992
grad_step = 000115, loss = 0.001985
grad_step = 000116, loss = 0.001980
grad_step = 000117, loss = 0.001974
grad_step = 000118, loss = 0.001969
grad_step = 000119, loss = 0.001964
grad_step = 000120, loss = 0.001959
grad_step = 000121, loss = 0.001955
grad_step = 000122, loss = 0.001950
grad_step = 000123, loss = 0.001946
grad_step = 000124, loss = 0.001941
grad_step = 000125, loss = 0.001937
grad_step = 000126, loss = 0.001932
grad_step = 000127, loss = 0.001928
grad_step = 000128, loss = 0.001924
grad_step = 000129, loss = 0.001920
grad_step = 000130, loss = 0.001916
grad_step = 000131, loss = 0.001913
grad_step = 000132, loss = 0.001914
grad_step = 000133, loss = 0.001919
grad_step = 000134, loss = 0.001927
grad_step = 000135, loss = 0.001925
grad_step = 000136, loss = 0.001912
grad_step = 000137, loss = 0.001891
grad_step = 000138, loss = 0.001880
grad_step = 000139, loss = 0.001883
grad_step = 000140, loss = 0.001888
grad_step = 000141, loss = 0.001887
grad_step = 000142, loss = 0.001873
grad_step = 000143, loss = 0.001860
grad_step = 000144, loss = 0.001855
grad_step = 000145, loss = 0.001857
grad_step = 000146, loss = 0.001858
grad_step = 000147, loss = 0.001853
grad_step = 000148, loss = 0.001843
grad_step = 000149, loss = 0.001834
grad_step = 000150, loss = 0.001829
grad_step = 000151, loss = 0.001828
grad_step = 000152, loss = 0.001827
grad_step = 000153, loss = 0.001824
grad_step = 000154, loss = 0.001818
grad_step = 000155, loss = 0.001811
grad_step = 000156, loss = 0.001803
grad_step = 000157, loss = 0.001797
grad_step = 000158, loss = 0.001793
grad_step = 000159, loss = 0.001790
grad_step = 000160, loss = 0.001787
grad_step = 000161, loss = 0.001786
grad_step = 000162, loss = 0.001787
grad_step = 000163, loss = 0.001789
grad_step = 000164, loss = 0.001793
grad_step = 000165, loss = 0.001796
grad_step = 000166, loss = 0.001797
grad_step = 000167, loss = 0.001790
grad_step = 000168, loss = 0.001777
grad_step = 000169, loss = 0.001757
grad_step = 000170, loss = 0.001740
grad_step = 000171, loss = 0.001731
grad_step = 000172, loss = 0.001728
grad_step = 000173, loss = 0.001730
grad_step = 000174, loss = 0.001733
grad_step = 000175, loss = 0.001736
grad_step = 000176, loss = 0.001737
grad_step = 000177, loss = 0.001736
grad_step = 000178, loss = 0.001728
grad_step = 000179, loss = 0.001717
grad_step = 000180, loss = 0.001702
grad_step = 000181, loss = 0.001688
grad_step = 000182, loss = 0.001677
grad_step = 000183, loss = 0.001671
grad_step = 000184, loss = 0.001671
grad_step = 000185, loss = 0.001679
grad_step = 000186, loss = 0.001692
grad_step = 000187, loss = 0.001699
grad_step = 000188, loss = 0.001689
grad_step = 000189, loss = 0.001662
grad_step = 000190, loss = 0.001643
grad_step = 000191, loss = 0.001645
grad_step = 000192, loss = 0.001664
grad_step = 000193, loss = 0.001683
grad_step = 000194, loss = 0.001695
grad_step = 000195, loss = 0.001700
grad_step = 000196, loss = 0.001716
grad_step = 000197, loss = 0.001748
grad_step = 000198, loss = 0.001796
grad_step = 000199, loss = 0.001770
grad_step = 000200, loss = 0.001687
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001597
grad_step = 000202, loss = 0.001596
grad_step = 000203, loss = 0.001656
grad_step = 000204, loss = 0.001678
grad_step = 000205, loss = 0.001647
grad_step = 000206, loss = 0.001598
grad_step = 000207, loss = 0.001589
grad_step = 000208, loss = 0.001607
grad_step = 000209, loss = 0.001615
grad_step = 000210, loss = 0.001588
grad_step = 000211, loss = 0.001557
grad_step = 000212, loss = 0.001556
grad_step = 000213, loss = 0.001580
grad_step = 000214, loss = 0.001596
grad_step = 000215, loss = 0.001586
grad_step = 000216, loss = 0.001557
grad_step = 000217, loss = 0.001537
grad_step = 000218, loss = 0.001537
grad_step = 000219, loss = 0.001546
grad_step = 000220, loss = 0.001551
grad_step = 000221, loss = 0.001542
grad_step = 000222, loss = 0.001527
grad_step = 000223, loss = 0.001512
grad_step = 000224, loss = 0.001505
grad_step = 000225, loss = 0.001507
grad_step = 000226, loss = 0.001520
grad_step = 000227, loss = 0.001545
grad_step = 000228, loss = 0.001586
grad_step = 000229, loss = 0.001641
grad_step = 000230, loss = 0.001717
grad_step = 000231, loss = 0.001764
grad_step = 000232, loss = 0.001735
grad_step = 000233, loss = 0.001672
grad_step = 000234, loss = 0.001669
grad_step = 000235, loss = 0.001658
grad_step = 000236, loss = 0.001633
grad_step = 000237, loss = 0.001573
grad_step = 000238, loss = 0.001531
grad_step = 000239, loss = 0.001576
grad_step = 000240, loss = 0.001619
grad_step = 000241, loss = 0.001588
grad_step = 000242, loss = 0.001483
grad_step = 000243, loss = 0.001491
grad_step = 000244, loss = 0.001572
grad_step = 000245, loss = 0.001568
grad_step = 000246, loss = 0.001483
grad_step = 000247, loss = 0.001450
grad_step = 000248, loss = 0.001499
grad_step = 000249, loss = 0.001529
grad_step = 000250, loss = 0.001483
grad_step = 000251, loss = 0.001445
grad_step = 000252, loss = 0.001462
grad_step = 000253, loss = 0.001485
grad_step = 000254, loss = 0.001469
grad_step = 000255, loss = 0.001439
grad_step = 000256, loss = 0.001442
grad_step = 000257, loss = 0.001461
grad_step = 000258, loss = 0.001461
grad_step = 000259, loss = 0.001439
grad_step = 000260, loss = 0.001424
grad_step = 000261, loss = 0.001430
grad_step = 000262, loss = 0.001443
grad_step = 000263, loss = 0.001445
grad_step = 000264, loss = 0.001430
grad_step = 000265, loss = 0.001419
grad_step = 000266, loss = 0.001420
grad_step = 000267, loss = 0.001427
grad_step = 000268, loss = 0.001431
grad_step = 000269, loss = 0.001425
grad_step = 000270, loss = 0.001417
grad_step = 000271, loss = 0.001410
grad_step = 000272, loss = 0.001407
grad_step = 000273, loss = 0.001408
grad_step = 000274, loss = 0.001411
grad_step = 000275, loss = 0.001412
grad_step = 000276, loss = 0.001410
grad_step = 000277, loss = 0.001408
grad_step = 000278, loss = 0.001407
grad_step = 000279, loss = 0.001410
grad_step = 000280, loss = 0.001423
grad_step = 000281, loss = 0.001455
grad_step = 000282, loss = 0.001527
grad_step = 000283, loss = 0.001661
grad_step = 000284, loss = 0.001857
grad_step = 000285, loss = 0.001993
grad_step = 000286, loss = 0.001831
grad_step = 000287, loss = 0.001554
grad_step = 000288, loss = 0.001445
grad_step = 000289, loss = 0.001560
grad_step = 000290, loss = 0.001642
grad_step = 000291, loss = 0.001542
grad_step = 000292, loss = 0.001432
grad_step = 000293, loss = 0.001507
grad_step = 000294, loss = 0.001617
grad_step = 000295, loss = 0.001532
grad_step = 000296, loss = 0.001413
grad_step = 000297, loss = 0.001479
grad_step = 000298, loss = 0.001487
grad_step = 000299, loss = 0.001472
grad_step = 000300, loss = 0.001407
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001435
grad_step = 000302, loss = 0.001457
grad_step = 000303, loss = 0.001418
grad_step = 000304, loss = 0.001409
grad_step = 000305, loss = 0.001401
grad_step = 000306, loss = 0.001435
grad_step = 000307, loss = 0.001407
grad_step = 000308, loss = 0.001386
grad_step = 000309, loss = 0.001400
grad_step = 000310, loss = 0.001402
grad_step = 000311, loss = 0.001403
grad_step = 000312, loss = 0.001375
grad_step = 000313, loss = 0.001384
grad_step = 000314, loss = 0.001394
grad_step = 000315, loss = 0.001384
grad_step = 000316, loss = 0.001377
grad_step = 000317, loss = 0.001368
grad_step = 000318, loss = 0.001379
grad_step = 000319, loss = 0.001377
grad_step = 000320, loss = 0.001366
grad_step = 000321, loss = 0.001365
grad_step = 000322, loss = 0.001364
grad_step = 000323, loss = 0.001367
grad_step = 000324, loss = 0.001363
grad_step = 000325, loss = 0.001356
grad_step = 000326, loss = 0.001357
grad_step = 000327, loss = 0.001357
grad_step = 000328, loss = 0.001356
grad_step = 000329, loss = 0.001354
grad_step = 000330, loss = 0.001348
grad_step = 000331, loss = 0.001348
grad_step = 000332, loss = 0.001349
grad_step = 000333, loss = 0.001348
grad_step = 000334, loss = 0.001346
grad_step = 000335, loss = 0.001343
grad_step = 000336, loss = 0.001340
grad_step = 000337, loss = 0.001340
grad_step = 000338, loss = 0.001339
grad_step = 000339, loss = 0.001338
grad_step = 000340, loss = 0.001336
grad_step = 000341, loss = 0.001334
grad_step = 000342, loss = 0.001332
grad_step = 000343, loss = 0.001331
grad_step = 000344, loss = 0.001330
grad_step = 000345, loss = 0.001329
grad_step = 000346, loss = 0.001328
grad_step = 000347, loss = 0.001326
grad_step = 000348, loss = 0.001325
grad_step = 000349, loss = 0.001324
grad_step = 000350, loss = 0.001323
grad_step = 000351, loss = 0.001323
grad_step = 000352, loss = 0.001325
grad_step = 000353, loss = 0.001332
grad_step = 000354, loss = 0.001348
grad_step = 000355, loss = 0.001380
grad_step = 000356, loss = 0.001450
grad_step = 000357, loss = 0.001548
grad_step = 000358, loss = 0.001675
grad_step = 000359, loss = 0.001717
grad_step = 000360, loss = 0.001608
grad_step = 000361, loss = 0.001432
grad_step = 000362, loss = 0.001326
grad_step = 000363, loss = 0.001395
grad_step = 000364, loss = 0.001504
grad_step = 000365, loss = 0.001450
grad_step = 000366, loss = 0.001338
grad_step = 000367, loss = 0.001322
grad_step = 000368, loss = 0.001393
grad_step = 000369, loss = 0.001408
grad_step = 000370, loss = 0.001341
grad_step = 000371, loss = 0.001302
grad_step = 000372, loss = 0.001336
grad_step = 000373, loss = 0.001371
grad_step = 000374, loss = 0.001347
grad_step = 000375, loss = 0.001300
grad_step = 000376, loss = 0.001293
grad_step = 000377, loss = 0.001324
grad_step = 000378, loss = 0.001344
grad_step = 000379, loss = 0.001326
grad_step = 000380, loss = 0.001297
grad_step = 000381, loss = 0.001288
grad_step = 000382, loss = 0.001305
grad_step = 000383, loss = 0.001321
grad_step = 000384, loss = 0.001318
grad_step = 000385, loss = 0.001304
grad_step = 000386, loss = 0.001301
grad_step = 000387, loss = 0.001320
grad_step = 000388, loss = 0.001344
grad_step = 000389, loss = 0.001366
grad_step = 000390, loss = 0.001370
grad_step = 000391, loss = 0.001379
grad_step = 000392, loss = 0.001373
grad_step = 000393, loss = 0.001368
grad_step = 000394, loss = 0.001337
grad_step = 000395, loss = 0.001301
grad_step = 000396, loss = 0.001273
grad_step = 000397, loss = 0.001268
grad_step = 000398, loss = 0.001283
grad_step = 000399, loss = 0.001305
grad_step = 000400, loss = 0.001319
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001315
grad_step = 000402, loss = 0.001304
grad_step = 000403, loss = 0.001288
grad_step = 000404, loss = 0.001279
grad_step = 000405, loss = 0.001278
grad_step = 000406, loss = 0.001278
grad_step = 000407, loss = 0.001277
grad_step = 000408, loss = 0.001270
grad_step = 000409, loss = 0.001261
grad_step = 000410, loss = 0.001254
grad_step = 000411, loss = 0.001252
grad_step = 000412, loss = 0.001255
grad_step = 000413, loss = 0.001260
grad_step = 000414, loss = 0.001267
grad_step = 000415, loss = 0.001270
grad_step = 000416, loss = 0.001272
grad_step = 000417, loss = 0.001271
grad_step = 000418, loss = 0.001271
grad_step = 000419, loss = 0.001272
grad_step = 000420, loss = 0.001277
grad_step = 000421, loss = 0.001289
grad_step = 000422, loss = 0.001312
grad_step = 000423, loss = 0.001353
grad_step = 000424, loss = 0.001392
grad_step = 000425, loss = 0.001445
grad_step = 000426, loss = 0.001452
grad_step = 000427, loss = 0.001442
grad_step = 000428, loss = 0.001398
grad_step = 000429, loss = 0.001326
grad_step = 000430, loss = 0.001283
grad_step = 000431, loss = 0.001271
grad_step = 000432, loss = 0.001300
grad_step = 000433, loss = 0.001324
grad_step = 000434, loss = 0.001299
grad_step = 000435, loss = 0.001262
grad_step = 000436, loss = 0.001247
grad_step = 000437, loss = 0.001246
grad_step = 000438, loss = 0.001255
grad_step = 000439, loss = 0.001271
grad_step = 000440, loss = 0.001276
grad_step = 000441, loss = 0.001263
grad_step = 000442, loss = 0.001235
grad_step = 000443, loss = 0.001224
grad_step = 000444, loss = 0.001234
grad_step = 000445, loss = 0.001244
grad_step = 000446, loss = 0.001246
grad_step = 000447, loss = 0.001241
grad_step = 000448, loss = 0.001234
grad_step = 000449, loss = 0.001227
grad_step = 000450, loss = 0.001222
grad_step = 000451, loss = 0.001221
grad_step = 000452, loss = 0.001224
grad_step = 000453, loss = 0.001228
grad_step = 000454, loss = 0.001227
grad_step = 000455, loss = 0.001221
grad_step = 000456, loss = 0.001215
grad_step = 000457, loss = 0.001213
grad_step = 000458, loss = 0.001213
grad_step = 000459, loss = 0.001214
grad_step = 000460, loss = 0.001216
grad_step = 000461, loss = 0.001219
grad_step = 000462, loss = 0.001221
grad_step = 000463, loss = 0.001225
grad_step = 000464, loss = 0.001228
grad_step = 000465, loss = 0.001235
grad_step = 000466, loss = 0.001240
grad_step = 000467, loss = 0.001258
grad_step = 000468, loss = 0.001277
grad_step = 000469, loss = 0.001325
grad_step = 000470, loss = 0.001367
grad_step = 000471, loss = 0.001437
grad_step = 000472, loss = 0.001441
grad_step = 000473, loss = 0.001420
grad_step = 000474, loss = 0.001310
grad_step = 000475, loss = 0.001219
grad_step = 000476, loss = 0.001203
grad_step = 000477, loss = 0.001254
grad_step = 000478, loss = 0.001303
grad_step = 000479, loss = 0.001283
grad_step = 000480, loss = 0.001230
grad_step = 000481, loss = 0.001197
grad_step = 000482, loss = 0.001212
grad_step = 000483, loss = 0.001245
grad_step = 000484, loss = 0.001255
grad_step = 000485, loss = 0.001236
grad_step = 000486, loss = 0.001210
grad_step = 000487, loss = 0.001200
grad_step = 000488, loss = 0.001216
grad_step = 000489, loss = 0.001234
grad_step = 000490, loss = 0.001246
grad_step = 000491, loss = 0.001236
grad_step = 000492, loss = 0.001226
grad_step = 000493, loss = 0.001217
grad_step = 000494, loss = 0.001226
grad_step = 000495, loss = 0.001224
grad_step = 000496, loss = 0.001221
grad_step = 000497, loss = 0.001196
grad_step = 000498, loss = 0.001180
grad_step = 000499, loss = 0.001182
grad_step = 000500, loss = 0.001191
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001201
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

  date_run                              2020-05-10 07:12:22.814089
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.248148
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 07:12:22.820210
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.163587
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 07:12:22.827586
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.14293
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 07:12:22.832649
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.48577
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
0   2020-05-10 07:11:55.771423  ...    mean_absolute_error
1   2020-05-10 07:11:55.775068  ...     mean_squared_error
2   2020-05-10 07:11:55.777436  ...  median_absolute_error
3   2020-05-10 07:11:55.780229  ...               r2_score
4   2020-05-10 07:12:03.487761  ...    mean_absolute_error
5   2020-05-10 07:12:03.492319  ...     mean_squared_error
6   2020-05-10 07:12:03.495629  ...  median_absolute_error
7   2020-05-10 07:12:03.498612  ...               r2_score
8   2020-05-10 07:12:22.814089  ...    mean_absolute_error
9   2020-05-10 07:12:22.820210  ...     mean_squared_error
10  2020-05-10 07:12:22.827586  ...  median_absolute_error
11  2020-05-10 07:12:22.832649  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 137842.28it/s] 87%|████████▋ | 8642560/9912422 [00:00<00:06, 196782.77it/s]9920512it [00:00, 43332590.29it/s]                           
0it [00:00, ?it/s]32768it [00:00, 638736.99it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 6788934.47it/s]          
0it [00:00, ?it/s]8192it [00:00, 203101.75it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f910e94b780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f90c12fdcc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f910e94be80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f90c12fdcc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f90ac0900b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f90b57b1518> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f910e94b780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f90b57b1518> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f910e94b780> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f90b57b1518> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f910e94b780> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f81edc04208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=6ebfe5df41c14946980bbe54d657a78f7cdd23abce69f7ceb3614778e0490cad
  Stored in directory: /tmp/pip-ephem-wheel-cache-x94go_yw/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f81857e8080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4046848/17464789 [=====>........................] - ETA: 0s
 9781248/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 07:13:46.942075: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 07:13:46.945993: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 07:13:46.946102: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564bae710ab0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 07:13:46.946112: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4877 - accuracy: 0.5117
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4788 - accuracy: 0.5123
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5440 - accuracy: 0.5080
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5542 - accuracy: 0.5073
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5461 - accuracy: 0.5079
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5804 - accuracy: 0.5056
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6036 - accuracy: 0.5041
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6068 - accuracy: 0.5039
11000/25000 [============>.................] - ETA: 3s - loss: 7.5955 - accuracy: 0.5046
12000/25000 [=============>................] - ETA: 3s - loss: 7.6232 - accuracy: 0.5028
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6536 - accuracy: 0.5008
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6502 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 2s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6091 - accuracy: 0.5038
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6369 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6343 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6247 - accuracy: 0.5027
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6398 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 6s 260us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 07:13:59.438437
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 07:13:59.438437  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 07:14:04.917629: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 07:14:04.923582: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 07:14:04.923893: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cfcc3dea00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 07:14:04.923911: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1f8389bbe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7022 - crf_viterbi_accuracy: 0.3467 - val_loss: 1.6076 - val_crf_viterbi_accuracy: 0.3333

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1f5e7e0f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7637 - accuracy: 0.4937
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8621 - accuracy: 0.4873
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8721 - accuracy: 0.4866
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8430 - accuracy: 0.4885
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8134 - accuracy: 0.4904
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.8161 - accuracy: 0.4902
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7893 - accuracy: 0.4920
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7770 - accuracy: 0.4928
11000/25000 [============>.................] - ETA: 3s - loss: 7.7865 - accuracy: 0.4922
12000/25000 [=============>................] - ETA: 3s - loss: 7.7740 - accuracy: 0.4930
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7846 - accuracy: 0.4923
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7772 - accuracy: 0.4928
15000/25000 [=================>............] - ETA: 2s - loss: 7.7556 - accuracy: 0.4942
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7548 - accuracy: 0.4942
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7280 - accuracy: 0.4960
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7211 - accuracy: 0.4964
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7037 - accuracy: 0.4976
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6996 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6944 - accuracy: 0.4982
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 7s 260us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1f1a549278> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:07<224:47:34, 1.07kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:07<157:35:10, 1.52kB/s].vector_cache/glove.6B.zip:   0%|          | 180k/862M [00:07<110:22:03, 2.17kB/s] .vector_cache/glove.6B.zip:   0%|          | 729k/862M [00:08<77:13:32, 3.10kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.92M/862M [00:08<53:55:28, 4.43kB/s].vector_cache/glove.6B.zip:   1%|          | 8.59M/862M [00:08<37:29:58, 6.32kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:08<26:08:48, 9.03kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:08<18:12:12, 12.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:08<12:40:45, 18.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:08<8:49:43, 26.3kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:08<6:09:09, 37.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:08<4:17:04, 53.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:09<2:59:13, 76.6kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:09<2:04:59, 109kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:09<1:26:59, 156kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:09<1:00:44, 223kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:10<44:20, 304kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:10<31:05, 432kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:12<28:14, 475kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:12<22:37, 593kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:12<16:31, 811kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:12<11:41, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:14<41:40, 320kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:14<30:34, 437kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:14<21:40, 615kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:16<18:07, 733kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:16<15:29, 857kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:16<11:27, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:16<08:10, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:17<12:52, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:18<10:27, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:18<07:39, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:19<08:17, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:20<08:35, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:20<06:42, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:20<04:49, 2.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:21<33:35, 389kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:22<24:53, 525kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:22<17:44, 736kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:23<15:22, 846kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:24<13:31, 961kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:24<10:03, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:24<07:11, 1.80MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:25<11:34, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:25<09:28, 1.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:26<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:27<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:27<06:47, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:28<05:05, 2.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:29<06:26, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:29<05:50, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:30<04:21, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:31<06:01, 2.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:31<06:54, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:32<05:30, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:32<03:59, 3.17MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:33<32:04, 395kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:33<23:46, 532kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:34<16:56, 746kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<14:43, 855kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:35<11:39, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:35<08:25, 1.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:37<08:43, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:37<08:45, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:37<06:43, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:38<04:50, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:39<12:20, 1.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:39<09:59, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:39<07:18, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:41<07:52, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:41<08:01, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:41<06:09, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:41<04:33, 2.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:43<06:39, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:43<05:57, 2.06MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:43<04:26, 2.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:45<05:53, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:45<05:25, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:45<04:04, 3.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:47<05:36, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:47<05:13, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:47<03:55, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:49<05:28, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:49<06:24, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:49<05:02, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:49<03:41, 3.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:51<07:35, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:51<06:35, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:51<04:55, 2.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:53<06:08, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:53<06:50, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:53<05:25, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:53<03:56, 3.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:55<31:10, 382kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:55<23:03, 516kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:55<16:22, 725kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:57<14:09, 836kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:57<12:25, 953kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:57<09:18, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:57<06:37, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:59<31:53, 369kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:59<23:33, 499kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:59<16:45, 700kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:01<14:23, 813kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:01<11:16, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:01<08:08, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:03<08:23, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:03<08:20, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:03<06:25, 1.81MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:03<04:38, 2.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:05<1:25:50, 135kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:05<1:01:15, 189kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:05<43:05, 268kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:07<32:44, 351kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:07<25:22, 453kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:07<18:21, 625kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:07<12:56, 883kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:09<45:20, 252kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:09<32:55, 347kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:09<23:18, 489kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:11<18:49, 603kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:11<15:31, 731kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:11<11:26, 991kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:11<08:07, 1.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:13<1:04:26, 175kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:13<46:17, 244kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:13<32:36, 345kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:15<25:16, 444kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:15<20:07, 557kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:15<14:40, 763kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:15<10:20, 1.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:17<27:32, 405kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:17<20:20, 548kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:17<14:50, 751kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:17<10:28, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:19<15:40, 707kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:19<12:09, 910kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:19<08:45, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:21<08:35, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:21<08:21, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:21<06:23, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:21<04:34, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:23<19:17, 567kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:23<14:40, 746kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:23<10:31, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:25<09:50, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:25<08:05, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:25<05:53, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<04:15, 2.54MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<23:24, 462kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<18:50, 573kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:27<13:43, 786kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:27<09:42, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:28<38:59, 275kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:29<28:26, 377kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:29<20:09, 531kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:29<14:12, 751kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:30<29:43, 359kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:31<22:58, 464kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:31<16:34, 642kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:31<11:40, 909kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:32<22:00, 481kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:33<16:31, 641kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:33<11:46, 897kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:33<08:21, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:34<3:37:12, 48.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:35<2:34:09, 68.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:35<1:48:20, 97.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:35<1:15:35, 138kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:36<1:28:28, 118kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:37<1:02:59, 166kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:37<44:14, 236kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:37<30:59, 335kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:38<54:14, 192kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:38<39:01, 266kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:39<27:31, 376kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<21:35, 478kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<16:10, 638kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:41<11:34, 889kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<10:26, 981kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<09:23, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:42<07:02, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:43<05:06, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:44<06:34, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:44<05:41, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:44<04:11, 2.42MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:46<05:12, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:46<05:49, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:46<04:30, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:47<03:19, 3.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:48<05:33, 1.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:48<04:57, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:48<03:41, 2.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:50<04:50, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:50<05:31, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:50<04:17, 2.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:50<03:11, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:52<05:14, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:52<04:41, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:52<03:32, 2.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<04:45, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<04:20, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:54<03:17, 2.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<04:33, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<05:10, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:56<04:03, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:56<03:01, 3.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:58<04:53, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:58<04:26, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:58<03:21, 2.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:00<04:33, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:00<05:14, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:00<04:06, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:00<03:00, 3.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:02<05:56, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:02<05:10, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:02<03:50, 2.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:04<04:50, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:04<05:24, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:04<04:12, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:04<03:05, 3.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:06<05:43, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:06<04:59, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:06<03:43, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<04:46, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:08<03:16, 2.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:10<04:22, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:10<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:10<03:57, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<02:51, 3.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<20:06, 459kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<15:02, 612kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:12<10:44, 855kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:14<09:33, 957kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:14<08:37, 1.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:14<06:26, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:14<04:37, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:16<07:07, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:16<05:56, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:16<04:20, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:18<05:07, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:18<05:31, 1.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<04:16, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:18<03:06, 2.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:20<05:46, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<04:59, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:20<03:41, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<04:35, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<05:06, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<04:02, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:22<02:54, 3.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:24<24:07, 365kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:24<17:47, 495kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:24<12:37, 696kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:26<10:48, 808kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:26<09:25, 927kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:26<06:58, 1.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:26<04:59, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:28<06:57, 1.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:28<05:47, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:28<04:16, 2.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:29<04:57, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:30<05:17, 1.62MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:30<04:05, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:30<02:58, 2.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:31<05:52, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:32<05:00, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:32<03:41, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:32<02:44, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<20:45, 408kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:34<15:49, 535kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:34<11:20, 745kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:35<09:25, 892kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:36<07:52, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:36<05:49, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<05:33, 1.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<05:09, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:38<03:55, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:39<04:13, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:39<04:12, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:40<03:15, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:41<03:45, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:41<03:52, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:42<03:01, 2.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:43<03:34, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:43<03:44, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<02:55, 2.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<03:29, 2.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<05:04, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:46<04:10, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:46<03:02, 2.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<04:43, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<04:31, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:47<03:27, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<03:50, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:49<03:54, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:49<03:01, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:51<03:31, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:51<03:41, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:51<02:52, 2.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:53<03:24, 2.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:53<03:34, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:53<02:45, 2.81MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:55<03:19, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:55<03:30, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:55<02:44, 2.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:57<03:17, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:57<03:28, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:57<02:43, 2.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<03:15, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<03:26, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:59<02:41, 2.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:01<02:41, 2.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<03:13, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<03:23, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:03<02:38, 2.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:05<03:10, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:05<03:20, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:05<02:34, 2.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:05<01:52, 3.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:07<14:53, 490kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:07<11:31, 632kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:07<08:19, 873kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:09<07:05, 1.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:09<06:04, 1.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:09<04:30, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:11<04:25, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:11<04:11, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:11<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:13<03:30, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:13<03:32, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:13<02:44, 2.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<03:10, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<03:17, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:15<02:33, 2.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:17<03:02, 2.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:17<03:11, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:17<02:29, 2.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<02:58, 2.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:19<03:08, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:19<02:24, 2.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:19<01:45, 3.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:21<07:23, 922kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:21<06:12, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:21<04:35, 1.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:23<04:24, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:23<04:09, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<03:09, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:25<02:37, 2.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<03:01, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<03:07, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<02:25, 2.71MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:29<02:52, 2.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:29<03:00, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:29<02:20, 2.78MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:31<02:48, 2.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:31<02:56, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:31<02:15, 2.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:31<01:39, 3.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:33<07:24, 864kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:33<06:09, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:33<04:32, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:35<04:17, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:35<03:53, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:35<02:56, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:37<03:13, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:37<04:18, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:37<03:29, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:37<02:32, 2.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:38<03:46, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:39<03:35, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:39<02:43, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<01:57, 3.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:40<29:20, 209kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:41<21:27, 285kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:41<15:12, 401kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:42<11:39, 519kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:43<09:05, 666kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:43<06:34, 918kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:44<05:38, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:45<04:50, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:45<03:33, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:45<02:33, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:46<07:18, 810kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:47<05:58, 988kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:47<04:23, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:48<04:07, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:48<03:46, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:49<02:51, 2.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:50<03:01, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:50<03:00, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:51<02:19, 2.48MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:52<02:38, 2.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:52<02:43, 2.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:53<02:07, 2.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:54<02:29, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:54<02:31, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:55<01:57, 2.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:56<02:27, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<02:04, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:57<01:32, 3.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:58<02:55, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:58<02:53, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:58<02:13, 2.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<02:31, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<02:35, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:00<01:59, 2.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:01<01:26, 3.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<16:11, 331kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:02<12:08, 441kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:02<08:40, 616kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<06:58, 758kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:04<05:41, 928kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:04<04:09, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:04<02:57, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:06<07:24, 705kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:06<05:58, 873kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:06<04:22, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:08<03:58, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:08<03:33, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:08<02:41, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:10<02:46, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:10<03:30, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:10<02:49, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:10<02:02, 2.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:12<03:10, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:12<02:58, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<02:16, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:14<02:28, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:14<02:28, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:14<01:55, 2.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:16<02:12, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:16<02:17, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:16<01:47, 2.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:18<02:06, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:18<02:13, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:18<01:44, 2.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:20<02:03, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:20<02:10, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<01:41, 2.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:22<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:22<02:07, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:22<01:38, 2.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<01:11, 3.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:24<07:20, 627kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:24<05:50, 787kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:24<04:15, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:26<03:45, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:26<03:19, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:26<02:29, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:28<02:31, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:28<02:27, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:28<01:52, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<01:21, 3.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:30<04:26, 990kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:30<03:47, 1.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:30<02:48, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:32<02:43, 1.59MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:32<02:34, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:32<01:57, 2.20MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<02:07, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<02:08, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:34<01:37, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:34<01:11, 3.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:36<03:34, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:36<03:08, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:36<02:21, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:38<02:22, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:38<02:17, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:38<01:44, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:38<01:15, 3.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:40<03:55, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:40<03:21, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:40<02:30, 1.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:42<02:26, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:42<02:12, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:42<01:38, 2.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:44<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:44<02:36, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:44<02:08, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:44<01:33, 2.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:46<02:19, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:46<02:12, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:46<01:41, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:48<01:51, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:48<01:52, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:48<01:26, 2.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:50<01:40, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:50<01:44, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:50<01:20, 2.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:52<01:35, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:52<02:13, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:52<01:50, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:52<01:20, 2.67MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:54<02:05, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:54<02:00, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:54<01:32, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:55<01:41, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:56<01:44, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:56<01:20, 2.59MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:57<01:32, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:58<01:36, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:58<01:13, 2.78MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:58<00:54, 3.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:59<02:17, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:00<02:06, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:00<01:35, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:01<01:41, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:02<01:41, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:02<01:18, 2.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:03<01:28, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:04<01:31, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:04<01:11, 2.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:05<01:23, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:05<01:27, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:06<01:08, 2.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:07<01:20, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:07<01:18, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:08<00:59, 3.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:09<01:18, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:09<01:22, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:10<01:04, 2.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:11<01:16, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:11<01:14, 2.36MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:12<00:57, 3.05MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:13<01:14, 2.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:13<01:13, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:14<00:56, 3.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:15<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:15<01:42, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:15<01:23, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:16<01:02, 2.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:17<01:17, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:17<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:17<01:01, 2.67MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:19<01:11, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:19<01:14, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:19<00:57, 2.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:21<01:07, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:21<01:11, 2.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:21<00:55, 2.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:23<01:05, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:23<01:09, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:23<00:52, 2.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<00:38, 3.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:25<03:09, 779kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:25<02:35, 950kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:25<01:52, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:25<01:20, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:27<02:05, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:27<01:50, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:27<01:21, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:27<00:58, 2.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:29<01:53, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:29<01:36, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:29<01:11, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:31<01:16, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:31<01:14, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:31<00:55, 2.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:31<00:40, 3.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:33<01:44, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:33<01:33, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:33<01:09, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:35<01:10, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:35<01:08, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:35<00:52, 2.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:37<00:58, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:37<00:59, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:37<00:46, 2.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:39<00:52, 2.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:39<00:55, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:39<00:42, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:41<00:50, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:41<00:51, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:41<00:40, 2.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<00:29, 3.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:43<01:50, 1.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:43<01:33, 1.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:43<01:09, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:45<01:06, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:45<01:02, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:45<00:47, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:47<00:50, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:47<00:51, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:47<00:39, 2.58MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:49<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:49<00:46, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:49<00:35, 2.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:51<00:41, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:51<00:43, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:51<00:32, 2.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:51<00:23, 3.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:53<02:10, 689kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:53<01:44, 856kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:53<01:15, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:55<01:07, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:55<00:59, 1.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:55<00:44, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:57<00:45, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:57<00:43, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:57<00:33, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:59<00:36, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:59<00:37, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:59<00:28, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:01<00:32, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:01<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:01<00:26, 2.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<00:30, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:03<00:24, 2.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<00:28, 2.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<00:29, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:05<00:22, 2.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<00:26, 2.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<00:27, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:07<00:20, 2.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<00:14, 3.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:08<00:49, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:09<00:43, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:09<00:31, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:10<00:30, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<00:29, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:11<00:22, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:13<00:24, 2.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:13<00:19, 2.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:13<00:13, 3.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:14<00:24, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<00:23, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:15<00:18, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:16<00:19, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:16<00:19, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:17<00:14, 2.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:17<00:10, 3.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:18<00:34, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:18<00:29, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:19<00:21, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:19<00:14, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:20<00:47, 678kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:20<00:37, 844kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:21<00:26, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:21<00:17, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:22<00:43, 644kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:22<00:34, 806kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:23<00:24, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:24<00:19, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:24<00:17, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:24<00:12, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:25<00:08, 2.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:26<00:17, 1.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:26<00:18, 1.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:26<00:13, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:27<00:08, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:28<00:09, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:28<00:09, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:28<00:06, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:29<00:03, 3.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:30<00:15, 741kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:30<00:12, 911kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:30<00:08, 1.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:32<00:05, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:32<00:04, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:32<00:03, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:34<00:01, 1.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:34<00:01, 1.89MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:34<00:00, 2.44MB/s].vector_cache/glove.6B.zip: 862MB [06:34, 2.18MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1008/400000 [00:00<00:39, 10072.95it/s]  0%|          | 1902/400000 [00:00<00:41, 9702.38it/s]   1%|          | 2936/400000 [00:00<00:40, 9882.15it/s]  1%|          | 3883/400000 [00:00<00:40, 9754.36it/s]  1%|          | 4913/400000 [00:00<00:39, 9909.49it/s]  1%|▏         | 5899/400000 [00:00<00:39, 9893.53it/s]  2%|▏         | 7000/400000 [00:00<00:38, 10201.61it/s]  2%|▏         | 7994/400000 [00:00<00:38, 10121.54it/s]  2%|▏         | 8964/400000 [00:00<00:39, 9990.59it/s]   2%|▏         | 9984/400000 [00:01<00:38, 10051.85it/s]  3%|▎         | 10963/400000 [00:01<00:39, 9881.07it/s]  3%|▎         | 11933/400000 [00:01<00:40, 9679.57it/s]  3%|▎         | 12890/400000 [00:01<00:40, 9630.28it/s]  3%|▎         | 13845/400000 [00:01<00:41, 9416.92it/s]  4%|▎         | 14870/400000 [00:01<00:39, 9649.07it/s]  4%|▍         | 15890/400000 [00:01<00:39, 9806.94it/s]  4%|▍         | 16883/400000 [00:01<00:38, 9843.11it/s]  4%|▍         | 17867/400000 [00:01<00:39, 9766.05it/s]  5%|▍         | 18844/400000 [00:01<00:39, 9610.84it/s]  5%|▍         | 19806/400000 [00:02<00:40, 9467.49it/s]  5%|▌         | 20754/400000 [00:02<00:40, 9291.74it/s]  5%|▌         | 21798/400000 [00:02<00:39, 9608.64it/s]  6%|▌         | 22763/400000 [00:02<00:39, 9527.27it/s]  6%|▌         | 23756/400000 [00:02<00:39, 9643.93it/s]  6%|▌         | 24751/400000 [00:02<00:38, 9732.61it/s]  6%|▋         | 25781/400000 [00:02<00:37, 9893.55it/s]  7%|▋         | 26805/400000 [00:02<00:37, 9994.41it/s]  7%|▋         | 27822/400000 [00:02<00:37, 10045.19it/s]  7%|▋         | 28828/400000 [00:02<00:38, 9524.19it/s]   7%|▋         | 29787/400000 [00:03<00:39, 9389.11it/s]  8%|▊         | 30775/400000 [00:03<00:38, 9530.43it/s]  8%|▊         | 31764/400000 [00:03<00:38, 9632.89it/s]  8%|▊         | 32731/400000 [00:03<00:38, 9606.74it/s]  8%|▊         | 33694/400000 [00:03<00:38, 9542.99it/s]  9%|▊         | 34650/400000 [00:03<00:38, 9459.27it/s]  9%|▉         | 35725/400000 [00:03<00:37, 9810.38it/s]  9%|▉         | 36737/400000 [00:03<00:36, 9898.92it/s]  9%|▉         | 37783/400000 [00:03<00:36, 10058.83it/s] 10%|▉         | 38792/400000 [00:03<00:36, 9869.28it/s]  10%|▉         | 39791/400000 [00:04<00:36, 9903.99it/s] 10%|█         | 40784/400000 [00:04<00:36, 9800.19it/s] 10%|█         | 41766/400000 [00:04<00:37, 9539.68it/s] 11%|█         | 42749/400000 [00:04<00:37, 9623.43it/s] 11%|█         | 43714/400000 [00:04<00:37, 9628.56it/s] 11%|█         | 44791/400000 [00:04<00:35, 9943.04it/s] 11%|█▏        | 45789/400000 [00:04<00:35, 9855.18it/s] 12%|█▏        | 46857/400000 [00:04<00:35, 10086.88it/s] 12%|█▏        | 47883/400000 [00:04<00:34, 10135.21it/s] 12%|█▏        | 48899/400000 [00:05<00:35, 9908.47it/s]  12%|█▏        | 49893/400000 [00:05<00:35, 9749.06it/s] 13%|█▎        | 50871/400000 [00:05<00:36, 9521.50it/s] 13%|█▎        | 51827/400000 [00:05<00:36, 9513.72it/s] 13%|█▎        | 52781/400000 [00:05<00:38, 9079.64it/s] 13%|█▎        | 53718/400000 [00:05<00:37, 9163.23it/s] 14%|█▎        | 54730/400000 [00:05<00:36, 9429.95it/s] 14%|█▍        | 55770/400000 [00:05<00:35, 9698.85it/s] 14%|█▍        | 56813/400000 [00:05<00:34, 9905.74it/s] 14%|█▍        | 57809/400000 [00:05<00:34, 9779.74it/s] 15%|█▍        | 58815/400000 [00:06<00:34, 9861.77it/s] 15%|█▍        | 59829/400000 [00:06<00:34, 9941.73it/s] 15%|█▌        | 60873/400000 [00:06<00:33, 10084.89it/s] 15%|█▌        | 61884/400000 [00:06<00:33, 10004.29it/s] 16%|█▌        | 62886/400000 [00:06<00:33, 9961.86it/s]  16%|█▌        | 63884/400000 [00:06<00:34, 9776.39it/s] 16%|█▌        | 64940/400000 [00:06<00:33, 9996.84it/s] 17%|█▋        | 66006/400000 [00:06<00:32, 10184.52it/s] 17%|█▋        | 67027/400000 [00:06<00:32, 10139.25it/s] 17%|█▋        | 68043/400000 [00:06<00:33, 9954.08it/s]  17%|█▋        | 69041/400000 [00:07<00:34, 9616.00it/s] 18%|█▊        | 70039/400000 [00:07<00:33, 9720.49it/s] 18%|█▊        | 71015/400000 [00:07<00:34, 9565.51it/s] 18%|█▊        | 71975/400000 [00:07<00:34, 9495.25it/s] 18%|█▊        | 72954/400000 [00:07<00:34, 9579.82it/s] 19%|█▊        | 74007/400000 [00:07<00:33, 9845.05it/s] 19%|█▉        | 75047/400000 [00:07<00:32, 10003.43it/s] 19%|█▉        | 76051/400000 [00:07<00:32, 9922.82it/s]  19%|█▉        | 77094/400000 [00:07<00:32, 10069.02it/s] 20%|█▉        | 78103/400000 [00:07<00:32, 9934.95it/s]  20%|█▉        | 79099/400000 [00:08<00:32, 9926.93it/s] 20%|██        | 80165/400000 [00:08<00:31, 10135.34it/s] 20%|██        | 81226/400000 [00:08<00:31, 10272.98it/s] 21%|██        | 82256/400000 [00:08<00:32, 9888.61it/s]  21%|██        | 83250/400000 [00:08<00:33, 9593.28it/s] 21%|██        | 84215/400000 [00:08<00:33, 9562.83it/s] 21%|██▏       | 85237/400000 [00:08<00:32, 9749.56it/s] 22%|██▏       | 86216/400000 [00:08<00:32, 9703.97it/s] 22%|██▏       | 87294/400000 [00:08<00:31, 10000.87it/s] 22%|██▏       | 88299/400000 [00:09<00:31, 9756.96it/s]  22%|██▏       | 89318/400000 [00:09<00:31, 9879.98it/s] 23%|██▎       | 90331/400000 [00:09<00:31, 9952.96it/s] 23%|██▎       | 91329/400000 [00:09<00:32, 9583.47it/s] 23%|██▎       | 92292/400000 [00:09<00:33, 9174.07it/s] 23%|██▎       | 93283/400000 [00:09<00:32, 9381.09it/s] 24%|██▎       | 94232/400000 [00:09<00:32, 9410.75it/s] 24%|██▍       | 95317/400000 [00:09<00:31, 9800.02it/s] 24%|██▍       | 96401/400000 [00:09<00:30, 10089.34it/s] 24%|██▍       | 97456/400000 [00:09<00:29, 10221.13it/s] 25%|██▍       | 98484/400000 [00:10<00:29, 10052.35it/s] 25%|██▍       | 99508/400000 [00:10<00:29, 10107.51it/s] 25%|██▌       | 100522/400000 [00:10<00:29, 10090.81it/s] 25%|██▌       | 101560/400000 [00:10<00:29, 10175.47it/s] 26%|██▌       | 102580/400000 [00:10<00:29, 9934.62it/s]  26%|██▌       | 103576/400000 [00:10<00:30, 9593.82it/s] 26%|██▌       | 104635/400000 [00:10<00:29, 9871.34it/s] 26%|██▋       | 105731/400000 [00:10<00:28, 10173.42it/s] 27%|██▋       | 106800/400000 [00:10<00:28, 10321.32it/s] 27%|██▋       | 107837/400000 [00:10<00:28, 10280.06it/s] 27%|██▋       | 108869/400000 [00:11<00:29, 9981.75it/s]  27%|██▋       | 109896/400000 [00:11<00:28, 10064.64it/s] 28%|██▊       | 110920/400000 [00:11<00:28, 10115.27it/s] 28%|██▊       | 111946/400000 [00:11<00:28, 10156.87it/s] 28%|██▊       | 112964/400000 [00:11<00:29, 9803.59it/s]  28%|██▊       | 113949/400000 [00:11<00:29, 9622.23it/s] 29%|██▊       | 114959/400000 [00:11<00:29, 9759.27it/s] 29%|██▉       | 115938/400000 [00:11<00:29, 9639.51it/s] 29%|██▉       | 116905/400000 [00:11<00:30, 9374.39it/s] 29%|██▉       | 117846/400000 [00:12<00:30, 9263.12it/s] 30%|██▉       | 118812/400000 [00:12<00:29, 9376.54it/s] 30%|██▉       | 119802/400000 [00:12<00:29, 9525.17it/s] 30%|███       | 120814/400000 [00:12<00:28, 9693.97it/s] 30%|███       | 121786/400000 [00:12<00:28, 9681.02it/s] 31%|███       | 122756/400000 [00:12<00:28, 9683.64it/s] 31%|███       | 123742/400000 [00:12<00:28, 9733.94it/s] 31%|███       | 124717/400000 [00:12<00:28, 9505.41it/s] 31%|███▏      | 125684/400000 [00:12<00:28, 9552.56it/s] 32%|███▏      | 126641/400000 [00:12<00:29, 9322.98it/s] 32%|███▏      | 127581/400000 [00:13<00:29, 9343.18it/s] 32%|███▏      | 128517/400000 [00:13<00:29, 9223.75it/s] 32%|███▏      | 129460/400000 [00:13<00:29, 9283.46it/s] 33%|███▎      | 130390/400000 [00:13<00:29, 9274.06it/s] 33%|███▎      | 131319/400000 [00:13<00:29, 9076.17it/s] 33%|███▎      | 132229/400000 [00:13<00:29, 8963.05it/s] 33%|███▎      | 133214/400000 [00:13<00:28, 9211.85it/s] 34%|███▎      | 134255/400000 [00:13<00:27, 9538.85it/s] 34%|███▍      | 135297/400000 [00:13<00:27, 9786.42it/s] 34%|███▍      | 136281/400000 [00:13<00:26, 9773.59it/s] 34%|███▍      | 137262/400000 [00:14<00:27, 9536.90it/s] 35%|███▍      | 138336/400000 [00:14<00:26, 9868.39it/s] 35%|███▍      | 139329/400000 [00:14<00:26, 9807.39it/s] 35%|███▌      | 140314/400000 [00:14<00:26, 9696.60it/s] 35%|███▌      | 141287/400000 [00:14<00:27, 9404.24it/s] 36%|███▌      | 142232/400000 [00:14<00:28, 9089.57it/s] 36%|███▌      | 143147/400000 [00:14<00:28, 8889.74it/s] 36%|███▌      | 144041/400000 [00:14<00:29, 8572.29it/s] 36%|███▌      | 144971/400000 [00:14<00:29, 8776.40it/s] 36%|███▋      | 145854/400000 [00:15<00:29, 8636.19it/s] 37%|███▋      | 146737/400000 [00:15<00:29, 8690.65it/s] 37%|███▋      | 147695/400000 [00:15<00:28, 8937.51it/s] 37%|███▋      | 148638/400000 [00:15<00:27, 9075.78it/s] 37%|███▋      | 149612/400000 [00:15<00:27, 9264.51it/s] 38%|███▊      | 150645/400000 [00:15<00:26, 9557.49it/s] 38%|███▊      | 151640/400000 [00:15<00:25, 9670.10it/s] 38%|███▊      | 152693/400000 [00:15<00:24, 9910.97it/s] 38%|███▊      | 153689/400000 [00:15<00:24, 9922.09it/s] 39%|███▊      | 154684/400000 [00:15<00:24, 9884.70it/s] 39%|███▉      | 155675/400000 [00:16<00:25, 9681.99it/s] 39%|███▉      | 156646/400000 [00:16<00:25, 9546.21it/s] 39%|███▉      | 157603/400000 [00:16<00:25, 9507.58it/s] 40%|███▉      | 158585/400000 [00:16<00:25, 9596.85it/s] 40%|███▉      | 159597/400000 [00:16<00:24, 9745.65it/s] 40%|████      | 160573/400000 [00:16<00:24, 9652.17it/s] 40%|████      | 161540/400000 [00:16<00:24, 9580.17it/s] 41%|████      | 162538/400000 [00:16<00:24, 9695.89it/s] 41%|████      | 163529/400000 [00:16<00:24, 9757.80it/s] 41%|████      | 164512/400000 [00:16<00:24, 9776.77it/s] 41%|████▏     | 165491/400000 [00:17<00:24, 9492.65it/s] 42%|████▏     | 166443/400000 [00:17<00:24, 9452.79it/s] 42%|████▏     | 167408/400000 [00:17<00:24, 9508.92it/s] 42%|████▏     | 168379/400000 [00:17<00:24, 9568.13it/s] 42%|████▏     | 169337/400000 [00:17<00:24, 9402.78it/s] 43%|████▎     | 170322/400000 [00:17<00:24, 9531.05it/s] 43%|████▎     | 171278/400000 [00:17<00:23, 9538.81it/s] 43%|████▎     | 172233/400000 [00:17<00:24, 9427.67it/s] 43%|████▎     | 173177/400000 [00:17<00:25, 9069.54it/s] 44%|████▎     | 174088/400000 [00:17<00:25, 8951.24it/s] 44%|████▍     | 175039/400000 [00:18<00:24, 9110.37it/s] 44%|████▍     | 175979/400000 [00:18<00:24, 9192.81it/s] 44%|████▍     | 176901/400000 [00:18<00:24, 9174.41it/s] 44%|████▍     | 177879/400000 [00:18<00:23, 9346.32it/s] 45%|████▍     | 178888/400000 [00:18<00:23, 9556.19it/s] 45%|████▍     | 179847/400000 [00:18<00:23, 9475.04it/s] 45%|████▌     | 180797/400000 [00:18<00:23, 9374.96it/s] 45%|████▌     | 181737/400000 [00:18<00:23, 9296.51it/s] 46%|████▌     | 182767/400000 [00:18<00:22, 9574.14it/s] 46%|████▌     | 183728/400000 [00:19<00:23, 9318.09it/s] 46%|████▌     | 184697/400000 [00:19<00:22, 9425.61it/s] 46%|████▋     | 185678/400000 [00:19<00:22, 9535.57it/s] 47%|████▋     | 186706/400000 [00:19<00:21, 9745.73it/s] 47%|████▋     | 187692/400000 [00:19<00:21, 9779.25it/s] 47%|████▋     | 188746/400000 [00:19<00:21, 9994.11it/s] 47%|████▋     | 189792/400000 [00:19<00:20, 10127.90it/s] 48%|████▊     | 190807/400000 [00:19<00:21, 9728.59it/s]  48%|████▊     | 191785/400000 [00:19<00:21, 9693.30it/s] 48%|████▊     | 192758/400000 [00:19<00:21, 9688.72it/s] 48%|████▊     | 193730/400000 [00:20<00:21, 9625.63it/s] 49%|████▊     | 194695/400000 [00:20<00:21, 9543.26it/s] 49%|████▉     | 195651/400000 [00:20<00:21, 9499.76it/s] 49%|████▉     | 196618/400000 [00:20<00:21, 9548.07it/s] 49%|████▉     | 197677/400000 [00:20<00:20, 9838.44it/s] 50%|████▉     | 198708/400000 [00:20<00:20, 9973.90it/s] 50%|████▉     | 199717/400000 [00:20<00:20, 10008.09it/s] 50%|█████     | 200720/400000 [00:20<00:20, 9888.51it/s]  50%|█████     | 201711/400000 [00:20<00:20, 9830.22it/s] 51%|█████     | 202713/400000 [00:20<00:19, 9885.84it/s] 51%|█████     | 203703/400000 [00:21<00:20, 9691.55it/s] 51%|█████     | 204700/400000 [00:21<00:19, 9772.18it/s] 51%|█████▏    | 205702/400000 [00:21<00:19, 9844.59it/s] 52%|█████▏    | 206734/400000 [00:21<00:19, 9982.15it/s] 52%|█████▏    | 207734/400000 [00:21<00:20, 9245.20it/s] 52%|█████▏    | 208785/400000 [00:21<00:19, 9588.92it/s] 52%|█████▏    | 209756/400000 [00:21<00:19, 9617.19it/s] 53%|█████▎    | 210801/400000 [00:21<00:19, 9850.37it/s] 53%|█████▎    | 211846/400000 [00:21<00:18, 10021.17it/s] 53%|█████▎    | 212854/400000 [00:21<00:18, 9975.49it/s]  53%|█████▎    | 213856/400000 [00:22<00:18, 9936.64it/s] 54%|█████▎    | 214897/400000 [00:22<00:18, 10071.97it/s] 54%|█████▍    | 215907/400000 [00:22<00:18, 10008.39it/s] 54%|█████▍    | 216910/400000 [00:22<00:18, 9699.85it/s]  54%|█████▍    | 217885/400000 [00:22<00:18, 9712.38it/s] 55%|█████▍    | 218950/400000 [00:22<00:18, 9973.99it/s] 55%|█████▍    | 219966/400000 [00:22<00:17, 10026.01it/s] 55%|█████▌    | 221006/400000 [00:22<00:17, 10134.66it/s] 56%|█████▌    | 222045/400000 [00:22<00:17, 10209.74it/s] 56%|█████▌    | 223068/400000 [00:23<00:18, 9826.22it/s]  56%|█████▌    | 224055/400000 [00:23<00:18, 9702.69it/s] 56%|█████▋    | 225104/400000 [00:23<00:17, 9925.43it/s] 57%|█████▋    | 226148/400000 [00:23<00:17, 10071.50it/s] 57%|█████▋    | 227159/400000 [00:23<00:17, 9965.67it/s]  57%|█████▋    | 228158/400000 [00:23<00:17, 9562.50it/s] 57%|█████▋    | 229120/400000 [00:23<00:18, 9491.63it/s] 58%|█████▊    | 230073/400000 [00:23<00:18, 9393.11it/s] 58%|█████▊    | 231097/400000 [00:23<00:17, 9630.23it/s] 58%|█████▊    | 232125/400000 [00:23<00:17, 9816.34it/s] 58%|█████▊    | 233112/400000 [00:24<00:16, 9829.03it/s] 59%|█████▊    | 234153/400000 [00:24<00:16, 9993.65it/s] 59%|█████▉    | 235184/400000 [00:24<00:16, 10085.79it/s] 59%|█████▉    | 236195/400000 [00:24<00:16, 10032.77it/s] 59%|█████▉    | 237200/400000 [00:24<00:16, 9728.32it/s]  60%|█████▉    | 238210/400000 [00:24<00:16, 9835.51it/s] 60%|█████▉    | 239196/400000 [00:24<00:16, 9821.95it/s] 60%|██████    | 240180/400000 [00:24<00:16, 9759.87it/s] 60%|██████    | 241188/400000 [00:24<00:16, 9853.39it/s] 61%|██████    | 242175/400000 [00:24<00:16, 9560.39it/s] 61%|██████    | 243154/400000 [00:25<00:16, 9626.04it/s] 61%|██████    | 244139/400000 [00:25<00:16, 9691.89it/s] 61%|██████▏   | 245110/400000 [00:25<00:16, 9619.14it/s] 62%|██████▏   | 246110/400000 [00:25<00:15, 9729.85it/s] 62%|██████▏   | 247085/400000 [00:25<00:15, 9731.77it/s] 62%|██████▏   | 248059/400000 [00:25<00:15, 9613.86it/s] 62%|██████▏   | 249022/400000 [00:25<00:15, 9444.52it/s] 63%|██████▎   | 250016/400000 [00:25<00:15, 9587.44it/s] 63%|██████▎   | 251034/400000 [00:25<00:15, 9756.55it/s] 63%|██████▎   | 252025/400000 [00:25<00:15, 9800.87it/s] 63%|██████▎   | 253045/400000 [00:26<00:14, 9915.93it/s] 64%|██████▎   | 254056/400000 [00:26<00:14, 9972.01it/s] 64%|██████▍   | 255055/400000 [00:26<00:14, 9861.84it/s] 64%|██████▍   | 256043/400000 [00:26<00:14, 9796.54it/s] 64%|██████▍   | 257081/400000 [00:26<00:14, 9962.42it/s] 65%|██████▍   | 258156/400000 [00:26<00:13, 10186.08it/s] 65%|██████▍   | 259177/400000 [00:26<00:13, 10062.56it/s] 65%|██████▌   | 260186/400000 [00:26<00:14, 9842.27it/s]  65%|██████▌   | 261173/400000 [00:26<00:14, 9567.81it/s] 66%|██████▌   | 262134/400000 [00:27<00:14, 9518.56it/s] 66%|██████▌   | 263091/400000 [00:27<00:14, 9532.82it/s] 66%|██████▌   | 264080/400000 [00:27<00:14, 9635.62it/s] 66%|██████▋   | 265045/400000 [00:27<00:14, 9544.04it/s] 67%|██████▋   | 266001/400000 [00:27<00:14, 9510.92it/s] 67%|██████▋   | 267010/400000 [00:27<00:13, 9674.86it/s] 67%|██████▋   | 267979/400000 [00:27<00:13, 9671.70it/s] 67%|██████▋   | 269031/400000 [00:27<00:13, 9908.63it/s] 68%|██████▊   | 270024/400000 [00:27<00:13, 9731.49it/s] 68%|██████▊   | 271026/400000 [00:27<00:13, 9813.25it/s] 68%|██████▊   | 272009/400000 [00:28<00:13, 9712.24it/s] 68%|██████▊   | 273049/400000 [00:28<00:12, 9906.52it/s] 69%|██████▊   | 274100/400000 [00:28<00:12, 10078.64it/s] 69%|██████▉   | 275110/400000 [00:28<00:12, 9744.71it/s]  69%|██████▉   | 276151/400000 [00:28<00:12, 9934.40it/s] 69%|██████▉   | 277169/400000 [00:28<00:12, 10005.40it/s] 70%|██████▉   | 278200/400000 [00:28<00:12, 10092.47it/s] 70%|██████▉   | 279237/400000 [00:28<00:11, 10170.78it/s] 70%|███████   | 280256/400000 [00:28<00:11, 10011.90it/s] 70%|███████   | 281284/400000 [00:28<00:11, 10090.27it/s] 71%|███████   | 282295/400000 [00:29<00:11, 10091.21it/s] 71%|███████   | 283347/400000 [00:29<00:11, 10213.99it/s] 71%|███████   | 284437/400000 [00:29<00:11, 10409.86it/s] 71%|███████▏  | 285480/400000 [00:29<00:11, 10103.06it/s] 72%|███████▏  | 286494/400000 [00:29<00:11, 9864.42it/s]  72%|███████▏  | 287567/400000 [00:29<00:11, 10107.56it/s] 72%|███████▏  | 288686/400000 [00:29<00:10, 10407.69it/s] 72%|███████▏  | 289756/400000 [00:29<00:10, 10493.17it/s] 73%|███████▎  | 290809/400000 [00:29<00:10, 10268.96it/s] 73%|███████▎  | 291840/400000 [00:29<00:10, 9839.10it/s]  73%|███████▎  | 292831/400000 [00:30<00:11, 9688.23it/s] 73%|███████▎  | 293805/400000 [00:30<00:10, 9655.83it/s] 74%|███████▎  | 294799/400000 [00:30<00:10, 9738.78it/s] 74%|███████▍  | 295877/400000 [00:30<00:10, 10028.87it/s] 74%|███████▍  | 296942/400000 [00:30<00:10, 10206.62it/s] 75%|███████▍  | 298003/400000 [00:30<00:09, 10323.31it/s] 75%|███████▍  | 299084/400000 [00:30<00:09, 10462.54it/s] 75%|███████▌  | 300174/400000 [00:30<00:09, 10589.93it/s] 75%|███████▌  | 301236/400000 [00:30<00:09, 10598.20it/s] 76%|███████▌  | 302298/400000 [00:31<00:09, 9793.31it/s]  76%|███████▌  | 303291/400000 [00:31<00:09, 9802.05it/s] 76%|███████▌  | 304281/400000 [00:31<00:09, 9579.24it/s] 76%|███████▋  | 305247/400000 [00:31<00:09, 9518.81it/s] 77%|███████▋  | 306241/400000 [00:31<00:09, 9639.98it/s] 77%|███████▋  | 307266/400000 [00:31<00:09, 9813.04it/s] 77%|███████▋  | 308379/400000 [00:31<00:09, 10172.47it/s] 77%|███████▋  | 309437/400000 [00:31<00:08, 10288.51it/s] 78%|███████▊  | 310471/400000 [00:31<00:08, 10235.42it/s] 78%|███████▊  | 311498/400000 [00:31<00:08, 10199.05it/s] 78%|███████▊  | 312521/400000 [00:32<00:08, 9847.04it/s]  78%|███████▊  | 313510/400000 [00:32<00:08, 9808.46it/s] 79%|███████▊  | 314519/400000 [00:32<00:08, 9889.26it/s] 79%|███████▉  | 315511/400000 [00:32<00:08, 9798.80it/s] 79%|███████▉  | 316493/400000 [00:32<00:08, 9492.38it/s] 79%|███████▉  | 317522/400000 [00:32<00:08, 9716.44it/s] 80%|███████▉  | 318564/400000 [00:32<00:08, 9915.58it/s] 80%|███████▉  | 319616/400000 [00:32<00:07, 10088.90it/s] 80%|████████  | 320629/400000 [00:32<00:07, 10026.02it/s] 80%|████████  | 321634/400000 [00:32<00:08, 9785.90it/s]  81%|████████  | 322616/400000 [00:33<00:08, 9558.83it/s] 81%|████████  | 323668/400000 [00:33<00:07, 9826.41it/s] 81%|████████  | 324655/400000 [00:33<00:07, 9831.71it/s] 81%|████████▏ | 325642/400000 [00:33<00:07, 9493.77it/s] 82%|████████▏ | 326633/400000 [00:33<00:07, 9612.67it/s] 82%|████████▏ | 327598/400000 [00:33<00:07, 9497.64it/s] 82%|████████▏ | 328626/400000 [00:33<00:07, 9717.86it/s] 82%|████████▏ | 329677/400000 [00:33<00:07, 9941.44it/s] 83%|████████▎ | 330675/400000 [00:33<00:06, 9906.90it/s] 83%|████████▎ | 331669/400000 [00:34<00:07, 9508.34it/s] 83%|████████▎ | 332625/400000 [00:34<00:07, 9259.63it/s] 83%|████████▎ | 333557/400000 [00:34<00:07, 9197.51it/s] 84%|████████▎ | 334481/400000 [00:34<00:07, 9133.71it/s] 84%|████████▍ | 335414/400000 [00:34<00:07, 9189.02it/s] 84%|████████▍ | 336469/400000 [00:34<00:06, 9558.41it/s] 84%|████████▍ | 337490/400000 [00:34<00:06, 9743.11it/s] 85%|████████▍ | 338469/400000 [00:34<00:06, 9395.35it/s] 85%|████████▍ | 339415/400000 [00:34<00:06, 9133.68it/s] 85%|████████▌ | 340368/400000 [00:34<00:06, 9247.33it/s] 85%|████████▌ | 341298/400000 [00:35<00:06, 9011.41it/s] 86%|████████▌ | 342204/400000 [00:35<00:06, 8977.61it/s] 86%|████████▌ | 343141/400000 [00:35<00:06, 9090.15it/s] 86%|████████▌ | 344066/400000 [00:35<00:06, 9135.61it/s] 86%|████████▋ | 345004/400000 [00:35<00:05, 9206.90it/s] 86%|████████▋ | 345983/400000 [00:35<00:05, 9372.23it/s] 87%|████████▋ | 346938/400000 [00:35<00:05, 9423.42it/s] 87%|████████▋ | 347973/400000 [00:35<00:05, 9681.46it/s] 87%|████████▋ | 348946/400000 [00:35<00:05, 9694.70it/s] 87%|████████▋ | 349918/400000 [00:35<00:05, 9660.34it/s] 88%|████████▊ | 350932/400000 [00:36<00:05, 9798.08it/s] 88%|████████▊ | 351957/400000 [00:36<00:04, 9928.04it/s] 88%|████████▊ | 352952/400000 [00:36<00:04, 9873.25it/s] 88%|████████▊ | 353941/400000 [00:36<00:04, 9767.68it/s] 89%|████████▊ | 354963/400000 [00:36<00:04, 9898.94it/s] 89%|████████▉ | 355972/400000 [00:36<00:04, 9953.64it/s] 89%|████████▉ | 356969/400000 [00:36<00:04, 9864.85it/s] 89%|████████▉ | 357968/400000 [00:36<00:04, 9899.99it/s] 90%|████████▉ | 358959/400000 [00:36<00:04, 9849.06it/s] 90%|████████▉ | 359945/400000 [00:36<00:04, 9785.40it/s] 90%|█████████ | 360924/400000 [00:37<00:04, 9463.11it/s] 90%|█████████ | 361873/400000 [00:37<00:04, 9438.80it/s] 91%|█████████ | 362897/400000 [00:37<00:03, 9664.67it/s] 91%|█████████ | 363867/400000 [00:37<00:03, 9456.76it/s] 91%|█████████ | 364816/400000 [00:37<00:03, 9448.38it/s] 91%|█████████▏| 365787/400000 [00:37<00:03, 9523.85it/s] 92%|█████████▏| 366825/400000 [00:37<00:03, 9764.97it/s] 92%|█████████▏| 367834/400000 [00:37<00:03, 9859.13it/s] 92%|█████████▏| 368822/400000 [00:37<00:03, 9751.00it/s] 92%|█████████▏| 369840/400000 [00:38<00:03, 9874.54it/s] 93%|█████████▎| 370908/400000 [00:38<00:02, 10101.50it/s] 93%|█████████▎| 371994/400000 [00:38<00:02, 10316.48it/s] 93%|█████████▎| 373029/400000 [00:38<00:02, 10191.63it/s] 94%|█████████▎| 374051/400000 [00:38<00:02, 9885.38it/s]  94%|█████████▍| 375044/400000 [00:38<00:02, 9673.44it/s] 94%|█████████▍| 376064/400000 [00:38<00:02, 9824.81it/s] 94%|█████████▍| 377134/400000 [00:38<00:02, 10071.18it/s] 95%|█████████▍| 378145/400000 [00:38<00:02, 9767.26it/s]  95%|█████████▍| 379127/400000 [00:38<00:02, 9338.30it/s] 95%|█████████▌| 380069/400000 [00:39<00:02, 9190.93it/s] 95%|█████████▌| 381003/400000 [00:39<00:02, 9232.88it/s] 95%|█████████▌| 381931/400000 [00:39<00:01, 9173.45it/s] 96%|█████████▌| 382891/400000 [00:39<00:01, 9297.23it/s] 96%|█████████▌| 383897/400000 [00:39<00:01, 9511.42it/s] 96%|█████████▌| 384995/400000 [00:39<00:01, 9908.70it/s] 97%|█████████▋| 386005/400000 [00:39<00:01, 9964.30it/s] 97%|█████████▋| 387063/400000 [00:39<00:01, 10138.36it/s] 97%|█████████▋| 388114/400000 [00:39<00:01, 10246.57it/s] 97%|█████████▋| 389142/400000 [00:39<00:01, 10143.14it/s] 98%|█████████▊| 390159/400000 [00:40<00:01, 9777.23it/s]  98%|█████████▊| 391142/400000 [00:40<00:00, 9698.28it/s] 98%|█████████▊| 392116/400000 [00:40<00:00, 9464.66it/s] 98%|█████████▊| 393075/400000 [00:40<00:00, 9501.64it/s] 99%|█████████▊| 394028/400000 [00:40<00:00, 9238.22it/s] 99%|█████████▊| 394976/400000 [00:40<00:00, 9307.27it/s] 99%|█████████▉| 395990/400000 [00:40<00:00, 9541.43it/s] 99%|█████████▉| 396979/400000 [00:40<00:00, 9642.11it/s] 99%|█████████▉| 397946/400000 [00:40<00:00, 9589.12it/s]100%|█████████▉| 398907/400000 [00:41<00:00, 9546.37it/s]100%|█████████▉| 399863/400000 [00:41<00:00, 9430.93it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9725.24it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1f2355b4e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010967752708385943 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011036317882729215 	 Accuracy: 60

  model saves at 60% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15745 out of table with 15681 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15745 out of table with 15681 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

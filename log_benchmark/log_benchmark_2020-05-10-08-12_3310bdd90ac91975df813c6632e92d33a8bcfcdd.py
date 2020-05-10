
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f2a94081f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 08:12:52.641835
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 08:12:52.645517
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 08:12:52.648529
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 08:12:52.651809
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2aa7df3320> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 358728.5625
Epoch 2/10

1/1 [==============================] - 0s 93ms/step - loss: 336219.0938
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 249438.2031
Epoch 4/10

1/1 [==============================] - 0s 90ms/step - loss: 187331.4062
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 134401.8906
Epoch 6/10

1/1 [==============================] - 0s 97ms/step - loss: 95201.2422
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 67794.1328
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 49171.0703
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 36597.4258
Epoch 10/10

1/1 [==============================] - 0s 107ms/step - loss: 28012.9551

  #### Inference Need return ypred, ytrue ######################### 
[[-0.2516195   0.45028     0.32246965  0.26484674 -0.23514724 -0.3110413
   0.7426675   0.35355031  0.61335003  0.9672282  -0.22083892 -0.6625387
  -0.2218551  -0.4241362   0.20041627  0.12086886 -0.80253077 -0.71682054
   0.12327765  0.24834694 -0.7244839  -0.8278235  -0.5077161   0.2275309
   0.1505346  -0.6414202   0.8759544   0.45676988  0.4897969  -0.22821698
  -0.11294097  1.066159    0.509168    0.80908054  0.05905883  0.66035074
  -0.79548633  0.922091    0.74577874  0.9671056  -0.1983137  -0.25326395
  -0.66649306 -0.64620304  0.67168146  0.56021506 -0.25284687 -0.0078907
  -0.19979858 -0.29262498 -0.00634125  0.01878509 -0.50322104  0.30926177
   0.17054078  0.2618559   0.1907168  -0.12218757 -0.43797922  0.40927103
  -0.23700894  0.03485769 -0.2771917   0.750522   -0.3372026   0.07275093
  -0.24524231 -0.36128566  0.5151295  -0.8992202   0.31253234  0.9553256
   0.41106534 -0.14995587 -0.04493926  0.10862795  0.09948814  0.53957355
   0.64602166  0.50997543 -0.05344345  0.2742191   0.6803522  -0.7970279
   0.27355087 -0.48326164 -0.7067645  -0.5629867   0.2888838  -0.57989496
   0.61788523  0.3217377   0.59579223  0.14453936  0.24605495 -0.10593574
   0.11313337 -0.36856323 -0.5104229  -0.41339123  0.6629845   0.13649738
  -0.28465876  0.08566394  0.7852138   0.30378762  0.01967746 -0.08725297
  -0.43749008  1.0577334  -0.9107218  -0.08758247 -0.27856115 -0.24640363
   0.5598642   0.9593012  -0.5534659  -0.05580673  0.63683105 -0.7534702
  -0.02589408  2.9134715   3.6788445   2.6385884   3.2161145   3.3736768
   3.06742     3.1004744   2.9386904   3.6654632   3.1929946   2.3102078
   3.067143    3.20975     2.8888636   3.1399803   2.5278227   3.9561787
   2.4608672   3.5492952   2.66901     3.3173392   3.977161    4.1411824
   2.7534702   3.4466574   2.7781887   3.5295587   4.222443    3.499117
   2.5486887   2.3425226   3.628274    3.1379073   3.9816635   3.073505
   3.1509027   3.7744775   4.2881136   2.9845202   3.568521    2.7956762
   2.8687482   2.869737    3.7597506   3.0115776   3.389317    3.2433734
   3.9398396   3.7573805   2.7043068   2.9666107   3.196537    2.2942066
   3.169181    2.801033    2.8904583   2.3850276   2.9129512   3.7732308
   0.68187994  1.0994096   2.128822    1.4187305   1.5159447   0.48073328
   0.88213694  0.6677804   1.1256971   0.86683124  1.7255425   0.7757311
   1.7961774   0.6502793   0.5406609   0.7256776   1.1803082   0.53849906
   1.1243956   0.6576315   1.7389107   1.9824953   0.9374482   1.6144288
   1.5373578   1.0449089   0.6786015   0.79591674  1.8952519   0.94189936
   1.1705661   0.42764282  1.6569746   0.7813284   0.8119861   0.9534612
   0.86723495  0.67725337  1.2505085   1.1078795   0.93008345  0.35383493
   1.4734997   1.1641256   1.0136921   1.4539422   1.5176591   0.46013618
   0.34510458  0.93634105  1.2491761   0.9780702   0.9282224   0.81835675
   0.9213691   0.7931183   0.6172497   1.1580216   1.0411078   0.90899986
   1.3765361   1.7400205   0.86046606  0.97824275  1.565393    0.3811413
   1.2060667   0.68324983  1.0564523   0.78513545  0.35224062  0.82943785
   1.3586829   1.3666387   1.2231991   1.4591086   1.4512721   0.45163906
   0.8075701   0.36036813  1.0072808   0.59716874  1.3912004   0.771012
   0.694836    0.7788875   0.57876855  1.8913713   1.2504342   1.2970538
   0.47374225  1.0457265   0.70372593  0.75076056  0.5728399   1.2534996
   0.8945451   1.177067    1.7177808   0.49069846  0.3744142   1.7737579
   0.5968912   0.3704015   0.985594    0.9473386   0.78075415  1.1636323
   0.35290015  1.075325    1.4971994   1.2232041   1.0497103   1.8053292
   0.47611535  1.2701805   0.4952488   1.9156337   0.45029998  0.93470585
   0.05427706  3.2751603   4.4605846   3.464673    3.725905    3.9389539
   3.679062    4.32115     3.5511994   4.565125    3.8140001   4.0375037
   3.4412742   3.9157052   3.7561078   4.343015    3.3773384   3.4859138
   4.473841    3.612701    4.026871    4.7483335   3.8035822   3.7243943
   4.0145044   4.4616227   4.515805    3.551292    4.082426    4.0325603
   3.9593992   4.5322075   4.0757184   4.619384    3.7699447   4.4367843
   4.6923413   3.8180375   4.667696    3.672213    4.493643    4.853561
   3.8850813   4.195292    3.5117893   4.4028087   4.334681    4.3115735
   5.063082    3.3312273   4.0934286   4.1041355   3.7536983   4.480526
   3.7182527   4.047966    4.6705365   3.8804674   3.817659    3.2560697
  -4.8188972  -2.7738369   2.7767215 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 08:13:01.678095
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.5936
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 08:13:01.681713
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9736.63
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 08:13:01.684767
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.9287
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 08:13:01.687727
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -870.984
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139820489818464
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139817977205536
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139817977206040
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139817977206544
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139817977207048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139817977207552

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2a80b76940> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.468709
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.433873
grad_step = 000002, loss = 0.403480
grad_step = 000003, loss = 0.370757
grad_step = 000004, loss = 0.335471
grad_step = 000005, loss = 0.302978
grad_step = 000006, loss = 0.283477
grad_step = 000007, loss = 0.270051
grad_step = 000008, loss = 0.251049
grad_step = 000009, loss = 0.235143
grad_step = 000010, loss = 0.222984
grad_step = 000011, loss = 0.213320
grad_step = 000012, loss = 0.205337
grad_step = 000013, loss = 0.197075
grad_step = 000014, loss = 0.186634
grad_step = 000015, loss = 0.174565
grad_step = 000016, loss = 0.163377
grad_step = 000017, loss = 0.154806
grad_step = 000018, loss = 0.147748
grad_step = 000019, loss = 0.139933
grad_step = 000020, loss = 0.131034
grad_step = 000021, loss = 0.122730
grad_step = 000022, loss = 0.115666
grad_step = 000023, loss = 0.109095
grad_step = 000024, loss = 0.102361
grad_step = 000025, loss = 0.095484
grad_step = 000026, loss = 0.088817
grad_step = 000027, loss = 0.082741
grad_step = 000028, loss = 0.077423
grad_step = 000029, loss = 0.072245
grad_step = 000030, loss = 0.066871
grad_step = 000031, loss = 0.061777
grad_step = 000032, loss = 0.057188
grad_step = 000033, loss = 0.052689
grad_step = 000034, loss = 0.048280
grad_step = 000035, loss = 0.043777
grad_step = 000036, loss = 0.039931
grad_step = 000037, loss = 0.036767
grad_step = 000038, loss = 0.033742
grad_step = 000039, loss = 0.030733
grad_step = 000040, loss = 0.027833
grad_step = 000041, loss = 0.025268
grad_step = 000042, loss = 0.022961
grad_step = 000043, loss = 0.020708
grad_step = 000044, loss = 0.018676
grad_step = 000045, loss = 0.016953
grad_step = 000046, loss = 0.015369
grad_step = 000047, loss = 0.013767
grad_step = 000048, loss = 0.012220
grad_step = 000049, loss = 0.010950
grad_step = 000050, loss = 0.009857
grad_step = 000051, loss = 0.008843
grad_step = 000052, loss = 0.007867
grad_step = 000053, loss = 0.006996
grad_step = 000054, loss = 0.006268
grad_step = 000055, loss = 0.005609
grad_step = 000056, loss = 0.005027
grad_step = 000057, loss = 0.004534
grad_step = 000058, loss = 0.004135
grad_step = 000059, loss = 0.003761
grad_step = 000060, loss = 0.003446
grad_step = 000061, loss = 0.003199
grad_step = 000062, loss = 0.003008
grad_step = 000063, loss = 0.002840
grad_step = 000064, loss = 0.002716
grad_step = 000065, loss = 0.002641
grad_step = 000066, loss = 0.002591
grad_step = 000067, loss = 0.002532
grad_step = 000068, loss = 0.002491
grad_step = 000069, loss = 0.002481
grad_step = 000070, loss = 0.002480
grad_step = 000071, loss = 0.002467
grad_step = 000072, loss = 0.002463
grad_step = 000073, loss = 0.002466
grad_step = 000074, loss = 0.002472
grad_step = 000075, loss = 0.002475
grad_step = 000076, loss = 0.002481
grad_step = 000077, loss = 0.002477
grad_step = 000078, loss = 0.002463
grad_step = 000079, loss = 0.002453
grad_step = 000080, loss = 0.002449
grad_step = 000081, loss = 0.002438
grad_step = 000082, loss = 0.002422
grad_step = 000083, loss = 0.002409
grad_step = 000084, loss = 0.002393
grad_step = 000085, loss = 0.002374
grad_step = 000086, loss = 0.002357
grad_step = 000087, loss = 0.002342
grad_step = 000088, loss = 0.002324
grad_step = 000089, loss = 0.002306
grad_step = 000090, loss = 0.002290
grad_step = 000091, loss = 0.002274
grad_step = 000092, loss = 0.002260
grad_step = 000093, loss = 0.002248
grad_step = 000094, loss = 0.002235
grad_step = 000095, loss = 0.002222
grad_step = 000096, loss = 0.002211
grad_step = 000097, loss = 0.002202
grad_step = 000098, loss = 0.002192
grad_step = 000099, loss = 0.002184
grad_step = 000100, loss = 0.002176
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002168
grad_step = 000102, loss = 0.002161
grad_step = 000103, loss = 0.002155
grad_step = 000104, loss = 0.002149
grad_step = 000105, loss = 0.002143
grad_step = 000106, loss = 0.002137
grad_step = 000107, loss = 0.002132
grad_step = 000108, loss = 0.002126
grad_step = 000109, loss = 0.002121
grad_step = 000110, loss = 0.002116
grad_step = 000111, loss = 0.002111
grad_step = 000112, loss = 0.002106
grad_step = 000113, loss = 0.002101
grad_step = 000114, loss = 0.002096
grad_step = 000115, loss = 0.002091
grad_step = 000116, loss = 0.002086
grad_step = 000117, loss = 0.002081
grad_step = 000118, loss = 0.002075
grad_step = 000119, loss = 0.002070
grad_step = 000120, loss = 0.002063
grad_step = 000121, loss = 0.002057
grad_step = 000122, loss = 0.002051
grad_step = 000123, loss = 0.002045
grad_step = 000124, loss = 0.002038
grad_step = 000125, loss = 0.002032
grad_step = 000126, loss = 0.002025
grad_step = 000127, loss = 0.002018
grad_step = 000128, loss = 0.002010
grad_step = 000129, loss = 0.002003
grad_step = 000130, loss = 0.001995
grad_step = 000131, loss = 0.001987
grad_step = 000132, loss = 0.001978
grad_step = 000133, loss = 0.001970
grad_step = 000134, loss = 0.001961
grad_step = 000135, loss = 0.001953
grad_step = 000136, loss = 0.001944
grad_step = 000137, loss = 0.001935
grad_step = 000138, loss = 0.001927
grad_step = 000139, loss = 0.001922
grad_step = 000140, loss = 0.001919
grad_step = 000141, loss = 0.001919
grad_step = 000142, loss = 0.001917
grad_step = 000143, loss = 0.001908
grad_step = 000144, loss = 0.001889
grad_step = 000145, loss = 0.001864
grad_step = 000146, loss = 0.001850
grad_step = 000147, loss = 0.001843
grad_step = 000148, loss = 0.001842
grad_step = 000149, loss = 0.001841
grad_step = 000150, loss = 0.001841
grad_step = 000151, loss = 0.001829
grad_step = 000152, loss = 0.001813
grad_step = 000153, loss = 0.001806
grad_step = 000154, loss = 0.001793
grad_step = 000155, loss = 0.001780
grad_step = 000156, loss = 0.001761
grad_step = 000157, loss = 0.001752
grad_step = 000158, loss = 0.001751
grad_step = 000159, loss = 0.001749
grad_step = 000160, loss = 0.001769
grad_step = 000161, loss = 0.001813
grad_step = 000162, loss = 0.001912
grad_step = 000163, loss = 0.001901
grad_step = 000164, loss = 0.001904
grad_step = 000165, loss = 0.001764
grad_step = 000166, loss = 0.001692
grad_step = 000167, loss = 0.001748
grad_step = 000168, loss = 0.001790
grad_step = 000169, loss = 0.001796
grad_step = 000170, loss = 0.001733
grad_step = 000171, loss = 0.001701
grad_step = 000172, loss = 0.001678
grad_step = 000173, loss = 0.001754
grad_step = 000174, loss = 0.001774
grad_step = 000175, loss = 0.001708
grad_step = 000176, loss = 0.001723
grad_step = 000177, loss = 0.001709
grad_step = 000178, loss = 0.001709
grad_step = 000179, loss = 0.001730
grad_step = 000180, loss = 0.001668
grad_step = 000181, loss = 0.001668
grad_step = 000182, loss = 0.001692
grad_step = 000183, loss = 0.001665
grad_step = 000184, loss = 0.001678
grad_step = 000185, loss = 0.001625
grad_step = 000186, loss = 0.001656
grad_step = 000187, loss = 0.001643
grad_step = 000188, loss = 0.001645
grad_step = 000189, loss = 0.001633
grad_step = 000190, loss = 0.001616
grad_step = 000191, loss = 0.001617
grad_step = 000192, loss = 0.001619
grad_step = 000193, loss = 0.001623
grad_step = 000194, loss = 0.001603
grad_step = 000195, loss = 0.001607
grad_step = 000196, loss = 0.001595
grad_step = 000197, loss = 0.001590
grad_step = 000198, loss = 0.001598
grad_step = 000199, loss = 0.001592
grad_step = 000200, loss = 0.001586
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001580
grad_step = 000202, loss = 0.001577
grad_step = 000203, loss = 0.001568
grad_step = 000204, loss = 0.001564
grad_step = 000205, loss = 0.001568
grad_step = 000206, loss = 0.001563
grad_step = 000207, loss = 0.001561
grad_step = 000208, loss = 0.001561
grad_step = 000209, loss = 0.001561
grad_step = 000210, loss = 0.001563
grad_step = 000211, loss = 0.001562
grad_step = 000212, loss = 0.001571
grad_step = 000213, loss = 0.001584
grad_step = 000214, loss = 0.001607
grad_step = 000215, loss = 0.001637
grad_step = 000216, loss = 0.001679
grad_step = 000217, loss = 0.001704
grad_step = 000218, loss = 0.001707
grad_step = 000219, loss = 0.001648
grad_step = 000220, loss = 0.001574
grad_step = 000221, loss = 0.001525
grad_step = 000222, loss = 0.001526
grad_step = 000223, loss = 0.001559
grad_step = 000224, loss = 0.001589
grad_step = 000225, loss = 0.001595
grad_step = 000226, loss = 0.001569
grad_step = 000227, loss = 0.001526
grad_step = 000228, loss = 0.001501
grad_step = 000229, loss = 0.001506
grad_step = 000230, loss = 0.001528
grad_step = 000231, loss = 0.001543
grad_step = 000232, loss = 0.001540
grad_step = 000233, loss = 0.001525
grad_step = 000234, loss = 0.001507
grad_step = 000235, loss = 0.001490
grad_step = 000236, loss = 0.001482
grad_step = 000237, loss = 0.001487
grad_step = 000238, loss = 0.001497
grad_step = 000239, loss = 0.001505
grad_step = 000240, loss = 0.001506
grad_step = 000241, loss = 0.001504
grad_step = 000242, loss = 0.001501
grad_step = 000243, loss = 0.001494
grad_step = 000244, loss = 0.001485
grad_step = 000245, loss = 0.001476
grad_step = 000246, loss = 0.001469
grad_step = 000247, loss = 0.001464
grad_step = 000248, loss = 0.001460
grad_step = 000249, loss = 0.001456
grad_step = 000250, loss = 0.001452
grad_step = 000251, loss = 0.001450
grad_step = 000252, loss = 0.001449
grad_step = 000253, loss = 0.001448
grad_step = 000254, loss = 0.001448
grad_step = 000255, loss = 0.001449
grad_step = 000256, loss = 0.001456
grad_step = 000257, loss = 0.001475
grad_step = 000258, loss = 0.001522
grad_step = 000259, loss = 0.001627
grad_step = 000260, loss = 0.001809
grad_step = 000261, loss = 0.002059
grad_step = 000262, loss = 0.002076
grad_step = 000263, loss = 0.001768
grad_step = 000264, loss = 0.001449
grad_step = 000265, loss = 0.001519
grad_step = 000266, loss = 0.001750
grad_step = 000267, loss = 0.001670
grad_step = 000268, loss = 0.001437
grad_step = 000269, loss = 0.001496
grad_step = 000270, loss = 0.001643
grad_step = 000271, loss = 0.001546
grad_step = 000272, loss = 0.001423
grad_step = 000273, loss = 0.001499
grad_step = 000274, loss = 0.001574
grad_step = 000275, loss = 0.001472
grad_step = 000276, loss = 0.001414
grad_step = 000277, loss = 0.001470
grad_step = 000278, loss = 0.001489
grad_step = 000279, loss = 0.001434
grad_step = 000280, loss = 0.001399
grad_step = 000281, loss = 0.001437
grad_step = 000282, loss = 0.001453
grad_step = 000283, loss = 0.001399
grad_step = 000284, loss = 0.001386
grad_step = 000285, loss = 0.001422
grad_step = 000286, loss = 0.001413
grad_step = 000287, loss = 0.001378
grad_step = 000288, loss = 0.001378
grad_step = 000289, loss = 0.001398
grad_step = 000290, loss = 0.001389
grad_step = 000291, loss = 0.001364
grad_step = 000292, loss = 0.001368
grad_step = 000293, loss = 0.001381
grad_step = 000294, loss = 0.001369
grad_step = 000295, loss = 0.001353
grad_step = 000296, loss = 0.001356
grad_step = 000297, loss = 0.001361
grad_step = 000298, loss = 0.001355
grad_step = 000299, loss = 0.001344
grad_step = 000300, loss = 0.001341
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001344
grad_step = 000302, loss = 0.001342
grad_step = 000303, loss = 0.001333
grad_step = 000304, loss = 0.001328
grad_step = 000305, loss = 0.001329
grad_step = 000306, loss = 0.001328
grad_step = 000307, loss = 0.001324
grad_step = 000308, loss = 0.001318
grad_step = 000309, loss = 0.001314
grad_step = 000310, loss = 0.001313
grad_step = 000311, loss = 0.001312
grad_step = 000312, loss = 0.001309
grad_step = 000313, loss = 0.001304
grad_step = 000314, loss = 0.001300
grad_step = 000315, loss = 0.001296
grad_step = 000316, loss = 0.001294
grad_step = 000317, loss = 0.001293
grad_step = 000318, loss = 0.001290
grad_step = 000319, loss = 0.001286
grad_step = 000320, loss = 0.001282
grad_step = 000321, loss = 0.001278
grad_step = 000322, loss = 0.001274
grad_step = 000323, loss = 0.001270
grad_step = 000324, loss = 0.001267
grad_step = 000325, loss = 0.001265
grad_step = 000326, loss = 0.001262
grad_step = 000327, loss = 0.001260
grad_step = 000328, loss = 0.001258
grad_step = 000329, loss = 0.001257
grad_step = 000330, loss = 0.001258
grad_step = 000331, loss = 0.001260
grad_step = 000332, loss = 0.001265
grad_step = 000333, loss = 0.001270
grad_step = 000334, loss = 0.001268
grad_step = 000335, loss = 0.001265
grad_step = 000336, loss = 0.001256
grad_step = 000337, loss = 0.001254
grad_step = 000338, loss = 0.001253
grad_step = 000339, loss = 0.001255
grad_step = 000340, loss = 0.001256
grad_step = 000341, loss = 0.001250
grad_step = 000342, loss = 0.001238
grad_step = 000343, loss = 0.001222
grad_step = 000344, loss = 0.001213
grad_step = 000345, loss = 0.001216
grad_step = 000346, loss = 0.001226
grad_step = 000347, loss = 0.001242
grad_step = 000348, loss = 0.001257
grad_step = 000349, loss = 0.001295
grad_step = 000350, loss = 0.001319
grad_step = 000351, loss = 0.001378
grad_step = 000352, loss = 0.001368
grad_step = 000353, loss = 0.001336
grad_step = 000354, loss = 0.001268
grad_step = 000355, loss = 0.001213
grad_step = 000356, loss = 0.001203
grad_step = 000357, loss = 0.001220
grad_step = 000358, loss = 0.001250
grad_step = 000359, loss = 0.001255
grad_step = 000360, loss = 0.001215
grad_step = 000361, loss = 0.001170
grad_step = 000362, loss = 0.001159
grad_step = 000363, loss = 0.001177
grad_step = 000364, loss = 0.001192
grad_step = 000365, loss = 0.001185
grad_step = 000366, loss = 0.001184
grad_step = 000367, loss = 0.001207
grad_step = 000368, loss = 0.001217
grad_step = 000369, loss = 0.001217
grad_step = 000370, loss = 0.001186
grad_step = 000371, loss = 0.001164
grad_step = 000372, loss = 0.001155
grad_step = 000373, loss = 0.001148
grad_step = 000374, loss = 0.001141
grad_step = 000375, loss = 0.001130
grad_step = 000376, loss = 0.001127
grad_step = 000377, loss = 0.001136
grad_step = 000378, loss = 0.001151
grad_step = 000379, loss = 0.001162
grad_step = 000380, loss = 0.001167
grad_step = 000381, loss = 0.001170
grad_step = 000382, loss = 0.001186
grad_step = 000383, loss = 0.001189
grad_step = 000384, loss = 0.001191
grad_step = 000385, loss = 0.001160
grad_step = 000386, loss = 0.001128
grad_step = 000387, loss = 0.001100
grad_step = 000388, loss = 0.001090
grad_step = 000389, loss = 0.001093
grad_step = 000390, loss = 0.001100
grad_step = 000391, loss = 0.001107
grad_step = 000392, loss = 0.001110
grad_step = 000393, loss = 0.001114
grad_step = 000394, loss = 0.001110
grad_step = 000395, loss = 0.001104
grad_step = 000396, loss = 0.001092
grad_step = 000397, loss = 0.001078
grad_step = 000398, loss = 0.001065
grad_step = 000399, loss = 0.001056
grad_step = 000400, loss = 0.001053
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001051
grad_step = 000402, loss = 0.001049
grad_step = 000403, loss = 0.001046
grad_step = 000404, loss = 0.001041
grad_step = 000405, loss = 0.001035
grad_step = 000406, loss = 0.001031
grad_step = 000407, loss = 0.001029
grad_step = 000408, loss = 0.001028
grad_step = 000409, loss = 0.001028
grad_step = 000410, loss = 0.001033
grad_step = 000411, loss = 0.001050
grad_step = 000412, loss = 0.001096
grad_step = 000413, loss = 0.001235
grad_step = 000414, loss = 0.001349
grad_step = 000415, loss = 0.001594
grad_step = 000416, loss = 0.001374
grad_step = 000417, loss = 0.001146
grad_step = 000418, loss = 0.001030
grad_step = 000419, loss = 0.001130
grad_step = 000420, loss = 0.001244
grad_step = 000421, loss = 0.001130
grad_step = 000422, loss = 0.001017
grad_step = 000423, loss = 0.001055
grad_step = 000424, loss = 0.001116
grad_step = 000425, loss = 0.001087
grad_step = 000426, loss = 0.001000
grad_step = 000427, loss = 0.001003
grad_step = 000428, loss = 0.001069
grad_step = 000429, loss = 0.001047
grad_step = 000430, loss = 0.000982
grad_step = 000431, loss = 0.000978
grad_step = 000432, loss = 0.001015
grad_step = 000433, loss = 0.001012
grad_step = 000434, loss = 0.000977
grad_step = 000435, loss = 0.000970
grad_step = 000436, loss = 0.000978
grad_step = 000437, loss = 0.000968
grad_step = 000438, loss = 0.000956
grad_step = 000439, loss = 0.000964
grad_step = 000440, loss = 0.000975
grad_step = 000441, loss = 0.000961
grad_step = 000442, loss = 0.000952
grad_step = 000443, loss = 0.000961
grad_step = 000444, loss = 0.000964
grad_step = 000445, loss = 0.000948
grad_step = 000446, loss = 0.000934
grad_step = 000447, loss = 0.000932
grad_step = 000448, loss = 0.000927
grad_step = 000449, loss = 0.000914
grad_step = 000450, loss = 0.000907
grad_step = 000451, loss = 0.000910
grad_step = 000452, loss = 0.000911
grad_step = 000453, loss = 0.000910
grad_step = 000454, loss = 0.000913
grad_step = 000455, loss = 0.000931
grad_step = 000456, loss = 0.000956
grad_step = 000457, loss = 0.001010
grad_step = 000458, loss = 0.001032
grad_step = 000459, loss = 0.001082
grad_step = 000460, loss = 0.000982
grad_step = 000461, loss = 0.000913
grad_step = 000462, loss = 0.000878
grad_step = 000463, loss = 0.000894
grad_step = 000464, loss = 0.000932
grad_step = 000465, loss = 0.000932
grad_step = 000466, loss = 0.000909
grad_step = 000467, loss = 0.000874
grad_step = 000468, loss = 0.000862
grad_step = 000469, loss = 0.000879
grad_step = 000470, loss = 0.000897
grad_step = 000471, loss = 0.000899
grad_step = 000472, loss = 0.000874
grad_step = 000473, loss = 0.000854
grad_step = 000474, loss = 0.000845
grad_step = 000475, loss = 0.000842
grad_step = 000476, loss = 0.000840
grad_step = 000477, loss = 0.000842
grad_step = 000478, loss = 0.000853
grad_step = 000479, loss = 0.000859
grad_step = 000480, loss = 0.000871
grad_step = 000481, loss = 0.000863
grad_step = 000482, loss = 0.000870
grad_step = 000483, loss = 0.000862
grad_step = 000484, loss = 0.000858
grad_step = 000485, loss = 0.000842
grad_step = 000486, loss = 0.000832
grad_step = 000487, loss = 0.000825
grad_step = 000488, loss = 0.000825
grad_step = 000489, loss = 0.000829
grad_step = 000490, loss = 0.000835
grad_step = 000491, loss = 0.000844
grad_step = 000492, loss = 0.000844
grad_step = 000493, loss = 0.000850
grad_step = 000494, loss = 0.000835
grad_step = 000495, loss = 0.000825
grad_step = 000496, loss = 0.000804
grad_step = 000497, loss = 0.000787
grad_step = 000498, loss = 0.000773
grad_step = 000499, loss = 0.000767
grad_step = 000500, loss = 0.000767
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000771
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

  date_run                              2020-05-10 08:13:19.589921
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.218252
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 08:13:19.595910
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.125378
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 08:13:19.603552
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.118184
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 08:13:19.608640
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.905159
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
0   2020-05-10 08:12:52.641835  ...    mean_absolute_error
1   2020-05-10 08:12:52.645517  ...     mean_squared_error
2   2020-05-10 08:12:52.648529  ...  median_absolute_error
3   2020-05-10 08:12:52.651809  ...               r2_score
4   2020-05-10 08:13:01.678095  ...    mean_absolute_error
5   2020-05-10 08:13:01.681713  ...     mean_squared_error
6   2020-05-10 08:13:01.684767  ...  median_absolute_error
7   2020-05-10 08:13:01.687727  ...               r2_score
8   2020-05-10 08:13:19.589921  ...    mean_absolute_error
9   2020-05-10 08:13:19.595910  ...     mean_squared_error
10  2020-05-10 08:13:19.603552  ...  median_absolute_error
11  2020-05-10 08:13:19.608640  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:19, 124219.51it/s] 80%|████████  | 7954432/9912422 [00:00<00:11, 177330.45it/s]9920512it [00:00, 40286630.23it/s]                           
0it [00:00, ?it/s]32768it [00:00, 606553.48it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470837.22it/s]1654784it [00:00, 11721704.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186081.37it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac5be6d780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fabf95b1978> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac5be24e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fabf95b1d68> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac5be24e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac0e81ecc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac5be6df98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac02cd3438> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac5be6df98> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fac02cd3438> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fabf95b20b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2317d0a1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4de903e2d2f96b3059f13893c0c82dbd5e9cac464eb41de16a0788dcc8d5099a
  Stored in directory: /tmp/pip-ephem-wheel-cache-iwavtue3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f230de78048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2998272/17464789 [====>.........................] - ETA: 0s
10043392/17464789 [================>.............] - ETA: 0s
16605184/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 08:14:44.420565: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 08:14:44.425503: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-10 08:14:44.425699: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e7fff1fd00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 08:14:44.425714: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7893 - accuracy: 0.4920 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7791 - accuracy: 0.4927
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8008 - accuracy: 0.4913
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7263 - accuracy: 0.4961
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7341 - accuracy: 0.4956
11000/25000 [============>.................] - ETA: 3s - loss: 7.7182 - accuracy: 0.4966
12000/25000 [=============>................] - ETA: 3s - loss: 7.7356 - accuracy: 0.4955
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7339 - accuracy: 0.4956
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7367 - accuracy: 0.4954
15000/25000 [=================>............] - ETA: 2s - loss: 7.7228 - accuracy: 0.4963
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7021 - accuracy: 0.4977
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7171 - accuracy: 0.4967
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7058 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
25000/25000 [==============================] - 7s 265us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 08:14:57.218051
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 08:14:57.218051  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 08:15:02.709693: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 08:15:02.714219: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-10 08:15:02.714339: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5582d61cf980 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 08:15:02.714351: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f73d1efed30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3523 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.3911 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f73c72a6f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6697 - accuracy: 0.4998
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6935 - accuracy: 0.4983
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7109 - accuracy: 0.4971
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6574 - accuracy: 0.5006
11000/25000 [============>.................] - ETA: 3s - loss: 7.6415 - accuracy: 0.5016
12000/25000 [=============>................] - ETA: 2s - loss: 7.6449 - accuracy: 0.5014
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 2s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6935 - accuracy: 0.4983
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6763 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6951 - accuracy: 0.4981
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6840 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 6s 258us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f73c41be438> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:03:10, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:58:09, 16.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:31:55, 22.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:22:50, 32.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:09:12, 46.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<3:35:06, 66.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:29:40, 94.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<1:44:13, 135kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:12:47, 192kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.7M/862M [00:01<50:43, 274kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<35:28, 390kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.3M/862M [00:02<24:45, 555kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:02<17:23, 787kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.9M/862M [00:02<12:10, 1.12MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<08:36, 1.57MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:42, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:35, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<08:16, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:36, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<04:50, 2.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<08:55, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<07:51, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<05:53, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:51, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<07:29, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<05:54, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<04:16, 3.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<12:45:29, 17.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<8:56:57, 24.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<6:15:29, 35.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<4:25:10, 49.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<3:08:22, 69.8kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<2:12:24, 99.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:13<1:32:28, 142kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<2:28:41, 88.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<1:45:23, 124kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:14<1:13:57, 176kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<54:41, 238kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<40:57, 318kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<29:12, 445kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:16<20:35, 630kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<19:32, 663kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<15:00, 862kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:18<10:49, 1.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<10:32, 1.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<09:59, 1.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<07:38, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:24, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<06:30, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<04:51, 2.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<06:22, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<07:03, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<05:34, 2.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:56, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:28, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:06, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:22, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:04, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:48, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:16, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:16, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:00, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:41, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:38, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:17, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<03:49, 3.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:21:49, 151kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<58:31, 211kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<41:10, 299kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<31:35, 388kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<24:36, 498kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<17:44, 690kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<12:31, 975kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<17:33, 694kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<13:33, 899kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<09:44, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:38, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:25, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<07:16, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:12, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<15:42, 767kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:40, 882kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<10:09, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<07:13, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<14:45, 812kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<12:58, 923kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:37, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<06:52, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:44, 1.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:02, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:33, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<05:27, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:15, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:10, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:14, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:30, 2.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<09:33, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:04, 1.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:57, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:59, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<32:54, 356kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<25:30, 459kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<18:28, 633kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<13:01, 894kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<26:30, 439kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<20:54, 556kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<15:08, 768kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<10:40, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<31:01, 373kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<24:16, 477kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<17:29, 661kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<12:22, 931kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<13:26, 856kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:50, 972kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:49, 1.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:18, 1.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<10:46, 1.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<10:04, 1.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:40, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<05:29, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<20:49, 546kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<17:04, 665kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:33, 904kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<08:54, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<23:06, 489kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<18:33, 608kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<13:35, 830kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<09:37, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<23:30, 477kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<18:56, 592kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<13:47, 813kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<09:44, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<16:00, 697kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<13:35, 820kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<10:01, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<07:09, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<09:41, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<09:07, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:53, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:56, 2.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<10:32, 1.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:42, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:19, 1.50MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<10:23, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<09:35, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:17, 1.50MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:12, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<26:54, 404kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<20:57, 519kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<15:10, 716kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<12:21, 875kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<11:02, 979kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:14, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<05:53, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:50, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:29, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:30, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<04:40, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<27:40, 386kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<21:45, 490kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<15:44, 677kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<11:07, 954kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<20:14, 524kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<16:09, 656kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<11:47, 897kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<09:57, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<09:18, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<07:05, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<05:04, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<14:59, 698kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:48, 817kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<09:27, 1.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<06:43, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:23, 1.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<09:18, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:55, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<04:57, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<11:18, 913kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:50, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<07:23, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<05:16, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<2:30:28, 68.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<1:47:17, 95.6kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<1:15:30, 136kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:35<52:42, 193kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<1:50:23, 92.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<1:19:30, 128kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<56:09, 181kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<39:15, 258kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<39:44, 255kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<29:57, 338kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<21:28, 471kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<15:04, 667kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<31:17, 321kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<23:51, 421kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<17:09, 585kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<13:36, 734kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<11:43, 851kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:41, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<06:10, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<10:09, 976kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<09:12, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:57, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<04:58, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<24:28, 402kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<19:18, 510kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<14:01, 701kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<09:54, 987kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<21:55, 446kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<17:25, 561kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<12:37, 773kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<08:55, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<11:13, 864kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<09:41, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:10, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<05:06, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<16:14, 593kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<15:37, 617kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<11:59, 803kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<08:36, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<08:01, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:24, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:36, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:30, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:51, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:35, 2.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:19, 2.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<27:59, 337kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<21:16, 443kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<15:18, 615kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<12:13, 765kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<10:37, 881kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:52, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<05:35, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<11:15, 825kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:29, 978kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<07:03, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:28, 1.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:58, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:24, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:38, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:36, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:59, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:37, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:19, 2.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<08:15, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:41, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:47, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:08, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:05, 992kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:16, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:10, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:25, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<07:35, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<07:07, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:22, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:52, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:38, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:31, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:01, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:36, 2.44MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<22:03, 400kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<17:17, 509kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<12:28, 705kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<08:49, 993kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<10:05, 867kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<08:56, 978kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:43, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:46, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<15:13, 570kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<12:03, 719kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<08:42, 993kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<07:36, 1.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<07:08, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:26, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:53, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<20:58, 407kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<15:45, 541kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<11:20, 751kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<08:01, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<12:05, 700kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:10, 832kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<07:32, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<05:20, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<2:05:02, 67.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<1:29:12, 94.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<1:02:46, 134kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<43:45, 190kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<3:11:25, 43.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<2:15:40, 61.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<1:35:14, 87.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<1:06:20, 125kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<3:25:09, 40.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<2:25:13, 56.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<1:41:52, 80.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<1:10:58, 115kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<2:08:31, 63.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<1:31:03, 89.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<1:03:49, 128kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<45:59, 177kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<33:38, 241kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<23:49, 340kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<16:42, 483kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<16:45, 481kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<13:14, 608kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<09:33, 841kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<06:48, 1.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:42, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:54, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:11, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:56, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:56, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:49, 2.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<02:47, 2.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:29, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:18, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:03, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:56, 2.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:46, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:30, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:12, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:12, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:50, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:41, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:27, 2.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:01, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:16, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:22, 3.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<11:23, 665kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<09:23, 806kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:54, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:03, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:39, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:17, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:14, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:21, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:23, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:35, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:53, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:03, 2.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:20, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:42, 1.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:55, 2.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:14, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:38, 1.99MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:52, 2.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:04, 3.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<10:58, 653kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<09:01, 794kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:34, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:04<04:41, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:32, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:52, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:26, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:15, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:15, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:17, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:27, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:01, 1.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:09, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:03, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:50, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:59, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:04, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:27, 2.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:17, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:16, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:23, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:28, 2.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:47, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:29, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:33, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:35, 2.58MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<04:56, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:55, 1.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:50, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:45, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:07, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:57, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:52, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:49, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<04:27, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:32, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:34, 2.51MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:37, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<04:58, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:54, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:48, 2.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:50, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<05:02, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:52, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:47, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:01, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:28, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:31, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:33, 2.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:43, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:50, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:44, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:42, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:43, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:53, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:48, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:45, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:48, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:53, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:44, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:40, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:55, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:56, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:46, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:44, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:35, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:08, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:13, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:20, 2.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:28, 1.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:54, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:05, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:14, 2.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:59, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:14, 1.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:19, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:23, 2.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:35, 1.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:23, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:25, 1.68MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:28, 2.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:43, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:55, 1.45MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:02, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:11, 2.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:43, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:58, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:04, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:13, 2.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:25, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:27, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:24, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:42, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:54, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:16, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:27, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:33, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:51, 2.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<09:49, 553kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<09:04, 598kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<06:49, 795kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:53, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:36, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<05:26, 985kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:21, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:17, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:22, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:38, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:33, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:36, 1.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:38, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:06, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:07, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:21, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:27, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:47, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<13:03, 395kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<11:05, 465kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<08:12, 626kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<05:49, 879kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<04:07, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<1:12:46, 69.9kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<52:41, 96.5kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<37:17, 136kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<26:03, 194kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<19:21, 259kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<15:19, 327kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<11:05, 451kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<07:50, 635kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<06:41, 739kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<06:25, 770kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:51, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<03:29, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:30, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<55:14, 88.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<40:21, 121kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<28:37, 170kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<20:03, 242kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<14:03, 343kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<12:54, 372kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<10:42, 449kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:53, 608kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<05:35, 854kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<03:59, 1.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<08:59, 527kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<07:58, 594kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:55, 798kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<04:15, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<03:02, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<05:05, 916kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<05:13, 894kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:59, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:53, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<02:05, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<04:39, 990kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<04:51, 946kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:43, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:42, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:02, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:37, 1.25MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:54, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<02:07, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:39, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:18, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:41, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:58, 2.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:30, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:17, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:36, 1.68MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:55, 2.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:26, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:12, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:36, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:54, 2.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:27, 1.74MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:12, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:32, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:51, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:23, 3.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:49, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:08, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:28, 1.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:38, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:57, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:28, 2.81MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:12, 3.42MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<08:18, 495kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:37, 621kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:47, 855kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<03:23, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:02, 1.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:08, 975kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:14, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<02:20, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:39, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:10, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:34, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:53, 2.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:22, 2.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:12, 928kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:14, 923kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:14, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:21, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:42, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:57, 970kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:01, 955kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:04, 1.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:13, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:37, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:06, 916kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:06, 919kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:10, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:17, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:35, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:02, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:22, 1.56MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:43, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:09, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:42, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:10, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:34, 2.27MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:04, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:32, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:02, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:29, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:01, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:29, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:58, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:28, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:04, 3.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:33, 616kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:56, 693kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:42, 921kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:37, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:57, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:35, 936kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:28, 965kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:38, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:54, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:16, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:35, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:01, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:27, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:04, 2.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<30:31, 105kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<22:13, 145kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<15:42, 204kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<11:00, 289kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<07:39, 411kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<16:25, 192kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<12:13, 257kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<08:44, 358kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<06:07, 507kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:07, 600kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:28, 687kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:20, 918kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:22, 1.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:35, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:47, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:09, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:33, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:54, 1.54MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:03, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:36, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:10, 2.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<02:00, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:11, 1.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:41, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:44, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:59, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:38, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:15, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:56, 2.91MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:35, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:19, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:56, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:30, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:24, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:01, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:23, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:38, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:18, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:56, 2.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:36, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:38, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:14, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:53, 2.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:45, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:48, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:22, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:59, 2.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:34, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:31, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:09, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:49, 2.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:46, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:46, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:22, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:58, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:34, 874kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<02:12, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:39, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:11, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:33, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:26, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:04, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:45, 2.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<18:26, 114kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<13:49, 152kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<09:51, 213kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<06:52, 302kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<05:09, 396kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<04:02, 504kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<02:55, 694kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<02:01, 978kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<06:05, 323kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<04:33, 431kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<03:14, 602kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:33, 743kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:20, 814kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:44, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:15, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:14, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:12, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:55, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:55, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:57, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:45, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<00:32, 3.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:04, 815kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:47, 947kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:19, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:10, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:07, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:50, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:36, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:00, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:59, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:45, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:45, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:05, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:53, 1.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:39, 2.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:48, 1.77MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:48, 1.74MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:42, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:33, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:35, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:39, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:30, 2.52MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:21, 3.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:48, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:47, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:35, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:24, 2.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:16, 503kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:48, 633kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:17, 868kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:02, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:55, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:41, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:30, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:36, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:21, 2.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:22, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:17, 2.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:11, 3.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<02:11, 304kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:38, 402kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:09, 559kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:50, 704kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:48, 728kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:36, 959kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:25, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:24, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:22, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:17, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:16, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:04, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<01:41, 149kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<01:12, 204kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:48, 288kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:28, 385kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:23, 455kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:16, 618kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:10, 866kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 981kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:05, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.58MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.04MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 922/400000 [00:00<00:43, 9216.76it/s]  0%|          | 1836/400000 [00:00<00:43, 9192.28it/s]  1%|          | 2710/400000 [00:00<00:43, 9050.77it/s]  1%|          | 3593/400000 [00:00<00:44, 8981.48it/s]  1%|          | 4448/400000 [00:00<00:44, 8846.83it/s]  1%|▏         | 5308/400000 [00:00<00:45, 8769.85it/s]  2%|▏         | 6141/400000 [00:00<00:45, 8632.33it/s]  2%|▏         | 6947/400000 [00:00<00:46, 8449.71it/s]  2%|▏         | 7798/400000 [00:00<00:46, 8466.03it/s]  2%|▏         | 8649/400000 [00:01<00:46, 8478.03it/s]  2%|▏         | 9515/400000 [00:01<00:45, 8531.27it/s]  3%|▎         | 10381/400000 [00:01<00:45, 8569.47it/s]  3%|▎         | 11258/400000 [00:01<00:45, 8626.05it/s]  3%|▎         | 12113/400000 [00:01<00:45, 8575.01it/s]  3%|▎         | 12965/400000 [00:01<00:45, 8486.55it/s]  3%|▎         | 13850/400000 [00:01<00:44, 8591.40it/s]  4%|▎         | 14720/400000 [00:01<00:44, 8621.82it/s]  4%|▍         | 15581/400000 [00:01<00:45, 8492.97it/s]  4%|▍         | 16430/400000 [00:01<00:45, 8430.24it/s]  4%|▍         | 17303/400000 [00:02<00:44, 8516.94it/s]  5%|▍         | 18155/400000 [00:02<00:45, 8476.06it/s]  5%|▍         | 19020/400000 [00:02<00:44, 8526.50it/s]  5%|▍         | 19873/400000 [00:02<00:45, 8443.53it/s]  5%|▌         | 20777/400000 [00:02<00:44, 8613.46it/s]  5%|▌         | 21640/400000 [00:02<00:43, 8608.39it/s]  6%|▌         | 22502/400000 [00:02<00:43, 8593.64it/s]  6%|▌         | 23362/400000 [00:02<00:44, 8451.16it/s]  6%|▌         | 24208/400000 [00:02<00:44, 8428.74it/s]  6%|▋         | 25052/400000 [00:02<00:44, 8420.45it/s]  6%|▋         | 25895/400000 [00:03<00:44, 8371.89it/s]  7%|▋         | 26770/400000 [00:03<00:44, 8481.00it/s]  7%|▋         | 27622/400000 [00:03<00:43, 8490.37it/s]  7%|▋         | 28472/400000 [00:03<00:44, 8358.09it/s]  7%|▋         | 29333/400000 [00:03<00:43, 8429.83it/s]  8%|▊         | 30199/400000 [00:03<00:43, 8495.91it/s]  8%|▊         | 31053/400000 [00:03<00:43, 8508.89it/s]  8%|▊         | 31905/400000 [00:03<00:43, 8478.24it/s]  8%|▊         | 32754/400000 [00:03<00:43, 8383.77it/s]  8%|▊         | 33650/400000 [00:03<00:42, 8547.95it/s]  9%|▊         | 34506/400000 [00:04<00:43, 8464.26it/s]  9%|▉         | 35362/400000 [00:04<00:42, 8490.08it/s]  9%|▉         | 36218/400000 [00:04<00:42, 8508.52it/s]  9%|▉         | 37070/400000 [00:04<00:45, 8026.33it/s]  9%|▉         | 37879/400000 [00:04<00:45, 8016.04it/s] 10%|▉         | 38778/400000 [00:04<00:43, 8283.69it/s] 10%|▉         | 39668/400000 [00:04<00:42, 8459.21it/s] 10%|█         | 40535/400000 [00:04<00:42, 8521.09it/s] 10%|█         | 41441/400000 [00:04<00:41, 8675.13it/s] 11%|█         | 42341/400000 [00:04<00:40, 8768.05it/s] 11%|█         | 43232/400000 [00:05<00:40, 8808.82it/s] 11%|█         | 44115/400000 [00:05<00:40, 8795.78it/s] 11%|█▏        | 45001/400000 [00:05<00:40, 8814.22it/s] 11%|█▏        | 45884/400000 [00:05<00:45, 7804.69it/s] 12%|█▏        | 46766/400000 [00:05<00:43, 8083.60it/s] 12%|█▏        | 47644/400000 [00:05<00:42, 8280.08it/s] 12%|█▏        | 48488/400000 [00:05<00:42, 8325.39it/s] 12%|█▏        | 49411/400000 [00:05<00:40, 8575.32it/s] 13%|█▎        | 50401/400000 [00:05<00:39, 8933.75it/s] 13%|█▎        | 51305/400000 [00:06<00:38, 8958.61it/s] 13%|█▎        | 52306/400000 [00:06<00:37, 9248.80it/s] 13%|█▎        | 53242/400000 [00:06<00:37, 9279.92it/s] 14%|█▎        | 54176/400000 [00:06<00:38, 9070.40it/s] 14%|█▍        | 55088/400000 [00:06<00:39, 8646.90it/s] 14%|█▍        | 55961/400000 [00:06<00:40, 8514.79it/s] 14%|█▍        | 56819/400000 [00:06<00:41, 8235.05it/s] 14%|█▍        | 57649/400000 [00:06<00:41, 8187.34it/s] 15%|█▍        | 58513/400000 [00:06<00:41, 8316.86it/s] 15%|█▍        | 59381/400000 [00:06<00:40, 8421.24it/s] 15%|█▌        | 60226/400000 [00:07<00:40, 8393.75it/s] 15%|█▌        | 61103/400000 [00:07<00:39, 8502.44it/s] 15%|█▌        | 61955/400000 [00:07<00:39, 8482.40it/s] 16%|█▌        | 62811/400000 [00:07<00:39, 8503.17it/s] 16%|█▌        | 63663/400000 [00:07<00:39, 8506.53it/s] 16%|█▌        | 64515/400000 [00:07<00:39, 8486.78it/s] 16%|█▋        | 65365/400000 [00:07<00:40, 8308.07it/s] 17%|█▋        | 66228/400000 [00:07<00:39, 8399.27it/s] 17%|█▋        | 67085/400000 [00:07<00:39, 8447.55it/s] 17%|█▋        | 67978/400000 [00:07<00:38, 8585.74it/s] 17%|█▋        | 68838/400000 [00:08<00:39, 8450.25it/s] 17%|█▋        | 69748/400000 [00:08<00:38, 8634.23it/s] 18%|█▊        | 70626/400000 [00:08<00:37, 8676.54it/s] 18%|█▊        | 71535/400000 [00:08<00:37, 8794.97it/s] 18%|█▊        | 72458/400000 [00:08<00:36, 8919.46it/s] 18%|█▊        | 73369/400000 [00:08<00:36, 8973.79it/s] 19%|█▊        | 74286/400000 [00:08<00:36, 9029.64it/s] 19%|█▉        | 75190/400000 [00:08<00:36, 8921.81it/s] 19%|█▉        | 76084/400000 [00:08<00:37, 8731.92it/s] 19%|█▉        | 76959/400000 [00:08<00:37, 8700.28it/s] 19%|█▉        | 77831/400000 [00:09<00:37, 8706.17it/s] 20%|█▉        | 78710/400000 [00:09<00:36, 8729.82it/s] 20%|█▉        | 79591/400000 [00:09<00:36, 8751.35it/s] 20%|██        | 80467/400000 [00:09<00:37, 8600.78it/s] 20%|██        | 81328/400000 [00:09<00:37, 8594.98it/s] 21%|██        | 82198/400000 [00:09<00:36, 8624.02it/s] 21%|██        | 83115/400000 [00:09<00:36, 8779.03it/s] 21%|██        | 83994/400000 [00:09<00:36, 8745.54it/s] 21%|██        | 84870/400000 [00:09<00:37, 8501.62it/s] 21%|██▏       | 85723/400000 [00:10<00:37, 8493.07it/s] 22%|██▏       | 86574/400000 [00:10<00:37, 8448.85it/s] 22%|██▏       | 87420/400000 [00:10<00:37, 8387.88it/s] 22%|██▏       | 88260/400000 [00:10<00:37, 8359.15it/s] 22%|██▏       | 89097/400000 [00:10<00:37, 8263.97it/s] 22%|██▏       | 89949/400000 [00:10<00:37, 8336.91it/s] 23%|██▎       | 90795/400000 [00:10<00:36, 8372.38it/s] 23%|██▎       | 91731/400000 [00:10<00:35, 8645.20it/s] 23%|██▎       | 92599/400000 [00:10<00:35, 8630.82it/s] 23%|██▎       | 93464/400000 [00:10<00:35, 8611.83it/s] 24%|██▎       | 94327/400000 [00:11<00:36, 8455.93it/s] 24%|██▍       | 95175/400000 [00:11<00:36, 8459.13it/s] 24%|██▍       | 96022/400000 [00:11<00:36, 8393.28it/s] 24%|██▍       | 96863/400000 [00:11<00:36, 8335.61it/s] 24%|██▍       | 97698/400000 [00:11<00:36, 8331.05it/s] 25%|██▍       | 98559/400000 [00:11<00:35, 8411.43it/s] 25%|██▍       | 99423/400000 [00:11<00:35, 8477.76it/s] 25%|██▌       | 100272/400000 [00:11<00:35, 8380.48it/s] 25%|██▌       | 101167/400000 [00:11<00:34, 8541.05it/s] 26%|██▌       | 102023/400000 [00:11<00:35, 8500.25it/s] 26%|██▌       | 102914/400000 [00:12<00:34, 8616.08it/s] 26%|██▌       | 103799/400000 [00:12<00:34, 8683.31it/s] 26%|██▌       | 104675/400000 [00:12<00:33, 8703.71it/s] 26%|██▋       | 105546/400000 [00:12<00:34, 8570.21it/s] 27%|██▋       | 106404/400000 [00:12<00:34, 8401.97it/s] 27%|██▋       | 107257/400000 [00:12<00:34, 8439.37it/s] 27%|██▋       | 108116/400000 [00:12<00:34, 8483.05it/s] 27%|██▋       | 108966/400000 [00:12<00:35, 8108.16it/s] 27%|██▋       | 109788/400000 [00:12<00:35, 8139.82it/s] 28%|██▊       | 110605/400000 [00:12<00:36, 8000.69it/s] 28%|██▊       | 111438/400000 [00:13<00:35, 8096.30it/s] 28%|██▊       | 112295/400000 [00:13<00:34, 8230.95it/s] 28%|██▊       | 113127/400000 [00:13<00:34, 8256.39it/s] 28%|██▊       | 113995/400000 [00:13<00:34, 8378.89it/s] 29%|██▊       | 114835/400000 [00:13<00:34, 8361.07it/s] 29%|██▉       | 115686/400000 [00:13<00:33, 8402.96it/s] 29%|██▉       | 116528/400000 [00:13<00:33, 8389.82it/s] 29%|██▉       | 117373/400000 [00:13<00:33, 8405.09it/s] 30%|██▉       | 118221/400000 [00:13<00:33, 8424.23it/s] 30%|██▉       | 119064/400000 [00:13<00:33, 8423.41it/s] 30%|██▉       | 119909/400000 [00:14<00:33, 8430.87it/s] 30%|███       | 120769/400000 [00:14<00:32, 8480.60it/s] 30%|███       | 121648/400000 [00:14<00:32, 8570.20it/s] 31%|███       | 122507/400000 [00:14<00:32, 8575.86it/s] 31%|███       | 123365/400000 [00:14<00:33, 8329.38it/s] 31%|███       | 124200/400000 [00:14<00:33, 8269.77it/s] 31%|███▏      | 125029/400000 [00:14<00:33, 8182.52it/s] 31%|███▏      | 125849/400000 [00:14<00:33, 8150.63it/s] 32%|███▏      | 126689/400000 [00:14<00:33, 8221.19it/s] 32%|███▏      | 127512/400000 [00:14<00:33, 8188.70it/s] 32%|███▏      | 128332/400000 [00:15<00:33, 8157.75it/s] 32%|███▏      | 129217/400000 [00:15<00:32, 8353.48it/s] 33%|███▎      | 130096/400000 [00:15<00:31, 8477.28it/s] 33%|███▎      | 130975/400000 [00:15<00:31, 8568.09it/s] 33%|███▎      | 131834/400000 [00:15<00:32, 8327.29it/s] 33%|███▎      | 132677/400000 [00:15<00:31, 8357.75it/s] 33%|███▎      | 133528/400000 [00:15<00:31, 8401.26it/s] 34%|███▎      | 134378/400000 [00:15<00:31, 8429.46it/s] 34%|███▍      | 135222/400000 [00:15<00:32, 8235.66it/s] 34%|███▍      | 136054/400000 [00:16<00:31, 8259.58it/s] 34%|███▍      | 136903/400000 [00:16<00:31, 8324.73it/s] 34%|███▍      | 137737/400000 [00:16<00:31, 8325.77it/s] 35%|███▍      | 138642/400000 [00:16<00:30, 8528.34it/s] 35%|███▍      | 139497/400000 [00:16<00:30, 8448.68it/s] 35%|███▌      | 140344/400000 [00:16<00:30, 8418.73it/s] 35%|███▌      | 141205/400000 [00:16<00:30, 8474.37it/s] 36%|███▌      | 142055/400000 [00:16<00:30, 8481.64it/s] 36%|███▌      | 142919/400000 [00:16<00:30, 8526.58it/s] 36%|███▌      | 143773/400000 [00:16<00:30, 8444.34it/s] 36%|███▌      | 144618/400000 [00:17<00:30, 8299.84it/s] 36%|███▋      | 145449/400000 [00:17<00:31, 8107.55it/s] 37%|███▋      | 146277/400000 [00:17<00:31, 8157.03it/s] 37%|███▋      | 147094/400000 [00:17<00:31, 8080.25it/s] 37%|███▋      | 147962/400000 [00:17<00:30, 8249.74it/s] 37%|███▋      | 148797/400000 [00:17<00:30, 8279.27it/s] 37%|███▋      | 149671/400000 [00:17<00:29, 8411.28it/s] 38%|███▊      | 150523/400000 [00:17<00:29, 8439.72it/s] 38%|███▊      | 151368/400000 [00:17<00:29, 8403.11it/s] 38%|███▊      | 152244/400000 [00:17<00:29, 8506.62it/s] 38%|███▊      | 153096/400000 [00:18<00:29, 8351.66it/s] 38%|███▊      | 153933/400000 [00:18<00:30, 8168.81it/s] 39%|███▊      | 154752/400000 [00:18<00:30, 8042.87it/s] 39%|███▉      | 155611/400000 [00:18<00:29, 8197.37it/s] 39%|███▉      | 156433/400000 [00:18<00:29, 8135.57it/s] 39%|███▉      | 157283/400000 [00:18<00:29, 8240.26it/s] 40%|███▉      | 158134/400000 [00:18<00:29, 8317.63it/s] 40%|███▉      | 158993/400000 [00:18<00:28, 8394.99it/s] 40%|███▉      | 159846/400000 [00:18<00:28, 8433.87it/s] 40%|████      | 160691/400000 [00:18<00:28, 8348.54it/s] 40%|████      | 161537/400000 [00:19<00:28, 8379.12it/s] 41%|████      | 162395/400000 [00:19<00:28, 8436.14it/s] 41%|████      | 163308/400000 [00:19<00:27, 8632.21it/s] 41%|████      | 164216/400000 [00:19<00:26, 8759.16it/s] 41%|████▏     | 165099/400000 [00:19<00:26, 8774.56it/s] 42%|████▏     | 166014/400000 [00:19<00:26, 8882.99it/s] 42%|████▏     | 166942/400000 [00:19<00:25, 8997.16it/s] 42%|████▏     | 167932/400000 [00:19<00:25, 9248.31it/s] 42%|████▏     | 168930/400000 [00:19<00:24, 9455.60it/s] 42%|████▏     | 169915/400000 [00:19<00:24, 9568.25it/s] 43%|████▎     | 170875/400000 [00:20<00:25, 9060.88it/s] 43%|████▎     | 171789/400000 [00:20<00:25, 8991.76it/s] 43%|████▎     | 172811/400000 [00:20<00:24, 9326.65it/s] 43%|████▎     | 173886/400000 [00:20<00:23, 9711.41it/s] 44%|████▎     | 174921/400000 [00:20<00:22, 9892.80it/s] 44%|████▍     | 175918/400000 [00:20<00:22, 9751.23it/s] 44%|████▍     | 176899/400000 [00:20<00:23, 9452.76it/s] 44%|████▍     | 177907/400000 [00:20<00:23, 9631.73it/s] 45%|████▍     | 178914/400000 [00:20<00:22, 9757.64it/s] 45%|████▍     | 179894/400000 [00:21<00:22, 9759.59it/s] 45%|████▌     | 180873/400000 [00:21<00:22, 9567.10it/s] 45%|████▌     | 181833/400000 [00:21<00:23, 9299.50it/s] 46%|████▌     | 182767/400000 [00:21<00:23, 9110.02it/s] 46%|████▌     | 183682/400000 [00:21<00:24, 9011.28it/s] 46%|████▌     | 184586/400000 [00:21<00:24, 8891.88it/s] 46%|████▋     | 185495/400000 [00:21<00:23, 8949.77it/s] 47%|████▋     | 186417/400000 [00:21<00:23, 9028.84it/s] 47%|████▋     | 187492/400000 [00:21<00:22, 9482.51it/s] 47%|████▋     | 188448/400000 [00:21<00:22, 9395.53it/s] 47%|████▋     | 189393/400000 [00:22<00:22, 9178.13it/s] 48%|████▊     | 190392/400000 [00:22<00:22, 9406.29it/s] 48%|████▊     | 191387/400000 [00:22<00:21, 9560.66it/s] 48%|████▊     | 192347/400000 [00:22<00:22, 9330.60it/s] 48%|████▊     | 193284/400000 [00:22<00:22, 9209.19it/s] 49%|████▊     | 194222/400000 [00:22<00:22, 9258.83it/s] 49%|████▉     | 195151/400000 [00:22<00:22, 9243.36it/s] 49%|████▉     | 196077/400000 [00:22<00:22, 9181.40it/s] 49%|████▉     | 197037/400000 [00:22<00:21, 9300.88it/s] 49%|████▉     | 197969/400000 [00:22<00:21, 9293.36it/s] 50%|████▉     | 198911/400000 [00:23<00:21, 9330.37it/s] 50%|████▉     | 199845/400000 [00:23<00:21, 9245.81it/s] 50%|█████     | 200771/400000 [00:23<00:21, 9188.39it/s] 50%|█████     | 201691/400000 [00:23<00:22, 8839.26it/s] 51%|█████     | 202579/400000 [00:23<00:22, 8827.39it/s] 51%|█████     | 203475/400000 [00:23<00:22, 8866.50it/s] 51%|█████     | 204364/400000 [00:23<00:22, 8552.97it/s] 51%|█████▏    | 205223/400000 [00:23<00:22, 8537.92it/s] 52%|█████▏    | 206158/400000 [00:23<00:22, 8764.44it/s] 52%|█████▏    | 207038/400000 [00:24<00:22, 8755.87it/s] 52%|█████▏    | 207916/400000 [00:24<00:22, 8724.77it/s] 52%|█████▏    | 208791/400000 [00:24<00:22, 8620.81it/s] 52%|█████▏    | 209688/400000 [00:24<00:21, 8719.97it/s] 53%|█████▎    | 210664/400000 [00:24<00:21, 9006.54it/s] 53%|█████▎    | 211568/400000 [00:24<00:21, 8888.05it/s] 53%|█████▎    | 212460/400000 [00:24<00:21, 8779.04it/s] 53%|█████▎    | 213374/400000 [00:24<00:21, 8883.55it/s] 54%|█████▎    | 214380/400000 [00:24<00:20, 9204.87it/s] 54%|█████▍    | 215357/400000 [00:24<00:19, 9365.24it/s] 54%|█████▍    | 216371/400000 [00:25<00:19, 9582.44it/s] 54%|█████▍    | 217334/400000 [00:25<00:19, 9585.39it/s] 55%|█████▍    | 218296/400000 [00:25<00:19, 9405.94it/s] 55%|█████▍    | 219240/400000 [00:25<00:19, 9397.45it/s] 55%|█████▌    | 220185/400000 [00:25<00:19, 9412.25it/s] 55%|█████▌    | 221128/400000 [00:25<00:19, 9349.48it/s] 56%|█████▌    | 222116/400000 [00:25<00:18, 9500.07it/s] 56%|█████▌    | 223068/400000 [00:25<00:19, 9284.27it/s] 56%|█████▌    | 223999/400000 [00:25<00:19, 9137.03it/s] 56%|█████▋    | 225021/400000 [00:25<00:18, 9436.01it/s] 57%|█████▋    | 226071/400000 [00:26<00:17, 9731.45it/s] 57%|█████▋    | 227050/400000 [00:26<00:18, 9605.19it/s] 57%|█████▋    | 228015/400000 [00:26<00:18, 9515.83it/s] 57%|█████▋    | 229009/400000 [00:26<00:17, 9636.93it/s] 57%|█████▋    | 229976/400000 [00:26<00:17, 9551.48it/s] 58%|█████▊    | 230939/400000 [00:26<00:17, 9572.44it/s] 58%|█████▊    | 231898/400000 [00:26<00:17, 9515.13it/s] 58%|█████▊    | 232851/400000 [00:26<00:17, 9452.99it/s] 58%|█████▊    | 233869/400000 [00:26<00:17, 9657.73it/s] 59%|█████▊    | 234837/400000 [00:26<00:17, 9641.12it/s] 59%|█████▉    | 235803/400000 [00:27<00:17, 9396.76it/s] 59%|█████▉    | 236745/400000 [00:27<00:17, 9085.96it/s] 59%|█████▉    | 237711/400000 [00:27<00:17, 9249.67it/s] 60%|█████▉    | 238687/400000 [00:27<00:17, 9395.47it/s] 60%|█████▉    | 239630/400000 [00:27<00:17, 9350.26it/s] 60%|██████    | 240568/400000 [00:27<00:17, 9099.53it/s] 60%|██████    | 241481/400000 [00:27<00:18, 8762.42it/s] 61%|██████    | 242362/400000 [00:27<00:18, 8683.65it/s] 61%|██████    | 243241/400000 [00:27<00:17, 8712.66it/s] 61%|██████    | 244226/400000 [00:28<00:17, 9025.29it/s] 61%|██████▏   | 245170/400000 [00:28<00:16, 9143.95it/s] 62%|██████▏   | 246088/400000 [00:28<00:17, 8992.68it/s] 62%|██████▏   | 247013/400000 [00:28<00:16, 9068.12it/s] 62%|██████▏   | 247936/400000 [00:28<00:16, 9115.02it/s] 62%|██████▏   | 248850/400000 [00:28<00:17, 8587.71it/s] 62%|██████▏   | 249717/400000 [00:28<00:17, 8585.50it/s] 63%|██████▎   | 250581/400000 [00:28<00:17, 8559.13it/s] 63%|██████▎   | 251441/400000 [00:28<00:17, 8541.49it/s] 63%|██████▎   | 252298/400000 [00:28<00:17, 8481.11it/s] 63%|██████▎   | 253149/400000 [00:29<00:17, 8375.57it/s] 64%|██████▎   | 254020/400000 [00:29<00:17, 8471.31it/s] 64%|██████▎   | 254869/400000 [00:29<00:17, 8458.64it/s] 64%|██████▍   | 255741/400000 [00:29<00:16, 8534.21it/s] 64%|██████▍   | 256604/400000 [00:29<00:16, 8560.61it/s] 64%|██████▍   | 257476/400000 [00:29<00:16, 8606.89it/s] 65%|██████▍   | 258338/400000 [00:29<00:16, 8588.05it/s] 65%|██████▍   | 259198/400000 [00:29<00:16, 8494.87it/s] 65%|██████▌   | 260048/400000 [00:29<00:16, 8442.24it/s] 65%|██████▌   | 260893/400000 [00:29<00:16, 8323.63it/s] 65%|██████▌   | 261750/400000 [00:30<00:16, 8393.84it/s] 66%|██████▌   | 262592/400000 [00:30<00:16, 8400.44it/s] 66%|██████▌   | 263433/400000 [00:30<00:16, 8397.41it/s] 66%|██████▌   | 264297/400000 [00:30<00:16, 8468.64it/s] 66%|██████▋   | 265158/400000 [00:30<00:15, 8508.83it/s] 67%|██████▋   | 266024/400000 [00:30<00:15, 8551.90it/s] 67%|██████▋   | 266899/400000 [00:30<00:15, 8610.07it/s] 67%|██████▋   | 267761/400000 [00:30<00:15, 8363.82it/s] 67%|██████▋   | 268636/400000 [00:30<00:15, 8474.89it/s] 67%|██████▋   | 269512/400000 [00:30<00:15, 8557.19it/s] 68%|██████▊   | 270375/400000 [00:31<00:15, 8578.29it/s] 68%|██████▊   | 271234/400000 [00:31<00:15, 8429.48it/s] 68%|██████▊   | 272079/400000 [00:31<00:15, 8423.85it/s] 68%|██████▊   | 272952/400000 [00:31<00:14, 8511.14it/s] 68%|██████▊   | 273805/400000 [00:31<00:14, 8516.72it/s] 69%|██████▊   | 274678/400000 [00:31<00:14, 8577.28it/s] 69%|██████▉   | 275537/400000 [00:31<00:14, 8522.45it/s] 69%|██████▉   | 276406/400000 [00:31<00:14, 8569.22it/s] 69%|██████▉   | 277264/400000 [00:31<00:14, 8432.87it/s] 70%|██████▉   | 278175/400000 [00:31<00:14, 8622.57it/s] 70%|██████▉   | 279045/400000 [00:32<00:13, 8643.48it/s] 70%|██████▉   | 279912/400000 [00:32<00:13, 8649.94it/s] 70%|███████   | 280793/400000 [00:32<00:13, 8694.41it/s] 70%|███████   | 281664/400000 [00:32<00:13, 8697.88it/s] 71%|███████   | 282580/400000 [00:32<00:13, 8831.46it/s] 71%|███████   | 283604/400000 [00:32<00:12, 9209.53it/s] 71%|███████   | 284530/400000 [00:32<00:12, 9135.84it/s] 71%|███████▏  | 285468/400000 [00:32<00:12, 9206.31it/s] 72%|███████▏  | 286475/400000 [00:32<00:12, 9447.88it/s] 72%|███████▏  | 287483/400000 [00:32<00:11, 9627.68it/s] 72%|███████▏  | 288467/400000 [00:33<00:11, 9686.60it/s] 72%|███████▏  | 289438/400000 [00:33<00:11, 9327.53it/s] 73%|███████▎  | 290376/400000 [00:33<00:12, 9040.81it/s] 73%|███████▎  | 291363/400000 [00:33<00:11, 9272.36it/s] 73%|███████▎  | 292296/400000 [00:33<00:11, 9218.87it/s] 73%|███████▎  | 293222/400000 [00:33<00:11, 9205.96it/s] 74%|███████▎  | 294146/400000 [00:33<00:12, 8719.69it/s] 74%|███████▍  | 295026/400000 [00:33<00:12, 8482.01it/s] 74%|███████▍  | 295881/400000 [00:33<00:12, 8428.05it/s] 74%|███████▍  | 296762/400000 [00:34<00:12, 8538.39it/s] 74%|███████▍  | 297671/400000 [00:34<00:11, 8695.34it/s] 75%|███████▍  | 298611/400000 [00:34<00:11, 8895.04it/s] 75%|███████▍  | 299504/400000 [00:34<00:11, 8699.54it/s] 75%|███████▌  | 300384/400000 [00:34<00:11, 8729.36it/s] 75%|███████▌  | 301264/400000 [00:34<00:11, 8749.29it/s] 76%|███████▌  | 302158/400000 [00:34<00:11, 8803.62it/s] 76%|███████▌  | 303040/400000 [00:34<00:11, 8755.46it/s] 76%|███████▌  | 303917/400000 [00:34<00:10, 8750.52it/s] 76%|███████▌  | 304811/400000 [00:34<00:10, 8804.97it/s] 76%|███████▋  | 305693/400000 [00:35<00:10, 8762.83it/s] 77%|███████▋  | 306570/400000 [00:35<00:10, 8736.52it/s] 77%|███████▋  | 307520/400000 [00:35<00:10, 8950.36it/s] 77%|███████▋  | 308441/400000 [00:35<00:10, 9026.41it/s] 77%|███████▋  | 309360/400000 [00:35<00:09, 9073.52it/s] 78%|███████▊  | 310277/400000 [00:35<00:09, 9101.50it/s] 78%|███████▊  | 311232/400000 [00:35<00:09, 9229.07it/s] 78%|███████▊  | 312156/400000 [00:35<00:09, 9178.81it/s] 78%|███████▊  | 313091/400000 [00:35<00:09, 9228.71it/s] 79%|███████▊  | 314015/400000 [00:35<00:09, 9218.31it/s] 79%|███████▊  | 314976/400000 [00:36<00:09, 9329.62it/s] 79%|███████▉  | 315910/400000 [00:36<00:09, 9300.13it/s] 79%|███████▉  | 316843/400000 [00:36<00:08, 9308.33it/s] 79%|███████▉  | 317816/400000 [00:36<00:08, 9430.30it/s] 80%|███████▉  | 318760/400000 [00:36<00:08, 9311.04it/s] 80%|███████▉  | 319730/400000 [00:36<00:08, 9423.68it/s] 80%|████████  | 320693/400000 [00:36<00:08, 9483.15it/s] 80%|████████  | 321647/400000 [00:36<00:08, 9498.59it/s] 81%|████████  | 322624/400000 [00:36<00:08, 9577.36it/s] 81%|████████  | 323583/400000 [00:36<00:08, 9245.37it/s] 81%|████████  | 324511/400000 [00:37<00:08, 9017.52it/s] 81%|████████▏ | 325416/400000 [00:37<00:08, 8846.25it/s] 82%|████████▏ | 326304/400000 [00:37<00:08, 8813.86it/s] 82%|████████▏ | 327188/400000 [00:37<00:08, 8652.26it/s] 82%|████████▏ | 328056/400000 [00:37<00:08, 8456.16it/s] 82%|████████▏ | 328906/400000 [00:37<00:08, 8468.74it/s] 82%|████████▏ | 329783/400000 [00:37<00:08, 8555.20it/s] 83%|████████▎ | 330704/400000 [00:37<00:07, 8740.88it/s] 83%|████████▎ | 331610/400000 [00:37<00:07, 8833.92it/s] 83%|████████▎ | 332550/400000 [00:38<00:07, 8996.09it/s] 83%|████████▎ | 333551/400000 [00:38<00:07, 9277.30it/s] 84%|████████▎ | 334548/400000 [00:38<00:06, 9471.89it/s] 84%|████████▍ | 335528/400000 [00:38<00:06, 9566.30it/s] 84%|████████▍ | 336563/400000 [00:38<00:06, 9785.79it/s] 84%|████████▍ | 337548/400000 [00:38<00:06, 9804.06it/s] 85%|████████▍ | 338625/400000 [00:38<00:06, 10074.60it/s] 85%|████████▍ | 339636/400000 [00:38<00:06, 10019.18it/s] 85%|████████▌ | 340641/400000 [00:38<00:06, 9840.62it/s]  85%|████████▌ | 341628/400000 [00:38<00:06, 9512.29it/s] 86%|████████▌ | 342584/400000 [00:39<00:06, 9406.81it/s] 86%|████████▌ | 343562/400000 [00:39<00:05, 9514.84it/s] 86%|████████▌ | 344516/400000 [00:39<00:05, 9459.74it/s] 86%|████████▋ | 345464/400000 [00:39<00:05, 9127.80it/s] 87%|████████▋ | 346381/400000 [00:39<00:05, 8974.63it/s] 87%|████████▋ | 347282/400000 [00:39<00:05, 8850.29it/s] 87%|████████▋ | 348170/400000 [00:39<00:05, 8801.54it/s] 87%|████████▋ | 349053/400000 [00:39<00:05, 8762.12it/s] 87%|████████▋ | 349931/400000 [00:39<00:05, 8708.97it/s] 88%|████████▊ | 350803/400000 [00:39<00:05, 8659.45it/s] 88%|████████▊ | 351670/400000 [00:40<00:05, 8606.15it/s] 88%|████████▊ | 352532/400000 [00:40<00:05, 8538.34it/s] 88%|████████▊ | 353387/400000 [00:40<00:05, 8538.35it/s] 89%|████████▊ | 354242/400000 [00:40<00:05, 8481.30it/s] 89%|████████▉ | 355095/400000 [00:40<00:05, 8495.57it/s] 89%|████████▉ | 355945/400000 [00:40<00:05, 8462.03it/s] 89%|████████▉ | 356801/400000 [00:40<00:05, 8491.02it/s] 89%|████████▉ | 357651/400000 [00:40<00:04, 8474.66it/s] 90%|████████▉ | 358521/400000 [00:40<00:04, 8540.23it/s] 90%|████████▉ | 359388/400000 [00:40<00:04, 8576.21it/s] 90%|█████████ | 360256/400000 [00:41<00:04, 8604.62it/s] 90%|█████████ | 361125/400000 [00:41<00:04, 8627.83it/s] 91%|█████████ | 362005/400000 [00:41<00:04, 8678.20it/s] 91%|█████████ | 362873/400000 [00:41<00:04, 8311.48it/s] 91%|█████████ | 363772/400000 [00:41<00:04, 8503.85it/s] 91%|█████████ | 364701/400000 [00:41<00:04, 8722.00it/s] 91%|█████████▏| 365578/400000 [00:41<00:04, 8518.24it/s] 92%|█████████▏| 366506/400000 [00:41<00:03, 8732.17it/s] 92%|█████████▏| 367540/400000 [00:41<00:03, 9158.16it/s] 92%|█████████▏| 368485/400000 [00:42<00:03, 9243.65it/s] 92%|█████████▏| 369416/400000 [00:42<00:03, 9180.80it/s] 93%|█████████▎| 370339/400000 [00:42<00:03, 8935.07it/s] 93%|█████████▎| 371237/400000 [00:42<00:03, 8685.33it/s] 93%|█████████▎| 372111/400000 [00:42<00:03, 8695.25it/s] 93%|█████████▎| 372984/400000 [00:42<00:03, 8645.65it/s] 93%|█████████▎| 373910/400000 [00:42<00:02, 8820.09it/s] 94%|█████████▎| 374853/400000 [00:42<00:02, 8994.25it/s] 94%|█████████▍| 375756/400000 [00:42<00:02, 8934.40it/s] 94%|█████████▍| 376711/400000 [00:42<00:02, 9109.22it/s] 94%|█████████▍| 377628/400000 [00:43<00:02, 9125.21it/s] 95%|█████████▍| 378543/400000 [00:43<00:02, 9051.85it/s] 95%|█████████▍| 379470/400000 [00:43<00:02, 9113.96it/s] 95%|█████████▌| 380383/400000 [00:43<00:02, 9027.25it/s] 95%|█████████▌| 381287/400000 [00:43<00:02, 8931.74it/s] 96%|█████████▌| 382181/400000 [00:43<00:02, 8751.29it/s] 96%|█████████▌| 383072/400000 [00:43<00:01, 8797.87it/s] 96%|█████████▌| 383953/400000 [00:43<00:01, 8412.80it/s] 96%|█████████▌| 384836/400000 [00:43<00:01, 8532.94it/s] 96%|█████████▋| 385729/400000 [00:43<00:01, 8647.24it/s] 97%|█████████▋| 386597/400000 [00:44<00:01, 8544.32it/s] 97%|█████████▋| 387454/400000 [00:44<00:01, 8487.72it/s] 97%|█████████▋| 388307/400000 [00:44<00:01, 8498.23it/s] 97%|█████████▋| 389158/400000 [00:44<00:01, 8482.82it/s] 98%|█████████▊| 390020/400000 [00:44<00:01, 8523.09it/s] 98%|█████████▊| 390904/400000 [00:44<00:01, 8615.61it/s] 98%|█████████▊| 391804/400000 [00:44<00:00, 8726.34it/s] 98%|█████████▊| 392741/400000 [00:44<00:00, 8909.37it/s] 98%|█████████▊| 393682/400000 [00:44<00:00, 9051.84it/s] 99%|█████████▊| 394608/400000 [00:44<00:00, 9112.27it/s] 99%|█████████▉| 395521/400000 [00:45<00:00, 9000.41it/s] 99%|█████████▉| 396423/400000 [00:45<00:00, 8932.52it/s] 99%|█████████▉| 397360/400000 [00:45<00:00, 9058.41it/s]100%|█████████▉| 398273/400000 [00:45<00:00, 9077.18it/s]100%|█████████▉| 399182/400000 [00:45<00:00, 8976.67it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8775.63it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f73fa5c94e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01133865527329015 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010938543539780837 	 Accuracy: 69

  model saves at 69% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16045 out of table with 15796 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16045 out of table with 15796 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

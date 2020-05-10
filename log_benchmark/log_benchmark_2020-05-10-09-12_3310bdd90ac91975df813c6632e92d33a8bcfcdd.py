
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f2a7ef384a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 09:13:03.083264
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 09:13:03.090859
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 09:13:03.095345
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 09:13:03.099078
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2a6b4c3b00> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 359550.0000
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 322673.1562
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 286103.6250
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 230236.3750
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 169773.0000
Epoch 6/10

1/1 [==============================] - 0s 115ms/step - loss: 116480.8203
Epoch 7/10

1/1 [==============================] - 0s 104ms/step - loss: 76773.6094
Epoch 8/10

1/1 [==============================] - 0s 103ms/step - loss: 50871.1484
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 34669.9531
Epoch 10/10

1/1 [==============================] - 0s 110ms/step - loss: 24331.9180

  #### Inference Need return ypred, ytrue ######################### 
[[ 6.7401305e-02  3.9534357e+00  3.6733675e+00  3.3106024e+00
   3.0575938e+00  2.9402022e+00  3.0121818e+00  3.1757252e+00
   3.2217059e+00  4.5730324e+00  3.0460181e+00  4.0435896e+00
   2.3852139e+00  2.9721675e+00  4.8506989e+00  3.6366272e+00
   4.0433493e+00  4.6847100e+00  4.3600183e+00  4.0669270e+00
   3.8366227e+00  3.8839645e+00  3.3683972e+00  4.1263514e+00
   3.7904916e+00  2.4816492e+00  2.5771010e+00  3.5682039e+00
   3.6796668e+00  3.3735566e+00  3.3637803e+00  3.9934053e+00
   3.8005590e+00  4.5994210e+00  3.6331487e+00  4.0849876e+00
   3.4033554e+00  3.8396349e+00  3.4183891e+00  3.3333492e+00
   3.2715616e+00  3.3633769e+00  2.6342642e+00  3.6249399e+00
   3.0245178e+00  4.4920006e+00  4.1155243e+00  4.1212368e+00
   3.4218614e+00  4.6593843e+00  3.4859600e+00  3.6456501e+00
   3.9773955e+00  3.8349216e+00  3.0791678e+00  3.6967282e+00
   4.1972179e+00  3.2269983e+00  3.5980039e+00  3.8054280e+00
   3.7637135e-01 -5.7370973e-01  1.0164839e-01 -3.9267987e-01
   1.0162884e-01 -7.9304832e-01  4.8186779e-03 -4.3888432e-01
  -5.3374404e-01  2.2769514e-01 -5.2430469e-01  4.7448510e-01
  -4.2288548e-01  6.7143905e-01  1.6950858e-01 -4.1846201e-01
   2.8342748e-01  3.4753263e-02 -1.1635760e+00 -6.5835524e-01
  -9.0775216e-01 -2.4465469e-01  7.9926717e-01  8.6827517e-01
   1.7280242e-01  3.9187366e-01 -8.6152816e-01 -1.1112567e+00
  -6.0577011e-01 -1.8253943e-01 -1.2681794e+00 -4.9021766e-01
  -8.9102310e-01 -3.0188537e-01  3.8063318e-01  2.9956120e-01
  -2.9306889e-02  4.8174983e-01 -2.5296074e-01  6.9302022e-02
   1.1086106e+00 -8.0269229e-01  1.0386631e-01 -1.4595106e+00
  -2.5259700e-01  5.9254634e-01  1.0018108e+00  2.8744185e-01
  -7.2461230e-01 -5.9719247e-01 -2.8655267e-01  3.5101414e-02
  -7.5762975e-01  1.4809379e-01 -6.3241714e-01  5.4174149e-01
  -3.2485378e-01 -5.2579045e-03  6.0394675e-01 -4.5347783e-01
  -1.8534800e-01  9.1276193e-01  4.5515805e-01 -1.4888927e-01
   8.2008207e-01 -1.0816202e-01  2.1373302e-02  2.3598637e-01
   3.6749196e-01 -9.7217274e-01  7.5352490e-01  1.0576040e+00
   1.0172990e-01  6.7156166e-01  7.6386869e-01  3.5236284e-01
  -1.1338854e-01 -1.9321412e-02 -6.2839311e-01  1.2493557e-01
   6.9241762e-01 -2.9994112e-01  3.0867720e-01 -1.7406468e-01
  -2.3654950e-01 -7.5596160e-01 -1.0333389e-02  4.0204719e-01
   1.5766130e-01  3.3998567e-01  4.3541113e-01 -7.1738768e-01
  -1.6277200e-01 -6.5612477e-01 -4.9822545e-01 -5.9716326e-01
   3.0935431e-01 -1.0203128e+00 -1.1932406e+00  9.6011496e-01
  -6.6890782e-01  8.7749565e-01 -1.4523441e-01 -1.2438868e-01
  -1.8204081e-01 -8.0710554e-01 -3.0325192e-01  1.5853184e-01
   2.7907079e-01 -1.3335302e-01 -6.3479793e-01  2.0460075e-01
   8.0462778e-01  5.0954890e-01 -5.6854266e-01 -6.3712001e-01
  -2.7500567e-01 -1.3878235e-01  3.6971146e-01  4.4757012e-01
   3.0093610e-02  4.9085870e+00  3.9830189e+00  4.6322317e+00
   4.7293105e+00  4.8154244e+00  5.2207246e+00  4.5024958e+00
   4.4051127e+00  4.4098959e+00  5.0083127e+00  5.2895842e+00
   4.7328553e+00  4.7960076e+00  4.7128201e+00  4.3086591e+00
   3.6985903e+00  3.5839033e+00  5.7324686e+00  5.0531287e+00
   4.6414442e+00  4.0145602e+00  3.8613601e+00  5.1123729e+00
   3.9326353e+00  5.3868356e+00  4.1372604e+00  5.1000805e+00
   4.6986022e+00  4.6570191e+00  4.7269783e+00  4.1298018e+00
   4.1546488e+00  4.0659533e+00  4.3755708e+00  5.1749678e+00
   3.9322901e+00  3.8698859e+00  5.1076488e+00  5.2758203e+00
   3.7763057e+00  4.3811135e+00  5.3505063e+00  5.2414646e+00
   5.3054161e+00  4.7181077e+00  4.1724215e+00  3.7003469e+00
   4.8364649e+00  3.3730173e+00  4.2761297e+00  4.2370601e+00
   3.6443949e+00  4.9121628e+00  5.3861475e+00  4.2976193e+00
   4.9146261e+00  4.5046878e+00  4.3647847e+00  5.0420108e+00
   1.9519423e+00  1.0437357e+00  1.4330052e+00  8.1615841e-01
   9.6314216e-01  1.7768139e+00  1.5367131e+00  9.4657290e-01
   2.2030687e+00  6.9227773e-01  1.9158567e+00  1.8646410e+00
   2.1050539e+00  1.0375501e+00  8.1701893e-01  5.1017630e-01
   6.5673566e-01  5.6317067e-01  1.4659996e+00  3.4500372e-01
   1.9294034e+00  2.0320792e+00  6.3455105e-01  1.4548064e+00
   1.9566125e+00  1.6486833e+00  1.5416241e+00  1.2797052e+00
   1.6725225e+00  6.3724327e-01  1.7840266e+00  5.0716925e-01
   9.4127047e-01  1.1378262e+00  1.3530140e+00  1.1194561e+00
   2.0288622e+00  1.0786630e+00  7.3615074e-01  6.0109979e-01
   1.4732810e+00  1.1642560e+00  1.7303112e+00  6.8831754e-01
   1.6512125e+00  2.1270056e+00  1.1900465e+00  1.5266645e+00
   5.7858938e-01  1.2104347e+00  3.2961071e-01  6.0088414e-01
   8.9121461e-01  1.2800857e+00  7.5207341e-01  9.7906518e-01
   4.3309796e-01  1.4155080e+00  9.5002276e-01  4.2496765e-01
   1.4294195e+00  7.1965706e-01  1.2140423e+00  2.5267897e+00
   6.6200447e-01  7.6767170e-01  1.4695635e+00  9.9764431e-01
   7.0099133e-01  1.2463777e+00  1.5034870e+00  1.1534355e+00
   6.4701951e-01  1.4573365e+00  9.3968552e-01  1.9272571e+00
   2.0160608e+00  6.3425779e-01  7.2757912e-01  1.1763971e+00
   5.2188069e-01  1.9767054e+00  5.9538221e-01  1.4307582e+00
   1.1676571e+00  4.0408266e-01  7.2023088e-01  1.6466986e+00
   3.2138330e-01  3.3478433e-01  2.1192269e+00  8.9507872e-01
   1.4568787e+00  1.2658367e+00  1.3100609e+00  3.9190549e-01
   1.2157362e+00  5.6761724e-01  5.4349887e-01  4.7094876e-01
   1.4998600e+00  3.1281978e-01  1.2214993e+00  5.7851374e-01
   1.9493008e+00  1.3046792e+00  1.5306950e+00  1.4281831e+00
   7.0335752e-01  1.0297335e+00  1.4016035e+00  1.1659876e+00
   4.6445930e-01  1.7757013e+00  1.6546485e+00  1.7725109e+00
   7.2693264e-01  7.4643612e-01  1.3364402e+00  1.8027440e+00
   2.5721023e+00 -1.1440028e+00 -5.2270994e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 09:13:12.543502
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.4491
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 09:13:12.547874
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9705.24
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 09:13:12.551654
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.0695
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 09:13:12.556391
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -868.173
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139819806368208
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139818596483984
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139818596484488
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139818596484992
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139818596485496
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139818596486000

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2a696ebe80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.465489
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.444028
grad_step = 000002, loss = 0.428324
grad_step = 000003, loss = 0.414323
grad_step = 000004, loss = 0.403538
grad_step = 000005, loss = 0.395284
grad_step = 000006, loss = 0.385354
grad_step = 000007, loss = 0.373774
grad_step = 000008, loss = 0.363106
grad_step = 000009, loss = 0.353366
grad_step = 000010, loss = 0.344029
grad_step = 000011, loss = 0.335131
grad_step = 000012, loss = 0.326668
grad_step = 000013, loss = 0.318096
grad_step = 000014, loss = 0.309225
grad_step = 000015, loss = 0.300289
grad_step = 000016, loss = 0.291466
grad_step = 000017, loss = 0.282854
grad_step = 000018, loss = 0.274459
grad_step = 000019, loss = 0.266200
grad_step = 000020, loss = 0.257917
grad_step = 000021, loss = 0.249549
grad_step = 000022, loss = 0.241238
grad_step = 000023, loss = 0.233132
grad_step = 000024, loss = 0.225231
grad_step = 000025, loss = 0.217439
grad_step = 000026, loss = 0.209631
grad_step = 000027, loss = 0.201796
grad_step = 000028, loss = 0.194039
grad_step = 000029, loss = 0.186471
grad_step = 000030, loss = 0.179145
grad_step = 000031, loss = 0.171987
grad_step = 000032, loss = 0.164906
grad_step = 000033, loss = 0.157875
grad_step = 000034, loss = 0.150979
grad_step = 000035, loss = 0.144297
grad_step = 000036, loss = 0.137842
grad_step = 000037, loss = 0.131533
grad_step = 000038, loss = 0.125353
grad_step = 000039, loss = 0.119347
grad_step = 000040, loss = 0.113544
grad_step = 000041, loss = 0.107960
grad_step = 000042, loss = 0.102568
grad_step = 000043, loss = 0.097337
grad_step = 000044, loss = 0.092285
grad_step = 000045, loss = 0.087424
grad_step = 000046, loss = 0.082758
grad_step = 000047, loss = 0.078284
grad_step = 000048, loss = 0.073989
grad_step = 000049, loss = 0.069870
grad_step = 000050, loss = 0.065921
grad_step = 000051, loss = 0.062150
grad_step = 000052, loss = 0.058551
grad_step = 000053, loss = 0.055109
grad_step = 000054, loss = 0.051808
grad_step = 000055, loss = 0.048659
grad_step = 000056, loss = 0.045670
grad_step = 000057, loss = 0.042828
grad_step = 000058, loss = 0.040118
grad_step = 000059, loss = 0.037545
grad_step = 000060, loss = 0.035114
grad_step = 000061, loss = 0.032839
grad_step = 000062, loss = 0.030740
grad_step = 000063, loss = 0.028788
grad_step = 000064, loss = 0.026800
grad_step = 000065, loss = 0.024741
grad_step = 000066, loss = 0.022992
grad_step = 000067, loss = 0.021522
grad_step = 000068, loss = 0.019962
grad_step = 000069, loss = 0.018311
grad_step = 000070, loss = 0.016950
grad_step = 000071, loss = 0.015768
grad_step = 000072, loss = 0.014517
grad_step = 000073, loss = 0.013390
grad_step = 000074, loss = 0.012491
grad_step = 000075, loss = 0.011589
grad_step = 000076, loss = 0.010619
grad_step = 000077, loss = 0.009798
grad_step = 000078, loss = 0.009099
grad_step = 000079, loss = 0.008386
grad_step = 000080, loss = 0.007715
grad_step = 000081, loss = 0.007186
grad_step = 000082, loss = 0.006695
grad_step = 000083, loss = 0.006177
grad_step = 000084, loss = 0.005699
grad_step = 000085, loss = 0.005318
grad_step = 000086, loss = 0.004974
grad_step = 000087, loss = 0.004635
grad_step = 000088, loss = 0.004333
grad_step = 000089, loss = 0.004091
grad_step = 000090, loss = 0.003879
grad_step = 000091, loss = 0.003667
grad_step = 000092, loss = 0.003467
grad_step = 000093, loss = 0.003303
grad_step = 000094, loss = 0.003178
grad_step = 000095, loss = 0.003069
grad_step = 000096, loss = 0.002961
grad_step = 000097, loss = 0.002858
grad_step = 000098, loss = 0.002769
grad_step = 000099, loss = 0.002699
grad_step = 000100, loss = 0.002645
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002600
grad_step = 000102, loss = 0.002557
grad_step = 000103, loss = 0.002514
grad_step = 000104, loss = 0.002474
grad_step = 000105, loss = 0.002439
grad_step = 000106, loss = 0.002411
grad_step = 000107, loss = 0.002390
grad_step = 000108, loss = 0.002372
grad_step = 000109, loss = 0.002358
grad_step = 000110, loss = 0.002347
grad_step = 000111, loss = 0.002340
grad_step = 000112, loss = 0.002337
grad_step = 000113, loss = 0.002344
grad_step = 000114, loss = 0.002361
grad_step = 000115, loss = 0.002394
grad_step = 000116, loss = 0.002441
grad_step = 000117, loss = 0.002485
grad_step = 000118, loss = 0.002480
grad_step = 000119, loss = 0.002402
grad_step = 000120, loss = 0.002301
grad_step = 000121, loss = 0.002263
grad_step = 000122, loss = 0.002303
grad_step = 000123, loss = 0.002360
grad_step = 000124, loss = 0.002366
grad_step = 000125, loss = 0.002310
grad_step = 000126, loss = 0.002254
grad_step = 000127, loss = 0.002252
grad_step = 000128, loss = 0.002289
grad_step = 000129, loss = 0.002311
grad_step = 000130, loss = 0.002292
grad_step = 000131, loss = 0.002252
grad_step = 000132, loss = 0.002232
grad_step = 000133, loss = 0.002244
grad_step = 000134, loss = 0.002265
grad_step = 000135, loss = 0.002269
grad_step = 000136, loss = 0.002251
grad_step = 000137, loss = 0.002228
grad_step = 000138, loss = 0.002217
grad_step = 000139, loss = 0.002222
grad_step = 000140, loss = 0.002234
grad_step = 000141, loss = 0.002239
grad_step = 000142, loss = 0.002232
grad_step = 000143, loss = 0.002219
grad_step = 000144, loss = 0.002206
grad_step = 000145, loss = 0.002199
grad_step = 000146, loss = 0.002199
grad_step = 000147, loss = 0.002203
grad_step = 000148, loss = 0.002208
grad_step = 000149, loss = 0.002211
grad_step = 000150, loss = 0.002212
grad_step = 000151, loss = 0.002211
grad_step = 000152, loss = 0.002207
grad_step = 000153, loss = 0.002203
grad_step = 000154, loss = 0.002199
grad_step = 000155, loss = 0.002195
grad_step = 000156, loss = 0.002193
grad_step = 000157, loss = 0.002191
grad_step = 000158, loss = 0.002191
grad_step = 000159, loss = 0.002194
grad_step = 000160, loss = 0.002202
grad_step = 000161, loss = 0.002216
grad_step = 000162, loss = 0.002244
grad_step = 000163, loss = 0.002288
grad_step = 000164, loss = 0.002352
grad_step = 000165, loss = 0.002410
grad_step = 000166, loss = 0.002429
grad_step = 000167, loss = 0.002352
grad_step = 000168, loss = 0.002226
grad_step = 000169, loss = 0.002153
grad_step = 000170, loss = 0.002180
grad_step = 000171, loss = 0.002255
grad_step = 000172, loss = 0.002287
grad_step = 000173, loss = 0.002243
grad_step = 000174, loss = 0.002168
grad_step = 000175, loss = 0.002141
grad_step = 000176, loss = 0.002175
grad_step = 000177, loss = 0.002216
grad_step = 000178, loss = 0.002215
grad_step = 000179, loss = 0.002173
grad_step = 000180, loss = 0.002136
grad_step = 000181, loss = 0.002136
grad_step = 000182, loss = 0.002162
grad_step = 000183, loss = 0.002180
grad_step = 000184, loss = 0.002170
grad_step = 000185, loss = 0.002143
grad_step = 000186, loss = 0.002123
grad_step = 000187, loss = 0.002123
grad_step = 000188, loss = 0.002136
grad_step = 000189, loss = 0.002148
grad_step = 000190, loss = 0.002147
grad_step = 000191, loss = 0.002135
grad_step = 000192, loss = 0.002120
grad_step = 000193, loss = 0.002109
grad_step = 000194, loss = 0.002107
grad_step = 000195, loss = 0.002111
grad_step = 000196, loss = 0.002117
grad_step = 000197, loss = 0.002121
grad_step = 000198, loss = 0.002122
grad_step = 000199, loss = 0.002120
grad_step = 000200, loss = 0.002114
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002108
grad_step = 000202, loss = 0.002101
grad_step = 000203, loss = 0.002095
grad_step = 000204, loss = 0.002090
grad_step = 000205, loss = 0.002086
grad_step = 000206, loss = 0.002083
grad_step = 000207, loss = 0.002080
grad_step = 000208, loss = 0.002077
grad_step = 000209, loss = 0.002075
grad_step = 000210, loss = 0.002073
grad_step = 000211, loss = 0.002071
grad_step = 000212, loss = 0.002071
grad_step = 000213, loss = 0.002073
grad_step = 000214, loss = 0.002080
grad_step = 000215, loss = 0.002101
grad_step = 000216, loss = 0.002150
grad_step = 000217, loss = 0.002265
grad_step = 000218, loss = 0.002445
grad_step = 000219, loss = 0.002674
grad_step = 000220, loss = 0.002668
grad_step = 000221, loss = 0.002335
grad_step = 000222, loss = 0.002058
grad_step = 000223, loss = 0.002164
grad_step = 000224, loss = 0.002376
grad_step = 000225, loss = 0.002297
grad_step = 000226, loss = 0.002080
grad_step = 000227, loss = 0.002064
grad_step = 000228, loss = 0.002219
grad_step = 000229, loss = 0.002267
grad_step = 000230, loss = 0.002081
grad_step = 000231, loss = 0.002040
grad_step = 000232, loss = 0.002151
grad_step = 000233, loss = 0.002151
grad_step = 000234, loss = 0.002031
grad_step = 000235, loss = 0.002037
grad_step = 000236, loss = 0.002110
grad_step = 000237, loss = 0.002069
grad_step = 000238, loss = 0.002002
grad_step = 000239, loss = 0.002024
grad_step = 000240, loss = 0.002067
grad_step = 000241, loss = 0.002023
grad_step = 000242, loss = 0.001981
grad_step = 000243, loss = 0.001998
grad_step = 000244, loss = 0.002021
grad_step = 000245, loss = 0.001985
grad_step = 000246, loss = 0.001960
grad_step = 000247, loss = 0.001972
grad_step = 000248, loss = 0.001982
grad_step = 000249, loss = 0.001958
grad_step = 000250, loss = 0.001935
grad_step = 000251, loss = 0.001941
grad_step = 000252, loss = 0.001946
grad_step = 000253, loss = 0.001932
grad_step = 000254, loss = 0.001916
grad_step = 000255, loss = 0.001911
grad_step = 000256, loss = 0.001908
grad_step = 000257, loss = 0.001895
grad_step = 000258, loss = 0.001885
grad_step = 000259, loss = 0.001882
grad_step = 000260, loss = 0.001888
grad_step = 000261, loss = 0.001897
grad_step = 000262, loss = 0.001915
grad_step = 000263, loss = 0.001934
grad_step = 000264, loss = 0.001950
grad_step = 000265, loss = 0.001914
grad_step = 000266, loss = 0.001864
grad_step = 000267, loss = 0.001844
grad_step = 000268, loss = 0.001861
grad_step = 000269, loss = 0.001886
grad_step = 000270, loss = 0.001878
grad_step = 000271, loss = 0.001846
grad_step = 000272, loss = 0.001821
grad_step = 000273, loss = 0.001813
grad_step = 000274, loss = 0.001811
grad_step = 000275, loss = 0.001812
grad_step = 000276, loss = 0.001823
grad_step = 000277, loss = 0.001843
grad_step = 000278, loss = 0.001875
grad_step = 000279, loss = 0.001845
grad_step = 000280, loss = 0.001804
grad_step = 000281, loss = 0.001773
grad_step = 000282, loss = 0.001778
grad_step = 000283, loss = 0.001796
grad_step = 000284, loss = 0.001786
grad_step = 000285, loss = 0.001765
grad_step = 000286, loss = 0.001751
grad_step = 000287, loss = 0.001735
grad_step = 000288, loss = 0.001728
grad_step = 000289, loss = 0.001732
grad_step = 000290, loss = 0.001774
grad_step = 000291, loss = 0.001840
grad_step = 000292, loss = 0.001911
grad_step = 000293, loss = 0.001881
grad_step = 000294, loss = 0.001751
grad_step = 000295, loss = 0.001660
grad_step = 000296, loss = 0.001714
grad_step = 000297, loss = 0.001789
grad_step = 000298, loss = 0.001704
grad_step = 000299, loss = 0.001631
grad_step = 000300, loss = 0.001665
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001708
grad_step = 000302, loss = 0.001679
grad_step = 000303, loss = 0.001603
grad_step = 000304, loss = 0.001591
grad_step = 000305, loss = 0.001632
grad_step = 000306, loss = 0.001645
grad_step = 000307, loss = 0.001589
grad_step = 000308, loss = 0.001546
grad_step = 000309, loss = 0.001540
grad_step = 000310, loss = 0.001564
grad_step = 000311, loss = 0.001606
grad_step = 000312, loss = 0.001571
grad_step = 000313, loss = 0.001510
grad_step = 000314, loss = 0.001498
grad_step = 000315, loss = 0.001505
grad_step = 000316, loss = 0.001529
grad_step = 000317, loss = 0.001507
grad_step = 000318, loss = 0.001472
grad_step = 000319, loss = 0.001456
grad_step = 000320, loss = 0.001454
grad_step = 000321, loss = 0.001459
grad_step = 000322, loss = 0.001452
grad_step = 000323, loss = 0.001436
grad_step = 000324, loss = 0.001414
grad_step = 000325, loss = 0.001412
grad_step = 000326, loss = 0.001463
grad_step = 000327, loss = 0.001526
grad_step = 000328, loss = 0.001694
grad_step = 000329, loss = 0.001612
grad_step = 000330, loss = 0.001473
grad_step = 000331, loss = 0.001339
grad_step = 000332, loss = 0.001427
grad_step = 000333, loss = 0.001560
grad_step = 000334, loss = 0.001441
grad_step = 000335, loss = 0.001284
grad_step = 000336, loss = 0.001379
grad_step = 000337, loss = 0.001488
grad_step = 000338, loss = 0.001427
grad_step = 000339, loss = 0.001272
grad_step = 000340, loss = 0.001339
grad_step = 000341, loss = 0.001530
grad_step = 000342, loss = 0.001371
grad_step = 000343, loss = 0.001242
grad_step = 000344, loss = 0.001303
grad_step = 000345, loss = 0.001319
grad_step = 000346, loss = 0.001221
grad_step = 000347, loss = 0.001194
grad_step = 000348, loss = 0.001237
grad_step = 000349, loss = 0.001186
grad_step = 000350, loss = 0.001149
grad_step = 000351, loss = 0.001173
grad_step = 000352, loss = 0.001193
grad_step = 000353, loss = 0.001117
grad_step = 000354, loss = 0.001129
grad_step = 000355, loss = 0.001173
grad_step = 000356, loss = 0.001129
grad_step = 000357, loss = 0.001073
grad_step = 000358, loss = 0.001068
grad_step = 000359, loss = 0.001109
grad_step = 000360, loss = 0.001120
grad_step = 000361, loss = 0.001081
grad_step = 000362, loss = 0.001039
grad_step = 000363, loss = 0.001064
grad_step = 000364, loss = 0.001164
grad_step = 000365, loss = 0.001207
grad_step = 000366, loss = 0.001203
grad_step = 000367, loss = 0.001121
grad_step = 000368, loss = 0.001125
grad_step = 000369, loss = 0.001244
grad_step = 000370, loss = 0.001365
grad_step = 000371, loss = 0.001039
grad_step = 000372, loss = 0.000938
grad_step = 000373, loss = 0.001123
grad_step = 000374, loss = 0.001287
grad_step = 000375, loss = 0.001175
grad_step = 000376, loss = 0.001002
grad_step = 000377, loss = 0.001167
grad_step = 000378, loss = 0.001209
grad_step = 000379, loss = 0.000952
grad_step = 000380, loss = 0.000925
grad_step = 000381, loss = 0.001064
grad_step = 000382, loss = 0.001021
grad_step = 000383, loss = 0.000876
grad_step = 000384, loss = 0.000879
grad_step = 000385, loss = 0.000927
grad_step = 000386, loss = 0.000865
grad_step = 000387, loss = 0.000864
grad_step = 000388, loss = 0.000898
grad_step = 000389, loss = 0.000903
grad_step = 000390, loss = 0.000818
grad_step = 000391, loss = 0.000787
grad_step = 000392, loss = 0.000841
grad_step = 000393, loss = 0.000830
grad_step = 000394, loss = 0.000814
grad_step = 000395, loss = 0.000798
grad_step = 000396, loss = 0.000798
grad_step = 000397, loss = 0.000775
grad_step = 000398, loss = 0.000733
grad_step = 000399, loss = 0.000729
grad_step = 000400, loss = 0.000749
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000761
grad_step = 000402, loss = 0.000745
grad_step = 000403, loss = 0.000735
grad_step = 000404, loss = 0.000764
grad_step = 000405, loss = 0.000863
grad_step = 000406, loss = 0.000850
grad_step = 000407, loss = 0.000800
grad_step = 000408, loss = 0.000739
grad_step = 000409, loss = 0.000734
grad_step = 000410, loss = 0.000722
grad_step = 000411, loss = 0.000685
grad_step = 000412, loss = 0.000653
grad_step = 000413, loss = 0.000642
grad_step = 000414, loss = 0.000647
grad_step = 000415, loss = 0.000647
grad_step = 000416, loss = 0.000631
grad_step = 000417, loss = 0.000614
grad_step = 000418, loss = 0.000603
grad_step = 000419, loss = 0.000603
grad_step = 000420, loss = 0.000604
grad_step = 000421, loss = 0.000604
grad_step = 000422, loss = 0.000605
grad_step = 000423, loss = 0.000620
grad_step = 000424, loss = 0.000707
grad_step = 000425, loss = 0.000818
grad_step = 000426, loss = 0.000993
grad_step = 000427, loss = 0.000986
grad_step = 000428, loss = 0.000902
grad_step = 000429, loss = 0.000721
grad_step = 000430, loss = 0.000608
grad_step = 000431, loss = 0.000624
grad_step = 000432, loss = 0.000711
grad_step = 000433, loss = 0.000742
grad_step = 000434, loss = 0.000650
grad_step = 000435, loss = 0.000591
grad_step = 000436, loss = 0.000612
grad_step = 000437, loss = 0.000638
grad_step = 000438, loss = 0.000629
grad_step = 000439, loss = 0.000570
grad_step = 000440, loss = 0.000553
grad_step = 000441, loss = 0.000590
grad_step = 000442, loss = 0.000617
grad_step = 000443, loss = 0.000608
grad_step = 000444, loss = 0.000536
grad_step = 000445, loss = 0.000499
grad_step = 000446, loss = 0.000540
grad_step = 000447, loss = 0.000598
grad_step = 000448, loss = 0.000634
grad_step = 000449, loss = 0.000569
grad_step = 000450, loss = 0.000510
grad_step = 000451, loss = 0.000501
grad_step = 000452, loss = 0.000515
grad_step = 000453, loss = 0.000527
grad_step = 000454, loss = 0.000522
grad_step = 000455, loss = 0.000510
grad_step = 000456, loss = 0.000504
grad_step = 000457, loss = 0.000491
grad_step = 000458, loss = 0.000478
grad_step = 000459, loss = 0.000467
grad_step = 000460, loss = 0.000464
grad_step = 000461, loss = 0.000473
grad_step = 000462, loss = 0.000471
grad_step = 000463, loss = 0.000461
grad_step = 000464, loss = 0.000447
grad_step = 000465, loss = 0.000440
grad_step = 000466, loss = 0.000449
grad_step = 000467, loss = 0.000454
grad_step = 000468, loss = 0.000454
grad_step = 000469, loss = 0.000445
grad_step = 000470, loss = 0.000437
grad_step = 000471, loss = 0.000442
grad_step = 000472, loss = 0.000456
grad_step = 000473, loss = 0.000486
grad_step = 000474, loss = 0.000521
grad_step = 000475, loss = 0.000587
grad_step = 000476, loss = 0.000625
grad_step = 000477, loss = 0.000620
grad_step = 000478, loss = 0.000530
grad_step = 000479, loss = 0.000448
grad_step = 000480, loss = 0.000419
grad_step = 000481, loss = 0.000435
grad_step = 000482, loss = 0.000480
grad_step = 000483, loss = 0.000512
grad_step = 000484, loss = 0.000480
grad_step = 000485, loss = 0.000433
grad_step = 000486, loss = 0.000415
grad_step = 000487, loss = 0.000432
grad_step = 000488, loss = 0.000443
grad_step = 000489, loss = 0.000437
grad_step = 000490, loss = 0.000416
grad_step = 000491, loss = 0.000403
grad_step = 000492, loss = 0.000411
grad_step = 000493, loss = 0.000422
grad_step = 000494, loss = 0.000419
grad_step = 000495, loss = 0.000408
grad_step = 000496, loss = 0.000393
grad_step = 000497, loss = 0.000387
grad_step = 000498, loss = 0.000390
grad_step = 000499, loss = 0.000396
grad_step = 000500, loss = 0.000402
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000404
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

  date_run                              2020-05-10 09:13:37.217449
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.206896
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 09:13:37.223796
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.110268
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 09:13:37.231634
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.11579
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 09:13:37.237754
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.675561
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
0   2020-05-10 09:13:03.083264  ...    mean_absolute_error
1   2020-05-10 09:13:03.090859  ...     mean_squared_error
2   2020-05-10 09:13:03.095345  ...  median_absolute_error
3   2020-05-10 09:13:03.099078  ...               r2_score
4   2020-05-10 09:13:12.543502  ...    mean_absolute_error
5   2020-05-10 09:13:12.547874  ...     mean_squared_error
6   2020-05-10 09:13:12.551654  ...  median_absolute_error
7   2020-05-10 09:13:12.556391  ...               r2_score
8   2020-05-10 09:13:37.217449  ...    mean_absolute_error
9   2020-05-10 09:13:37.223796  ...     mean_squared_error
10  2020-05-10 09:13:37.231634  ...  median_absolute_error
11  2020-05-10 09:13:37.237754  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 142970.18it/s] 96%|█████████▌| 9535488/9912422 [00:00<00:01, 204111.72it/s]9920512it [00:00, 45840339.80it/s]                           
0it [00:00, ?it/s]32768it [00:00, 631085.00it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 490731.89it/s]1654784it [00:00, 11366124.59it/s]                         
0it [00:00, ?it/s]8192it [00:00, 217370.40it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f40e9e8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2decb559b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f3cae10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2decb55da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f4136d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f413f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2decb58080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f413f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2decb58080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f413f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4f4136d8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7eff24f791d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=987d5727d66f4f36df60b9b209f8fcbfbf1684704d9a74dff833c9babca030ca
  Stored in directory: /tmp/pip-ephem-wheel-cache-jf0mn1o3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7eff1b0e7080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2367488/17464789 [===>..........................] - ETA: 0s
 6979584/17464789 [==========>...................] - ETA: 0s
11493376/17464789 [==================>...........] - ETA: 0s
15974400/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 09:15:07.331564: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 09:15:07.335861: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-10 09:15:07.336498: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563195152690 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 09:15:07.336525: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7126 - accuracy: 0.4970
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8148 - accuracy: 0.4903 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5532 - accuracy: 0.5074
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6078 - accuracy: 0.5038
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7058 - accuracy: 0.4974
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
11000/25000 [============>.................] - ETA: 4s - loss: 7.6875 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6560 - accuracy: 0.5007
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6721 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 3s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6561 - accuracy: 0.5007
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6547 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6602 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6566 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 09:15:25.092736
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 09:15:25.092736  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 09:15:31.983391: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 09:15:31.988867: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-10 09:15:31.989412: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558a3c2ef390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 09:15:31.989554: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1880507cc0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3520 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.2538 - val_crf_viterbi_accuracy: 0.1867

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f187796a5c0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.5133 - accuracy: 0.5100
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7970 - accuracy: 0.4915
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.9222 - accuracy: 0.4833 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.8545 - accuracy: 0.4877
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7586 - accuracy: 0.4940
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6724 - accuracy: 0.4996
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6326 - accuracy: 0.5022
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 4s - loss: 7.6234 - accuracy: 0.5028
12000/25000 [=============>................] - ETA: 4s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6938 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7050 - accuracy: 0.4975
15000/25000 [=================>............] - ETA: 3s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7069 - accuracy: 0.4974
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7009 - accuracy: 0.4978
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7050 - accuracy: 0.4975
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7070 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6935 - accuracy: 0.4983
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6863 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6973 - accuracy: 0.4980
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6840 - accuracy: 0.4989
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6871 - accuracy: 0.4987
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1831609208> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:02<74:20:13, 3.22kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:02<52:15:55, 4.58kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<36:38:04, 6.54kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<25:38:16, 9.33kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:03<17:53:36, 13.3kB/s].vector_cache/glove.6B.zip:   1%|          | 7.63M/862M [00:03<12:28:06, 19.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:03<8:40:31, 27.2kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.5M/862M [00:03<6:02:56, 38.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:03<4:12:31, 55.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:03<2:56:08, 79.2kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.1M/862M [00:03<2:02:32, 113kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 33.9M/862M [00:03<1:25:38, 161kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:03<59:37, 230kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:03<41:43, 327kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:04<29:08, 466kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:04<20:24, 663kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:04<15:34, 867kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:06<12:45, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<10:47, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:06<07:56, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<05:42, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:08<24:25, 547kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:08<18:51, 708kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:08<13:34, 983kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:10<12:02, 1.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:10<09:55, 1.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:10<07:16, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:12<07:54, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:12<06:52, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:12<05:08, 2.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:14<06:40, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:14<07:20, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:14<05:47, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:16<06:10, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:16<05:38, 2.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:16<04:12, 3.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:18<06:01, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:18<06:50, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:18<05:21, 2.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<03:53, 3.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:20<12:17, 1.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:20<09:57, 1.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:20<07:14, 1.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:22<08:04, 1.59MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<08:16, 1.56MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:22<06:21, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<04:34, 2.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:24<15:02, 852kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:24<11:48, 1.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:24<08:34, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:26<08:59, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:26<07:33, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:26<05:35, 2.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<06:54, 1.83MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<06:07, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<04:36, 2.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<06:11, 2.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<05:36, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<04:11, 3.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<05:52, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<05:23, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<04:05, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<05:48, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<06:35, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<05:09, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<03:44, 3.31MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<13:05, 947kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<20:22, 609kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<15:38, 788kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<12:18, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<08:56, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<08:53, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<08:42, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<06:43, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<04:50, 2.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<1:30:33, 135kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<1:04:35, 189kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<45:25, 268kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<34:33, 351kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<26:41, 454kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<19:10, 631kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<13:33, 891kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<15:08, 796kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<11:48, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<08:33, 1.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<08:46, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<07:21, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<06:36, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<05:50, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<04:23, 2.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<05:52, 2.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<05:19, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<03:57, 2.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<05:35, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<06:17, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<05:00, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<05:23, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<04:57, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<03:45, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<05:21, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<04:55, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<03:41, 3.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<05:20, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<04:43, 2.45MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<03:31, 3.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<02:39, 4.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<1:18:08, 147kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<57:02, 202kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<40:28, 284kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<30:03, 380kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<22:12, 515kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<15:48, 721kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<13:40, 831kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<11:52, 957kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<08:48, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<06:17, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<11:13, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<08:59, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<06:34, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<07:13, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<07:20, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<05:42, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:07, 2.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<1:34:04, 119kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<1:06:57, 167kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<46:59, 237kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<35:24, 313kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<27:02, 410kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<19:28, 569kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<15:21, 718kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<11:52, 927kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:15<08:31, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<08:32, 1.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<08:13, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<06:18, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<06:09, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<05:25, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<04:01, 2.70MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<05:22, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<05:51, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<04:33, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<03:19, 3.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<08:57, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<07:22, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<05:25, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<06:17, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<06:35, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<05:04, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<03:40, 2.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<08:16, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<06:52, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<05:02, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<06:00, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<06:19, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<04:53, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<03:33, 2.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<07:21, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<06:12, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:33, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<05:38, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<04:59, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<03:45, 2.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<05:04, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<05:38, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<04:23, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<03:13, 3.19MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<06:15, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<05:24, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<04:02, 2.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:13, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<05:38, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<04:23, 2.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<03:11, 3.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<09:06, 1.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<07:25, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<05:26, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<06:09, 1.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<06:27, 1.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<04:58, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<03:35, 2.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<09:08, 1.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<07:27, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<05:26, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<06:05, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<06:16, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<04:49, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<03:29, 2.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<08:31, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<06:47, 1.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<04:59, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<05:48, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<06:04, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<03:25, 2.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<9:18:27, 17.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<6:31:39, 24.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<4:33:38, 35.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<3:13:06, 50.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<2:16:03, 70.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<1:35:13, 101kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<1:08:41, 140kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<50:00, 192kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<35:26, 270kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<26:13, 363kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<19:18, 493kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<13:43, 691kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<11:46, 802kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<10:09, 930kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<07:30, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:22, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<07:59, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<06:33, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:49, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<05:33, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<04:49, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<03:34, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<04:40, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<05:08, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<03:59, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<02:54, 3.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<06:15, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<05:18, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<03:54, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<04:51, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<05:15, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<04:08, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<04:20, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<03:58, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<02:58, 3.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<04:44, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<03:46, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<04:04, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<03:45, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<02:51, 3.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<04:03, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<03:44, 2.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<02:47, 3.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<04:02, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<03:42, 2.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<02:46, 3.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<03:41, 2.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<02:47, 3.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<04:00, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<04:33, 1.89MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<03:36, 2.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<03:54, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<03:37, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<02:43, 3.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<04:26, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<03:31, 2.40MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:49, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<03:31, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<02:40, 3.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:49, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<03:31, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<02:39, 3.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<03:49, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<04:21, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<03:27, 2.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<02:30, 3.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<7:55:58, 17.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<5:33:46, 24.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<3:53:06, 35.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<2:44:22, 49.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<1:56:39, 69.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<1:21:53, 99.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<57:12, 142kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<43:31, 186kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<31:15, 258kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<22:00, 365kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<17:13, 465kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<12:52, 622kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<09:08, 872kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<08:16, 960kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<07:23, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:30, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<03:58, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<05:46, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<04:51, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:33, 2.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<04:19, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<04:36, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<03:46, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<03:26, 2.25MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<02:35, 2.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<03:35, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<04:04, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<03:10, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<02:18, 3.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<05:24, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<04:32, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<03:21, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<04:06, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<04:24, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<03:24, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:27, 3.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<07:45, 962kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<06:11, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:30, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<04:52, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<04:55, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<03:48, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<03:50, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<02:33, 2.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<03:28, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<03:10, 2.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<02:22, 3.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<03:21, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:05<03:04, 2.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<02:19, 3.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<03:18, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<03:46, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<02:56, 2.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:07, 3.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<07:31, 937kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<05:58, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<04:19, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<04:39, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:11<04:39, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<03:33, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:34, 2.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<05:41, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<04:41, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<03:25, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<04:00, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<04:11, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:15<03:16, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<03:22, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<03:03, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<02:18, 2.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<03:12, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<02:48, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<02:07, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:02, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<03:28, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<02:45, 2.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<02:58, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<02:45, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<02:04, 3.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<02:58, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<03:23, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<02:41, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<02:55, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<02:41, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<02:02, 3.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<02:54, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<03:19, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<02:35, 2.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<01:53, 3.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<04:13, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<03:35, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<02:40, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<03:19, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<03:35, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<02:49, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<02:57, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<02:35, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<01:55, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<01:25, 4.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<25:33, 238kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<19:06, 318kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<13:36, 446kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<09:34, 631kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<08:57, 671kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<06:53, 873kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:56, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<04:49, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<04:34, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:27, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:28, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<09:07, 645kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<06:59, 840kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<05:01, 1.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<04:51, 1.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:59, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:55, 1.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:46<03:24, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:32, 1.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:43, 2.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<01:57, 2.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:48<05:06, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<04:09, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:02, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<03:32, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<02:35, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<01:51, 2.98MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<34:05, 163kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<24:23, 227kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<17:08, 321kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<13:13, 414kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<10:21, 528kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<07:28, 730kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:14, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<18:08, 298kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<13:14, 407kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<09:21, 573kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<07:46, 687kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<05:58, 892kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<04:17, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<04:14, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<04:01, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<03:05, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<02:58, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<02:37, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<01:57, 2.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:33, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:49, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:11, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<01:34, 3.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<06:10, 820kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<04:50, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<03:28, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<03:35, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<03:32, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<02:43, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<02:41, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<02:22, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<01:46, 2.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:22, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<02:38, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:03, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<01:28, 3.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<04:39, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<03:45, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<02:43, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<02:59, 1.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<03:02, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:22, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<02:23, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<01:35, 2.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<02:11, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<01:59, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<01:29, 3.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:20, 1.93MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<01:39, 2.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<04:40, 951kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<03:43, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:42, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:54, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:28, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:49, 2.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:18, 1.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:28, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:56, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<02:02, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<01:51, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:23, 3.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<01:56, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<01:46, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<01:20, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<01:54, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<02:09, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<01:41, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:13, 3.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<03:45, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<02:58, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:09, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:33, 2.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<08:49, 449kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<06:34, 602kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<04:39, 845kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<04:09, 937kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<03:41, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:45, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:31, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:34, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:41<01:58, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:24, 2.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<03:39, 1.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<02:56, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:07, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<02:20, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<02:23, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<01:51, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:52, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:40, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<01:15, 2.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:42, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:52, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<01:28, 2.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:03, 3.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<03:35, 969kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<02:52, 1.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<02:04, 1.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<02:14, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<02:47, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<02:19, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<01:57, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:26, 2.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:45, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:33, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:10, 2.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:34, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:22, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<01:02, 3.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<00:45, 4.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<21:19, 147kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<15:13, 206kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<10:38, 292kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<08:05, 379kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<06:16, 488kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<04:30, 677kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:08, 957kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<05:06, 588kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<03:52, 774kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<02:45, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<02:35, 1.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<02:26, 1.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:51, 1.58MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:18, 2.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<11:17, 254kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<08:10, 349kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<05:44, 493kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:10<04:37, 605kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<03:47, 735kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<02:47, 997kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:21, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<01:55, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:24, 1.92MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:14<01:35, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:39, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<01:15, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<00:54, 2.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<02:00, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:40, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:13, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<01:26, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<01:30, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:09, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<00:50, 2.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<01:58, 1.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<01:37, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:11, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:22, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:26, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:07, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:47, 2.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<03:58, 583kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<03:00, 768kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<02:08, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<02:00, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:35, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:08, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:49, 2.64MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<12:23, 176kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<09:06, 239kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<06:26, 335kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:26, 476kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<20:45, 102kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<14:32, 144kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<10:27, 200kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<07:09, 285kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<18:11, 112kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<12:53, 158kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<08:57, 224kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<06:37, 298kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<05:02, 391kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<03:35, 544kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<02:45, 690kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<02:07, 896kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:30, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:27, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:23, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<01:03, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:44, 2.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<06:35, 268kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<04:46, 369kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<03:20, 520kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<02:41, 633kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<02:01, 840kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:25, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<01:22, 1.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<01:07, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:48, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<00:55, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:57, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:44, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:44, 2.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:40, 2.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:29, 2.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:40, 2.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:45, 1.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:35, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:37, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:34, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:25, 3.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:35, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:32, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:54<00:23, 3.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:33, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:38, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:30, 2.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:21, 3.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<1:06:54, 17.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<46:41, 24.5kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<31:58, 35.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<21:54, 49.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<15:21, 70.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<10:30, 99.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<07:21, 138kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<05:12, 193kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<03:38, 272kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<02:36, 363kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<01:54, 492kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<01:19, 690kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<01:05, 801kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:56, 928kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:41, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:28, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:37, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:30, 1.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:22, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:25, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:26, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:20, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:13, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<01:19, 508kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:59, 675kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:40, 941kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:35, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:31, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:23, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:20, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:17, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:12, 2.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:14, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:15, 1.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:12, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:11, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:10, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:07, 3.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:08, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:05, 3.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:08, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:06, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:03, 3.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<01:25, 134kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:58, 188kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:35, 267kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:20, 350kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:15, 455kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:10, 629kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:04, 785kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:02, 1.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.38MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 657/400000 [00:00<01:00, 6563.56it/s]  0%|          | 1406/400000 [00:00<00:58, 6812.80it/s]  1%|          | 2133/400000 [00:00<00:57, 6943.26it/s]  1%|          | 2861/400000 [00:00<00:56, 7039.89it/s]  1%|          | 3608/400000 [00:00<00:55, 7161.59it/s]  1%|          | 4333/400000 [00:00<00:55, 7186.22it/s]  1%|▏         | 5089/400000 [00:00<00:54, 7292.73it/s]  1%|▏         | 5838/400000 [00:00<00:53, 7350.49it/s]  2%|▏         | 6588/400000 [00:00<00:53, 7393.73it/s]  2%|▏         | 7318/400000 [00:01<00:53, 7363.31it/s]  2%|▏         | 8037/400000 [00:01<00:54, 7217.30it/s]  2%|▏         | 8773/400000 [00:01<00:53, 7258.38it/s]  2%|▏         | 9508/400000 [00:01<00:53, 7282.95it/s]  3%|▎         | 10232/400000 [00:01<00:53, 7268.47it/s]  3%|▎         | 11002/400000 [00:01<00:52, 7392.42it/s]  3%|▎         | 11770/400000 [00:01<00:51, 7475.28it/s]  3%|▎         | 12521/400000 [00:01<00:51, 7485.49it/s]  3%|▎         | 13269/400000 [00:01<00:51, 7448.43it/s]  4%|▎         | 14014/400000 [00:01<00:52, 7401.32it/s]  4%|▎         | 14754/400000 [00:02<00:52, 7357.76it/s]  4%|▍         | 15490/400000 [00:02<00:52, 7313.94it/s]  4%|▍         | 16222/400000 [00:02<00:52, 7298.25it/s]  4%|▍         | 16952/400000 [00:02<00:52, 7296.99it/s]  4%|▍         | 17686/400000 [00:02<00:52, 7308.97it/s]  5%|▍         | 18422/400000 [00:02<00:52, 7320.67it/s]  5%|▍         | 19171/400000 [00:02<00:51, 7367.96it/s]  5%|▍         | 19917/400000 [00:02<00:51, 7393.13it/s]  5%|▌         | 20669/400000 [00:02<00:51, 7428.39it/s]  5%|▌         | 21412/400000 [00:02<00:51, 7410.66it/s]  6%|▌         | 22168/400000 [00:03<00:50, 7454.59it/s]  6%|▌         | 22914/400000 [00:03<00:50, 7422.30it/s]  6%|▌         | 23657/400000 [00:03<00:50, 7416.65it/s]  6%|▌         | 24412/400000 [00:03<00:50, 7455.89it/s]  6%|▋         | 25158/400000 [00:03<00:50, 7387.62it/s]  6%|▋         | 25905/400000 [00:03<00:50, 7409.40it/s]  7%|▋         | 26660/400000 [00:03<00:50, 7449.14it/s]  7%|▋         | 27406/400000 [00:03<00:50, 7450.17it/s]  7%|▋         | 28152/400000 [00:03<00:50, 7433.13it/s]  7%|▋         | 28896/400000 [00:03<00:52, 7034.22it/s]  7%|▋         | 29604/400000 [00:04<00:54, 6848.91it/s]  8%|▊         | 30294/400000 [00:04<00:54, 6734.02it/s]  8%|▊         | 31006/400000 [00:04<00:53, 6845.09it/s]  8%|▊         | 31768/400000 [00:04<00:52, 7059.27it/s]  8%|▊         | 32512/400000 [00:04<00:51, 7169.20it/s]  8%|▊         | 33261/400000 [00:04<00:50, 7261.04it/s]  9%|▊         | 34029/400000 [00:04<00:49, 7379.33it/s]  9%|▊         | 34780/400000 [00:04<00:49, 7417.77it/s]  9%|▉         | 35558/400000 [00:04<00:48, 7522.22it/s]  9%|▉         | 36313/400000 [00:04<00:48, 7528.31it/s]  9%|▉         | 37067/400000 [00:05<00:49, 7370.33it/s]  9%|▉         | 37828/400000 [00:05<00:48, 7438.27it/s] 10%|▉         | 38584/400000 [00:05<00:48, 7472.90it/s] 10%|▉         | 39333/400000 [00:05<00:48, 7446.98it/s] 10%|█         | 40079/400000 [00:05<00:49, 7315.91it/s] 10%|█         | 40812/400000 [00:05<00:49, 7295.02it/s] 10%|█         | 41556/400000 [00:05<00:48, 7336.57it/s] 11%|█         | 42312/400000 [00:05<00:48, 7402.06it/s] 11%|█         | 43069/400000 [00:05<00:47, 7450.05it/s] 11%|█         | 43815/400000 [00:05<00:48, 7330.88it/s] 11%|█         | 44549/400000 [00:06<00:48, 7281.80it/s] 11%|█▏        | 45287/400000 [00:06<00:48, 7308.83it/s] 12%|█▏        | 46025/400000 [00:06<00:48, 7327.85it/s] 12%|█▏        | 46767/400000 [00:06<00:48, 7354.65it/s] 12%|█▏        | 47509/400000 [00:06<00:47, 7373.59it/s] 12%|█▏        | 48288/400000 [00:06<00:46, 7493.57it/s] 12%|█▏        | 49050/400000 [00:06<00:46, 7529.77it/s] 12%|█▏        | 49804/400000 [00:06<00:47, 7379.81it/s] 13%|█▎        | 50562/400000 [00:06<00:46, 7436.40it/s] 13%|█▎        | 51316/400000 [00:06<00:46, 7464.79it/s] 13%|█▎        | 52064/400000 [00:07<00:46, 7412.30it/s] 13%|█▎        | 52806/400000 [00:07<00:47, 7367.91it/s] 13%|█▎        | 53544/400000 [00:07<00:47, 7218.94it/s] 14%|█▎        | 54274/400000 [00:07<00:47, 7241.51it/s] 14%|█▍        | 55020/400000 [00:07<00:47, 7304.12it/s] 14%|█▍        | 55752/400000 [00:07<00:47, 7301.17it/s] 14%|█▍        | 56494/400000 [00:07<00:46, 7333.84it/s] 14%|█▍        | 57228/400000 [00:07<00:47, 7281.11it/s] 14%|█▍        | 57976/400000 [00:07<00:46, 7338.36it/s] 15%|█▍        | 58713/400000 [00:08<00:46, 7346.23it/s] 15%|█▍        | 59448/400000 [00:08<00:47, 7179.77it/s] 15%|█▌        | 60179/400000 [00:08<00:47, 7217.67it/s] 15%|█▌        | 60902/400000 [00:08<00:49, 6842.74it/s] 15%|█▌        | 61628/400000 [00:08<00:48, 6962.43it/s] 16%|█▌        | 62357/400000 [00:08<00:47, 7055.32it/s] 16%|█▌        | 63087/400000 [00:08<00:47, 7125.46it/s] 16%|█▌        | 63807/400000 [00:08<00:47, 7146.01it/s] 16%|█▌        | 64543/400000 [00:08<00:46, 7206.84it/s] 16%|█▋        | 65287/400000 [00:08<00:46, 7274.07it/s] 17%|█▋        | 66033/400000 [00:09<00:45, 7328.36it/s] 17%|█▋        | 66767/400000 [00:09<00:45, 7288.30it/s] 17%|█▋        | 67534/400000 [00:09<00:44, 7397.51it/s] 17%|█▋        | 68290/400000 [00:09<00:44, 7444.49it/s] 17%|█▋        | 69036/400000 [00:09<00:44, 7445.20it/s] 17%|█▋        | 69781/400000 [00:09<00:44, 7445.42it/s] 18%|█▊        | 70526/400000 [00:09<00:44, 7439.70it/s] 18%|█▊        | 71271/400000 [00:09<00:44, 7322.05it/s] 18%|█▊        | 72020/400000 [00:09<00:44, 7369.83it/s] 18%|█▊        | 72758/400000 [00:09<00:44, 7348.35it/s] 18%|█▊        | 73494/400000 [00:10<00:44, 7306.03it/s] 19%|█▊        | 74225/400000 [00:10<00:44, 7303.49it/s] 19%|█▊        | 74968/400000 [00:10<00:44, 7339.70it/s] 19%|█▉        | 75703/400000 [00:10<00:48, 6742.14it/s] 19%|█▉        | 76403/400000 [00:10<00:47, 6816.10it/s] 19%|█▉        | 77162/400000 [00:10<00:45, 7029.25it/s] 19%|█▉        | 77930/400000 [00:10<00:44, 7212.34it/s] 20%|█▉        | 78662/400000 [00:10<00:44, 7244.00it/s] 20%|█▉        | 79391/400000 [00:10<00:44, 7216.35it/s] 20%|██        | 80150/400000 [00:10<00:43, 7324.02it/s] 20%|██        | 80886/400000 [00:11<00:43, 7333.29it/s] 20%|██        | 81622/400000 [00:11<00:43, 7266.56it/s] 21%|██        | 82369/400000 [00:11<00:43, 7325.99it/s] 21%|██        | 83103/400000 [00:11<00:43, 7300.29it/s] 21%|██        | 83845/400000 [00:11<00:43, 7333.50it/s] 21%|██        | 84585/400000 [00:11<00:42, 7352.62it/s] 21%|██▏       | 85334/400000 [00:11<00:42, 7392.55it/s] 22%|██▏       | 86077/400000 [00:11<00:42, 7401.24it/s] 22%|██▏       | 86818/400000 [00:11<00:43, 7125.68it/s] 22%|██▏       | 87533/400000 [00:11<00:44, 7092.31it/s] 22%|██▏       | 88290/400000 [00:12<00:43, 7220.39it/s] 22%|██▏       | 89029/400000 [00:12<00:42, 7270.06it/s] 22%|██▏       | 89783/400000 [00:12<00:42, 7348.20it/s] 23%|██▎       | 90519/400000 [00:12<00:42, 7329.20it/s] 23%|██▎       | 91285/400000 [00:12<00:41, 7422.03it/s] 23%|██▎       | 92055/400000 [00:12<00:41, 7500.56it/s] 23%|██▎       | 92814/400000 [00:12<00:40, 7527.02it/s] 23%|██▎       | 93580/400000 [00:12<00:40, 7564.26it/s] 24%|██▎       | 94337/400000 [00:12<00:41, 7419.28it/s] 24%|██▍       | 95080/400000 [00:13<00:41, 7389.41it/s] 24%|██▍       | 95832/400000 [00:13<00:40, 7427.01it/s] 24%|██▍       | 96593/400000 [00:13<00:40, 7480.48it/s] 24%|██▍       | 97361/400000 [00:13<00:40, 7537.46it/s] 25%|██▍       | 98120/400000 [00:13<00:39, 7551.88it/s] 25%|██▍       | 98876/400000 [00:13<00:40, 7475.12it/s] 25%|██▍       | 99624/400000 [00:13<00:40, 7403.54it/s] 25%|██▌       | 100365/400000 [00:13<00:40, 7402.19it/s] 25%|██▌       | 101132/400000 [00:13<00:39, 7478.39it/s] 25%|██▌       | 101881/400000 [00:13<00:40, 7435.83it/s] 26%|██▌       | 102650/400000 [00:14<00:39, 7509.33it/s] 26%|██▌       | 103402/400000 [00:14<00:39, 7510.22it/s] 26%|██▌       | 104154/400000 [00:14<00:40, 7395.22it/s] 26%|██▌       | 104895/400000 [00:14<00:40, 7358.19it/s] 26%|██▋       | 105632/400000 [00:14<00:40, 7279.87it/s] 27%|██▋       | 106397/400000 [00:14<00:39, 7386.59it/s] 27%|██▋       | 107146/400000 [00:14<00:39, 7415.65it/s] 27%|██▋       | 107908/400000 [00:14<00:39, 7473.14it/s] 27%|██▋       | 108656/400000 [00:14<00:40, 7271.73it/s] 27%|██▋       | 109407/400000 [00:14<00:39, 7338.97it/s] 28%|██▊       | 110143/400000 [00:15<00:39, 7256.63it/s] 28%|██▊       | 110873/400000 [00:15<00:39, 7268.14it/s] 28%|██▊       | 111615/400000 [00:15<00:39, 7309.68it/s] 28%|██▊       | 112353/400000 [00:15<00:39, 7327.89it/s] 28%|██▊       | 113087/400000 [00:15<00:40, 7111.80it/s] 28%|██▊       | 113866/400000 [00:15<00:39, 7300.82it/s] 29%|██▊       | 114623/400000 [00:15<00:38, 7377.70it/s] 29%|██▉       | 115415/400000 [00:15<00:37, 7532.16it/s] 29%|██▉       | 116192/400000 [00:15<00:37, 7600.45it/s] 29%|██▉       | 116987/400000 [00:15<00:36, 7698.95it/s] 29%|██▉       | 117759/400000 [00:16<00:37, 7453.38it/s] 30%|██▉       | 118534/400000 [00:16<00:37, 7537.86it/s] 30%|██▉       | 119301/400000 [00:16<00:37, 7574.69it/s] 30%|███       | 120061/400000 [00:16<00:37, 7521.10it/s] 30%|███       | 120815/400000 [00:16<00:37, 7511.97it/s] 30%|███       | 121568/400000 [00:16<00:37, 7463.87it/s] 31%|███       | 122316/400000 [00:16<00:37, 7424.93it/s] 31%|███       | 123059/400000 [00:16<00:37, 7420.83it/s] 31%|███       | 123802/400000 [00:16<00:38, 7208.82it/s] 31%|███       | 124562/400000 [00:16<00:37, 7320.09it/s] 31%|███▏      | 125318/400000 [00:17<00:37, 7389.87it/s] 32%|███▏      | 126073/400000 [00:17<00:36, 7434.79it/s] 32%|███▏      | 126845/400000 [00:17<00:36, 7517.65it/s] 32%|███▏      | 127598/400000 [00:17<00:37, 7311.68it/s] 32%|███▏      | 128365/400000 [00:17<00:36, 7413.56it/s] 32%|███▏      | 129136/400000 [00:17<00:36, 7498.15it/s] 32%|███▏      | 129888/400000 [00:17<00:36, 7480.10it/s] 33%|███▎      | 130643/400000 [00:17<00:35, 7499.36it/s] 33%|███▎      | 131394/400000 [00:17<00:36, 7459.31it/s] 33%|███▎      | 132141/400000 [00:17<00:35, 7453.95it/s] 33%|███▎      | 132887/400000 [00:18<00:35, 7449.09it/s] 33%|███▎      | 133665/400000 [00:18<00:35, 7545.00it/s] 34%|███▎      | 134420/400000 [00:18<00:35, 7532.61it/s] 34%|███▍      | 135174/400000 [00:18<00:35, 7476.15it/s] 34%|███▍      | 135940/400000 [00:18<00:35, 7529.44it/s] 34%|███▍      | 136694/400000 [00:18<00:35, 7470.76it/s] 34%|███▍      | 137485/400000 [00:18<00:34, 7595.01it/s] 35%|███▍      | 138246/400000 [00:18<00:34, 7596.95it/s] 35%|███▍      | 139007/400000 [00:18<00:34, 7552.36it/s] 35%|███▍      | 139777/400000 [00:19<00:34, 7594.39it/s] 35%|███▌      | 140537/400000 [00:19<00:34, 7588.80it/s] 35%|███▌      | 141297/400000 [00:19<00:34, 7552.66it/s] 36%|███▌      | 142053/400000 [00:19<00:34, 7487.47it/s] 36%|███▌      | 142803/400000 [00:19<00:35, 7186.34it/s] 36%|███▌      | 143535/400000 [00:19<00:35, 7223.15it/s] 36%|███▌      | 144260/400000 [00:19<00:35, 7194.38it/s] 36%|███▌      | 144993/400000 [00:19<00:35, 7233.06it/s] 36%|███▋      | 145725/400000 [00:19<00:35, 7257.41it/s] 37%|███▋      | 146452/400000 [00:19<00:35, 7231.96it/s] 37%|███▋      | 147176/400000 [00:20<00:35, 7129.63it/s] 37%|███▋      | 147902/400000 [00:20<00:35, 7166.28it/s] 37%|███▋      | 148664/400000 [00:20<00:34, 7294.63it/s] 37%|███▋      | 149427/400000 [00:20<00:33, 7384.34it/s] 38%|███▊      | 150167/400000 [00:20<00:33, 7382.29it/s] 38%|███▊      | 150911/400000 [00:20<00:33, 7397.95it/s] 38%|███▊      | 151652/400000 [00:20<00:34, 7267.01it/s] 38%|███▊      | 152430/400000 [00:20<00:33, 7412.86it/s] 38%|███▊      | 153214/400000 [00:20<00:32, 7533.68it/s] 38%|███▊      | 153976/400000 [00:20<00:32, 7556.51it/s] 39%|███▊      | 154752/400000 [00:21<00:32, 7615.89it/s] 39%|███▉      | 155531/400000 [00:21<00:31, 7667.11it/s] 39%|███▉      | 156299/400000 [00:21<00:31, 7655.44it/s] 39%|███▉      | 157065/400000 [00:21<00:31, 7623.28it/s] 39%|███▉      | 157828/400000 [00:21<00:32, 7528.46it/s] 40%|███▉      | 158582/400000 [00:21<00:32, 7520.01it/s] 40%|███▉      | 159351/400000 [00:21<00:31, 7568.26it/s] 40%|████      | 160110/400000 [00:21<00:31, 7572.88it/s] 40%|████      | 160872/400000 [00:21<00:31, 7584.24it/s] 40%|████      | 161631/400000 [00:21<00:31, 7550.15it/s] 41%|████      | 162387/400000 [00:22<00:31, 7552.42it/s] 41%|████      | 163143/400000 [00:22<00:32, 7217.40it/s] 41%|████      | 163907/400000 [00:22<00:32, 7337.87it/s] 41%|████      | 164644/400000 [00:22<00:32, 7308.05it/s] 41%|████▏     | 165377/400000 [00:22<00:32, 7184.53it/s] 42%|████▏     | 166102/400000 [00:22<00:32, 7201.69it/s] 42%|████▏     | 166824/400000 [00:22<00:32, 7202.51it/s] 42%|████▏     | 167565/400000 [00:22<00:32, 7263.36it/s] 42%|████▏     | 168316/400000 [00:22<00:31, 7333.95it/s] 42%|████▏     | 169059/400000 [00:22<00:31, 7361.33it/s] 42%|████▏     | 169804/400000 [00:23<00:31, 7387.13it/s] 43%|████▎     | 170544/400000 [00:23<00:32, 7091.39it/s] 43%|████▎     | 171311/400000 [00:23<00:31, 7255.13it/s] 43%|████▎     | 172065/400000 [00:23<00:31, 7337.32it/s] 43%|████▎     | 172809/400000 [00:23<00:30, 7367.13it/s] 43%|████▎     | 173548/400000 [00:23<00:31, 7236.95it/s] 44%|████▎     | 174312/400000 [00:23<00:30, 7353.18it/s] 44%|████▍     | 175088/400000 [00:23<00:30, 7469.75it/s] 44%|████▍     | 175851/400000 [00:23<00:29, 7515.37it/s] 44%|████▍     | 176607/400000 [00:23<00:29, 7526.61it/s] 44%|████▍     | 177385/400000 [00:24<00:29, 7599.16it/s] 45%|████▍     | 178151/400000 [00:24<00:29, 7616.39it/s] 45%|████▍     | 178948/400000 [00:24<00:28, 7716.72it/s] 45%|████▍     | 179726/400000 [00:24<00:28, 7735.06it/s] 45%|████▌     | 180500/400000 [00:24<00:28, 7661.32it/s] 45%|████▌     | 181267/400000 [00:24<00:28, 7550.28it/s] 46%|████▌     | 182023/400000 [00:24<00:29, 7320.41it/s] 46%|████▌     | 182758/400000 [00:24<00:30, 7165.35it/s] 46%|████▌     | 183482/400000 [00:24<00:30, 7187.42it/s] 46%|████▌     | 184227/400000 [00:25<00:29, 7263.92it/s] 46%|████▌     | 184955/400000 [00:25<00:29, 7228.50it/s] 46%|████▋     | 185679/400000 [00:25<00:30, 7136.60it/s] 47%|████▋     | 186428/400000 [00:25<00:29, 7236.76it/s] 47%|████▋     | 187153/400000 [00:25<00:29, 7207.24it/s] 47%|████▋     | 187901/400000 [00:25<00:29, 7286.75it/s] 47%|████▋     | 188652/400000 [00:25<00:28, 7351.71it/s] 47%|████▋     | 189402/400000 [00:25<00:28, 7394.50it/s] 48%|████▊     | 190173/400000 [00:25<00:28, 7482.10it/s] 48%|████▊     | 190945/400000 [00:25<00:27, 7551.79it/s] 48%|████▊     | 191703/400000 [00:26<00:27, 7559.34it/s] 48%|████▊     | 192460/400000 [00:26<00:28, 7381.17it/s] 48%|████▊     | 193200/400000 [00:26<00:28, 7228.37it/s] 48%|████▊     | 193925/400000 [00:26<00:28, 7116.89it/s] 49%|████▊     | 194652/400000 [00:26<00:28, 7160.70it/s] 49%|████▉     | 195383/400000 [00:26<00:28, 7204.48it/s] 49%|████▉     | 196114/400000 [00:26<00:28, 7233.62it/s] 49%|████▉     | 196855/400000 [00:26<00:27, 7284.43it/s] 49%|████▉     | 197603/400000 [00:26<00:27, 7341.42it/s] 50%|████▉     | 198338/400000 [00:26<00:27, 7253.16it/s] 50%|████▉     | 199064/400000 [00:27<00:27, 7235.51it/s] 50%|████▉     | 199788/400000 [00:27<00:27, 7219.16it/s] 50%|█████     | 200539/400000 [00:27<00:27, 7301.87it/s] 50%|█████     | 201293/400000 [00:27<00:26, 7371.71it/s] 51%|█████     | 202057/400000 [00:27<00:26, 7450.11it/s] 51%|█████     | 202803/400000 [00:27<00:26, 7452.36it/s] 51%|█████     | 203555/400000 [00:27<00:26, 7471.28it/s] 51%|█████     | 204303/400000 [00:27<00:26, 7398.16it/s] 51%|█████▏    | 205048/400000 [00:27<00:26, 7411.35it/s] 51%|█████▏    | 205792/400000 [00:27<00:26, 7417.37it/s] 52%|█████▏    | 206534/400000 [00:28<00:26, 7391.04it/s] 52%|█████▏    | 207274/400000 [00:28<00:26, 7209.90it/s] 52%|█████▏    | 208018/400000 [00:28<00:26, 7275.19it/s] 52%|█████▏    | 208754/400000 [00:28<00:26, 7300.03it/s] 52%|█████▏    | 209488/400000 [00:28<00:26, 7307.65it/s] 53%|█████▎    | 210220/400000 [00:28<00:26, 7241.51it/s] 53%|█████▎    | 210945/400000 [00:28<00:26, 7226.21it/s] 53%|█████▎    | 211668/400000 [00:28<00:26, 7218.47it/s] 53%|█████▎    | 212417/400000 [00:28<00:25, 7295.39it/s] 53%|█████▎    | 213181/400000 [00:28<00:25, 7393.98it/s] 53%|█████▎    | 213952/400000 [00:29<00:24, 7484.93it/s] 54%|█████▎    | 214702/400000 [00:29<00:24, 7431.25it/s] 54%|█████▍    | 215470/400000 [00:29<00:24, 7502.73it/s] 54%|█████▍    | 216237/400000 [00:29<00:24, 7551.08it/s] 54%|█████▍    | 216993/400000 [00:29<00:24, 7485.32it/s] 54%|█████▍    | 217742/400000 [00:29<00:24, 7458.80it/s] 55%|█████▍    | 218489/400000 [00:29<00:24, 7401.37it/s] 55%|█████▍    | 219230/400000 [00:29<00:24, 7295.23it/s] 55%|█████▍    | 219976/400000 [00:29<00:24, 7343.18it/s] 55%|█████▌    | 220711/400000 [00:29<00:24, 7337.62it/s] 55%|█████▌    | 221458/400000 [00:30<00:24, 7375.59it/s] 56%|█████▌    | 222196/400000 [00:30<00:24, 7340.05it/s] 56%|█████▌    | 222934/400000 [00:30<00:24, 7350.76it/s] 56%|█████▌    | 223670/400000 [00:30<00:23, 7349.17it/s] 56%|█████▌    | 224406/400000 [00:30<00:25, 6995.23it/s] 56%|█████▋    | 225162/400000 [00:30<00:24, 7154.42it/s] 56%|█████▋    | 225881/400000 [00:30<00:24, 7035.09it/s] 57%|█████▋    | 226645/400000 [00:30<00:24, 7205.60it/s] 57%|█████▋    | 227405/400000 [00:30<00:23, 7317.27it/s] 57%|█████▋    | 228140/400000 [00:31<00:24, 7095.19it/s] 57%|█████▋    | 228899/400000 [00:31<00:23, 7236.50it/s] 57%|█████▋    | 229639/400000 [00:31<00:23, 7282.52it/s] 58%|█████▊    | 230401/400000 [00:31<00:22, 7378.10it/s] 58%|█████▊    | 231165/400000 [00:31<00:22, 7451.97it/s] 58%|█████▊    | 231941/400000 [00:31<00:22, 7541.37it/s] 58%|█████▊    | 232727/400000 [00:31<00:21, 7633.33it/s] 58%|█████▊    | 233492/400000 [00:31<00:22, 7556.70it/s] 59%|█████▊    | 234249/400000 [00:31<00:21, 7536.49it/s] 59%|█████▉    | 235021/400000 [00:31<00:21, 7589.58it/s] 59%|█████▉    | 235781/400000 [00:32<00:22, 7149.01it/s] 59%|█████▉    | 236531/400000 [00:32<00:22, 7249.79it/s] 59%|█████▉    | 237273/400000 [00:32<00:22, 7299.02it/s] 60%|█████▉    | 238012/400000 [00:32<00:22, 7325.61it/s] 60%|█████▉    | 238770/400000 [00:32<00:21, 7397.56it/s] 60%|█████▉    | 239512/400000 [00:32<00:21, 7359.94it/s] 60%|██████    | 240250/400000 [00:32<00:22, 7186.63it/s] 60%|██████    | 240986/400000 [00:32<00:21, 7235.77it/s] 60%|██████    | 241726/400000 [00:32<00:21, 7282.29it/s] 61%|██████    | 242472/400000 [00:32<00:21, 7332.52it/s] 61%|██████    | 243225/400000 [00:33<00:21, 7388.73it/s] 61%|██████    | 243995/400000 [00:33<00:20, 7477.58it/s] 61%|██████    | 244744/400000 [00:33<00:20, 7446.72it/s] 61%|██████▏   | 245492/400000 [00:33<00:20, 7455.28it/s] 62%|██████▏   | 246238/400000 [00:33<00:20, 7442.73it/s] 62%|██████▏   | 246986/400000 [00:33<00:20, 7452.00it/s] 62%|██████▏   | 247732/400000 [00:33<00:20, 7356.02it/s] 62%|██████▏   | 248469/400000 [00:33<00:20, 7275.02it/s] 62%|██████▏   | 249220/400000 [00:33<00:20, 7341.61it/s] 62%|██████▏   | 249955/400000 [00:33<00:20, 7330.22it/s] 63%|██████▎   | 250689/400000 [00:34<00:20, 7287.74it/s] 63%|██████▎   | 251419/400000 [00:34<00:20, 7145.98it/s] 63%|██████▎   | 252147/400000 [00:34<00:20, 7185.15it/s] 63%|██████▎   | 252887/400000 [00:34<00:20, 7245.96it/s] 63%|██████▎   | 253662/400000 [00:34<00:19, 7388.89it/s] 64%|██████▎   | 254432/400000 [00:34<00:19, 7478.05it/s] 64%|██████▍   | 255204/400000 [00:34<00:19, 7547.34it/s] 64%|██████▍   | 255960/400000 [00:34<00:19, 7507.17it/s] 64%|██████▍   | 256712/400000 [00:34<00:19, 7452.67it/s] 64%|██████▍   | 257458/400000 [00:34<00:19, 7417.98it/s] 65%|██████▍   | 258219/400000 [00:35<00:18, 7472.78it/s] 65%|██████▍   | 258967/400000 [00:35<00:19, 7391.18it/s] 65%|██████▍   | 259707/400000 [00:35<00:19, 7315.25it/s] 65%|██████▌   | 260443/400000 [00:35<00:19, 7327.54it/s] 65%|██████▌   | 261204/400000 [00:35<00:18, 7408.48it/s] 65%|██████▌   | 261979/400000 [00:35<00:18, 7506.35it/s] 66%|██████▌   | 262755/400000 [00:35<00:18, 7579.94it/s] 66%|██████▌   | 263514/400000 [00:35<00:18, 7559.02it/s] 66%|██████▌   | 264293/400000 [00:35<00:17, 7626.40it/s] 66%|██████▋   | 265057/400000 [00:35<00:17, 7608.65it/s] 66%|██████▋   | 265819/400000 [00:36<00:17, 7575.03it/s] 67%|██████▋   | 266577/400000 [00:36<00:17, 7536.30it/s] 67%|██████▋   | 267331/400000 [00:36<00:17, 7484.29it/s] 67%|██████▋   | 268086/400000 [00:36<00:17, 7503.48it/s] 67%|██████▋   | 268837/400000 [00:36<00:17, 7491.25it/s] 67%|██████▋   | 269592/400000 [00:36<00:17, 7506.92it/s] 68%|██████▊   | 270358/400000 [00:36<00:17, 7549.36it/s] 68%|██████▊   | 271114/400000 [00:36<00:17, 7451.55it/s] 68%|██████▊   | 271867/400000 [00:36<00:17, 7471.12it/s] 68%|██████▊   | 272621/400000 [00:37<00:17, 7491.13it/s] 68%|██████▊   | 273371/400000 [00:37<00:16, 7476.48it/s] 69%|██████▊   | 274137/400000 [00:37<00:16, 7528.07it/s] 69%|██████▊   | 274890/400000 [00:37<00:16, 7464.55it/s] 69%|██████▉   | 275637/400000 [00:37<00:17, 7265.22it/s] 69%|██████▉   | 276379/400000 [00:37<00:16, 7308.88it/s] 69%|██████▉   | 277135/400000 [00:37<00:16, 7379.83it/s] 69%|██████▉   | 277886/400000 [00:37<00:16, 7418.32it/s] 70%|██████▉   | 278629/400000 [00:37<00:16, 7365.71it/s] 70%|██████▉   | 279380/400000 [00:37<00:16, 7407.42it/s] 70%|███████   | 280130/400000 [00:38<00:16, 7434.89it/s] 70%|███████   | 280889/400000 [00:38<00:15, 7480.13it/s] 70%|███████   | 281652/400000 [00:38<00:15, 7522.25it/s] 71%|███████   | 282405/400000 [00:38<00:15, 7458.87it/s] 71%|███████   | 283152/400000 [00:38<00:15, 7444.69it/s] 71%|███████   | 283897/400000 [00:38<00:15, 7440.21it/s] 71%|███████   | 284642/400000 [00:38<00:15, 7418.04it/s] 71%|███████▏  | 285386/400000 [00:38<00:15, 7424.54it/s] 72%|███████▏  | 286129/400000 [00:38<00:15, 7379.97it/s] 72%|███████▏  | 286868/400000 [00:38<00:15, 7201.12it/s] 72%|███████▏  | 287607/400000 [00:39<00:15, 7256.48it/s] 72%|███████▏  | 288334/400000 [00:39<00:15, 7240.29it/s] 72%|███████▏  | 289059/400000 [00:39<00:16, 6825.06it/s] 72%|███████▏  | 289747/400000 [00:39<00:16, 6836.03it/s] 73%|███████▎  | 290473/400000 [00:39<00:15, 6957.82it/s] 73%|███████▎  | 291243/400000 [00:39<00:15, 7164.54it/s] 73%|███████▎  | 291977/400000 [00:39<00:14, 7216.06it/s] 73%|███████▎  | 292702/400000 [00:39<00:14, 7160.94it/s] 73%|███████▎  | 293441/400000 [00:39<00:14, 7227.60it/s] 74%|███████▎  | 294196/400000 [00:39<00:14, 7320.34it/s] 74%|███████▎  | 294952/400000 [00:40<00:14, 7389.00it/s] 74%|███████▍  | 295710/400000 [00:40<00:14, 7443.13it/s] 74%|███████▍  | 296456/400000 [00:40<00:14, 7042.25it/s] 74%|███████▍  | 297191/400000 [00:40<00:14, 7130.98it/s] 74%|███████▍  | 297945/400000 [00:40<00:14, 7247.24it/s] 75%|███████▍  | 298685/400000 [00:40<00:13, 7288.87it/s] 75%|███████▍  | 299455/400000 [00:40<00:13, 7405.54it/s] 75%|███████▌  | 300216/400000 [00:40<00:13, 7463.54it/s] 75%|███████▌  | 300964/400000 [00:40<00:13, 7424.83it/s] 75%|███████▌  | 301725/400000 [00:40<00:13, 7478.12it/s] 76%|███████▌  | 302474/400000 [00:41<00:13, 7407.84it/s] 76%|███████▌  | 303246/400000 [00:41<00:12, 7497.15it/s] 76%|███████▌  | 303997/400000 [00:41<00:12, 7500.13it/s] 76%|███████▌  | 304748/400000 [00:41<00:12, 7499.06it/s] 76%|███████▋  | 305499/400000 [00:41<00:13, 7153.39it/s] 77%|███████▋  | 306257/400000 [00:41<00:12, 7275.85it/s] 77%|███████▋  | 307016/400000 [00:41<00:12, 7366.10it/s] 77%|███████▋  | 307810/400000 [00:41<00:12, 7528.75it/s] 77%|███████▋  | 308566/400000 [00:41<00:12, 7469.56it/s] 77%|███████▋  | 309324/400000 [00:42<00:12, 7499.77it/s] 78%|███████▊  | 310076/400000 [00:42<00:12, 7468.24it/s] 78%|███████▊  | 310827/400000 [00:42<00:11, 7478.97it/s] 78%|███████▊  | 311576/400000 [00:42<00:11, 7464.41it/s] 78%|███████▊  | 312323/400000 [00:42<00:11, 7405.03it/s] 78%|███████▊  | 313083/400000 [00:42<00:11, 7460.63it/s] 78%|███████▊  | 313845/400000 [00:42<00:11, 7505.44it/s] 79%|███████▊  | 314625/400000 [00:42<00:11, 7589.57it/s] 79%|███████▉  | 315385/400000 [00:42<00:11, 7519.46it/s] 79%|███████▉  | 316138/400000 [00:42<00:11, 7190.83it/s] 79%|███████▉  | 316883/400000 [00:43<00:11, 7264.15it/s] 79%|███████▉  | 317612/400000 [00:43<00:11, 7014.63it/s] 80%|███████▉  | 318380/400000 [00:43<00:11, 7201.72it/s] 80%|███████▉  | 319104/400000 [00:43<00:11, 7206.04it/s] 80%|███████▉  | 319828/400000 [00:43<00:11, 7126.22it/s] 80%|████████  | 320608/400000 [00:43<00:10, 7314.18it/s] 80%|████████  | 321391/400000 [00:43<00:10, 7459.69it/s] 81%|████████  | 322149/400000 [00:43<00:10, 7494.38it/s] 81%|████████  | 322914/400000 [00:43<00:10, 7539.43it/s] 81%|████████  | 323670/400000 [00:43<00:10, 7381.13it/s] 81%|████████  | 324433/400000 [00:44<00:10, 7452.25it/s] 81%|████████▏ | 325180/400000 [00:44<00:10, 7452.67it/s] 81%|████████▏ | 325957/400000 [00:44<00:09, 7542.51it/s] 82%|████████▏ | 326713/400000 [00:44<00:09, 7515.84it/s] 82%|████████▏ | 327466/400000 [00:44<00:09, 7445.84it/s] 82%|████████▏ | 328232/400000 [00:44<00:09, 7506.09it/s] 82%|████████▏ | 328984/400000 [00:44<00:09, 7482.45it/s] 82%|████████▏ | 329749/400000 [00:44<00:09, 7530.55it/s] 83%|████████▎ | 330512/400000 [00:44<00:09, 7559.59it/s] 83%|████████▎ | 331269/400000 [00:44<00:09, 7412.79it/s] 83%|████████▎ | 332021/400000 [00:45<00:09, 7442.65it/s] 83%|████████▎ | 332782/400000 [00:45<00:08, 7491.50it/s] 83%|████████▎ | 333533/400000 [00:45<00:08, 7495.77it/s] 84%|████████▎ | 334283/400000 [00:45<00:08, 7488.65it/s] 84%|████████▍ | 335033/400000 [00:45<00:08, 7396.61it/s] 84%|████████▍ | 335817/400000 [00:45<00:08, 7523.46it/s] 84%|████████▍ | 336576/400000 [00:45<00:08, 7541.21it/s] 84%|████████▍ | 337346/400000 [00:45<00:08, 7586.07it/s] 85%|████████▍ | 338106/400000 [00:45<00:08, 7537.26it/s] 85%|████████▍ | 338871/400000 [00:45<00:08, 7569.44it/s] 85%|████████▍ | 339637/400000 [00:46<00:07, 7594.79it/s] 85%|████████▌ | 340397/400000 [00:46<00:07, 7560.01it/s] 85%|████████▌ | 341154/400000 [00:46<00:07, 7444.93it/s] 85%|████████▌ | 341900/400000 [00:46<00:07, 7404.51it/s] 86%|████████▌ | 342641/400000 [00:46<00:07, 7201.37it/s] 86%|████████▌ | 343385/400000 [00:46<00:07, 7270.87it/s] 86%|████████▌ | 344134/400000 [00:46<00:07, 7333.67it/s] 86%|████████▌ | 344871/400000 [00:46<00:07, 7344.21it/s] 86%|████████▋ | 345639/400000 [00:46<00:07, 7441.50it/s] 87%|████████▋ | 346384/400000 [00:46<00:07, 7389.58it/s] 87%|████████▋ | 347124/400000 [00:47<00:07, 7340.45it/s] 87%|████████▋ | 347863/400000 [00:47<00:07, 7353.72it/s] 87%|████████▋ | 348610/400000 [00:47<00:06, 7387.63it/s] 87%|████████▋ | 349369/400000 [00:47<00:06, 7445.03it/s] 88%|████████▊ | 350116/400000 [00:47<00:06, 7451.04it/s] 88%|████████▊ | 350897/400000 [00:47<00:06, 7553.70it/s] 88%|████████▊ | 351661/400000 [00:47<00:06, 7578.52it/s] 88%|████████▊ | 352420/400000 [00:47<00:06, 7539.64it/s] 88%|████████▊ | 353175/400000 [00:47<00:06, 7456.65it/s] 88%|████████▊ | 353930/400000 [00:48<00:06, 7482.30it/s] 89%|████████▊ | 354679/400000 [00:48<00:06, 7456.96it/s] 89%|████████▉ | 355434/400000 [00:48<00:05, 7483.86it/s] 89%|████████▉ | 356197/400000 [00:48<00:05, 7526.80it/s] 89%|████████▉ | 356950/400000 [00:48<00:05, 7342.93it/s] 89%|████████▉ | 357686/400000 [00:48<00:05, 7333.98it/s] 90%|████████▉ | 358458/400000 [00:48<00:05, 7443.92it/s] 90%|████████▉ | 359210/400000 [00:48<00:05, 7466.52it/s] 90%|████████▉ | 359965/400000 [00:48<00:05, 7489.99it/s] 90%|█████████ | 360720/400000 [00:48<00:05, 7507.00it/s] 90%|█████████ | 361489/400000 [00:49<00:05, 7559.44it/s] 91%|█████████ | 362246/400000 [00:49<00:05, 7322.98it/s] 91%|█████████ | 363023/400000 [00:49<00:04, 7448.83it/s] 91%|█████████ | 363795/400000 [00:49<00:04, 7524.87it/s] 91%|█████████ | 364570/400000 [00:49<00:04, 7589.86it/s] 91%|█████████▏| 365331/400000 [00:49<00:04, 7528.69it/s] 92%|█████████▏| 366085/400000 [00:49<00:04, 7525.45it/s] 92%|█████████▏| 366839/400000 [00:49<00:04, 7509.13it/s] 92%|█████████▏| 367591/400000 [00:49<00:04, 7497.15it/s] 92%|█████████▏| 368342/400000 [00:49<00:04, 7488.04it/s] 92%|█████████▏| 369092/400000 [00:50<00:04, 7381.37it/s] 92%|█████████▏| 369832/400000 [00:50<00:04, 7384.09it/s] 93%|█████████▎| 370576/400000 [00:50<00:03, 7398.57it/s] 93%|█████████▎| 371317/400000 [00:50<00:03, 7386.81it/s] 93%|█████████▎| 372056/400000 [00:50<00:03, 7359.49it/s] 93%|█████████▎| 372796/400000 [00:50<00:03, 7369.63it/s] 93%|█████████▎| 373534/400000 [00:50<00:03, 7223.54it/s] 94%|█████████▎| 374303/400000 [00:50<00:03, 7356.92it/s] 94%|█████████▍| 375040/400000 [00:50<00:03, 7359.30it/s] 94%|█████████▍| 375793/400000 [00:50<00:03, 7407.38it/s] 94%|█████████▍| 376543/400000 [00:51<00:03, 7433.46it/s] 94%|█████████▍| 377287/400000 [00:51<00:03, 7324.55it/s] 95%|█████████▍| 378055/400000 [00:51<00:02, 7426.30it/s] 95%|█████████▍| 378823/400000 [00:51<00:02, 7500.22it/s] 95%|█████████▍| 379574/400000 [00:51<00:02, 7484.90it/s] 95%|█████████▌| 380324/400000 [00:51<00:02, 7152.71it/s] 95%|█████████▌| 381043/400000 [00:51<00:02, 7134.21it/s] 95%|█████████▌| 381783/400000 [00:51<00:02, 7209.20it/s] 96%|█████████▌| 382523/400000 [00:51<00:02, 7264.09it/s] 96%|█████████▌| 383251/400000 [00:51<00:02, 7247.10it/s] 96%|█████████▌| 383977/400000 [00:52<00:02, 7195.06it/s] 96%|█████████▌| 384698/400000 [00:52<00:02, 7188.99it/s] 96%|█████████▋| 385444/400000 [00:52<00:02, 7267.68it/s] 97%|█████████▋| 386190/400000 [00:52<00:01, 7322.31it/s] 97%|█████████▋| 386923/400000 [00:52<00:01, 7275.45it/s] 97%|█████████▋| 387678/400000 [00:52<00:01, 7354.79it/s] 97%|█████████▋| 388416/400000 [00:52<00:01, 7359.22it/s] 97%|█████████▋| 389186/400000 [00:52<00:01, 7458.17it/s] 97%|█████████▋| 389938/400000 [00:52<00:01, 7474.67it/s] 98%|█████████▊| 390686/400000 [00:52<00:01, 7117.86it/s] 98%|█████████▊| 391402/400000 [00:53<00:01, 7027.21it/s] 98%|█████████▊| 392135/400000 [00:53<00:01, 7114.71it/s] 98%|█████████▊| 392896/400000 [00:53<00:00, 7255.46it/s] 98%|█████████▊| 393650/400000 [00:53<00:00, 7336.76it/s] 99%|█████████▊| 394392/400000 [00:53<00:00, 7360.28it/s] 99%|█████████▉| 395130/400000 [00:53<00:00, 7345.54it/s] 99%|█████████▉| 395866/400000 [00:53<00:00, 7272.22it/s] 99%|█████████▉| 396604/400000 [00:53<00:00, 7303.82it/s] 99%|█████████▉| 397366/400000 [00:53<00:00, 7394.42it/s]100%|█████████▉| 398128/400000 [00:54<00:00, 7459.36it/s]100%|█████████▉| 398875/400000 [00:54<00:00, 7426.97it/s]100%|█████████▉| 399619/400000 [00:54<00:00, 7345.01it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7371.62it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f18a64e7f60> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011382216377149153 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011558143590206287 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15970 out of table with 15771 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15970 out of table with 15771 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

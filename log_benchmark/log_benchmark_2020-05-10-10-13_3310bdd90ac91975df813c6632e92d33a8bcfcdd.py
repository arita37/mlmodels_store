
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff5a213a4a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 10:13:19.083577
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 10:13:19.087164
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 10:13:19.090163
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 10:13:19.093198
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff5a2437320> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355125.4688
Epoch 2/10

1/1 [==============================] - 0s 94ms/step - loss: 264001.8438
Epoch 3/10

1/1 [==============================] - 0s 89ms/step - loss: 146030.8750
Epoch 4/10

1/1 [==============================] - 0s 87ms/step - loss: 74892.5938
Epoch 5/10

1/1 [==============================] - 0s 87ms/step - loss: 38251.3711
Epoch 6/10

1/1 [==============================] - 0s 92ms/step - loss: 21368.9375
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 13309.9043
Epoch 8/10

1/1 [==============================] - 0s 89ms/step - loss: 9071.9316
Epoch 9/10

1/1 [==============================] - 0s 92ms/step - loss: 6653.7983
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 5176.1294

  #### Inference Need return ypred, ytrue ######################### 
[[ -0.7532088    1.2702496   -0.5925925    0.8419831    1.1530873
   -0.9208156    0.04506406  -1.0936428   -0.08744127   0.51802385
    0.26269004  -0.646959    -0.876402    -1.1788844   -0.547549
    0.12562406   1.1302501    0.6785129    0.7315824   -1.0263115
    0.9330977    0.4615598    1.6160606    1.5031524   -0.3720214
   -0.07342689   1.3615879   -1.62029      0.42182052   0.5123962
    1.8880163    0.9264741    1.1982751    1.4360485    0.22334772
    0.06201622  -0.39218128   0.45885888  -0.08821741   0.9744365
   -0.22248828  -1.6587877   -0.10267636  -0.30810204   0.08765735
   -0.42041042   2.0283487   -0.9763727   -1.4811805    1.0739261
    0.21481079   0.28107676  -1.5484078   -0.05264445  -0.13725111
    1.5799809   -0.8387991    0.64012724  -0.23168677  -1.2988259
    0.04640722  -0.7253504   -1.0657011    1.6165806    0.01921862
   -0.7930011    0.8217572   -0.6253493    0.9208888   -1.3312378
   -0.20166042  -1.1670012    1.4048481    0.7260162    0.40230274
    0.67127955  -0.60544336  -1.5784957    1.7275143    0.21456152
   -0.70690376  -0.03258231  -0.58210915   0.8485422   -0.8755189
    0.31048205   0.04846045   0.4497583    0.23408365  -1.5937716
   -0.57382715   0.77099454  -0.8142406   -0.9945984    0.7868117
   -0.04132599  -1.0486382   -0.43832207  -0.23983212   0.57171386
    0.15599516  -0.5937922   -0.45402718   1.1043332    0.05460835
    1.9828714   -0.45893365  -1.1574426   -0.05526897  -2.2609727
   -1.2637086    0.54795945   0.41538545   0.3379856    0.6388411
    0.47723272  -1.0542653    1.1563734    0.8465338    0.06135003
    0.5794443    9.3975      11.051474     8.200218     8.932567
    9.303111    10.560082     8.598918     9.375996    10.457102
    9.80744      9.867888     7.799864    11.001586    10.099236
    8.778247    10.023511     8.665461     9.876064    10.148435
   10.17523      9.531484     8.863884     9.982029     9.728397
    8.3829975    8.713916     9.732156     9.957317    10.731283
    9.696112     8.617029     8.345047    11.527131     8.1992445
   10.44376      8.965637     9.136121     9.434326    10.410183
    8.204214     9.508168     9.61143      8.514182     7.177021
    9.158005     7.934322    10.161823     9.727988    10.858252
    9.514406    10.125506     9.315081     8.0004225    9.582309
    9.357429     8.193561     8.2453      10.34712      9.663325
    1.6672907    2.0060015    0.19831568   0.60509574   1.1967585
    1.4715216    1.9194543    1.9456776    0.29409826   1.5088996
    1.505169     1.3368893    0.85792816   2.0973976    0.45375496
    0.21537      1.4589901    0.5294616    0.9494147    1.9791574
    1.2793092    0.22709477   2.0199785    1.4901986    1.9262128
    0.40961885   0.53209317   0.9117486    0.43656963   1.2791498
    1.3704879    0.8388146    0.56570786   0.7000228    0.8968703
    0.3700235    0.6968267    0.7894711    0.49200404   1.0282146
    1.2356836    3.0966945    0.91690314   1.3901603    1.6650243
    0.6809343    1.9264623    0.38712907   1.1654085    0.66980857
    2.0414438    1.2487376    0.41049594   0.621911     1.3977798
    0.364097     1.2325681    0.49496502   0.57141876   0.47567558
    1.7482524    1.9840761    0.21333551   1.1589427    1.0904868
    0.08463126   3.0690808    1.48639      1.8510405    1.2155404
    0.33149332   0.1194948    1.2786931    0.82836366   1.797472
    1.9824556    0.26721215   0.08414191   1.5421317    1.1244158
    1.3065124    0.7037425    2.049747     2.1233215    2.0311606
    0.41212142   0.4210065    1.3545983    0.45085227   1.1638396
    2.065311     0.7661166    1.3541362    2.2151585    2.7672176
    1.1322293    0.37482244   1.1201668    0.3981955    0.5011385
    0.96902287   0.2714007    0.41534173   1.627648     0.25761712
    1.3340858    0.5024052    0.28545916   1.7831726    2.3037243
    0.7124486    1.4962239    1.7237056    0.94307834   0.72440875
    1.0198597    1.7431941    0.9849831    0.3564366    1.1287855
    0.4207877   10.411183     9.912256    10.757991     9.092935
    8.421817    10.21306      9.504492     8.300014     8.789802
    9.762225     9.11208      7.9331193    9.487437    10.467289
    9.516792    10.34741      8.893511     9.438038     8.154181
    9.509639     9.611592    10.0382       9.602507     9.198237
    8.290309    10.425736     9.698551     9.877654    11.008786
    9.344168     9.642217     8.856043     8.216626     8.347737
    9.338279    10.489923    11.620877     9.225446     9.629805
    9.992009    10.503385     8.453024     9.465356    11.504832
    7.667841     9.049294    10.693471     8.897616     9.846231
   10.319383     9.517311    10.767763     8.770299     8.232446
    9.644497     9.607132     9.517769     9.388595    10.811463
  -11.028321    -9.816928     8.2582855 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 10:13:27.876513
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    93.272
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 10:13:27.880285
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8724.99
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 10:13:27.883407
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7304
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 10:13:27.886466
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -780.384
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140692273960552
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140691332400016
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140691332400520
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140691332401024
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140691332401528
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140691332402032

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff58c8f5e80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.443033
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.406399
grad_step = 000002, loss = 0.375167
grad_step = 000003, loss = 0.341914
grad_step = 000004, loss = 0.306836
grad_step = 000005, loss = 0.276313
grad_step = 000006, loss = 0.260200
grad_step = 000007, loss = 0.253149
grad_step = 000008, loss = 0.238056
grad_step = 000009, loss = 0.217376
grad_step = 000010, loss = 0.201999
grad_step = 000011, loss = 0.193866
grad_step = 000012, loss = 0.188277
grad_step = 000013, loss = 0.180569
grad_step = 000014, loss = 0.171162
grad_step = 000015, loss = 0.161008
grad_step = 000016, loss = 0.152029
grad_step = 000017, loss = 0.145218
grad_step = 000018, loss = 0.139347
grad_step = 000019, loss = 0.132470
grad_step = 000020, loss = 0.124454
grad_step = 000021, loss = 0.116935
grad_step = 000022, loss = 0.110875
grad_step = 000023, loss = 0.105575
grad_step = 000024, loss = 0.100021
grad_step = 000025, loss = 0.094070
grad_step = 000026, loss = 0.088283
grad_step = 000027, loss = 0.083151
grad_step = 000028, loss = 0.078630
grad_step = 000029, loss = 0.074283
grad_step = 000030, loss = 0.069803
grad_step = 000031, loss = 0.065422
grad_step = 000032, loss = 0.061487
grad_step = 000033, loss = 0.057954
grad_step = 000034, loss = 0.054556
grad_step = 000035, loss = 0.051169
grad_step = 000036, loss = 0.047898
grad_step = 000037, loss = 0.044900
grad_step = 000038, loss = 0.042174
grad_step = 000039, loss = 0.039547
grad_step = 000040, loss = 0.036888
grad_step = 000041, loss = 0.034327
grad_step = 000042, loss = 0.032041
grad_step = 000043, loss = 0.029991
grad_step = 000044, loss = 0.028015
grad_step = 000045, loss = 0.026037
grad_step = 000046, loss = 0.024140
grad_step = 000047, loss = 0.022451
grad_step = 000048, loss = 0.020930
grad_step = 000049, loss = 0.019433
grad_step = 000050, loss = 0.017944
grad_step = 000051, loss = 0.016563
grad_step = 000052, loss = 0.015336
grad_step = 000053, loss = 0.014205
grad_step = 000054, loss = 0.013101
grad_step = 000055, loss = 0.012056
grad_step = 000056, loss = 0.011125
grad_step = 000057, loss = 0.010287
grad_step = 000058, loss = 0.009496
grad_step = 000059, loss = 0.008744
grad_step = 000060, loss = 0.008061
grad_step = 000061, loss = 0.007454
grad_step = 000062, loss = 0.006896
grad_step = 000063, loss = 0.006379
grad_step = 000064, loss = 0.005910
grad_step = 000065, loss = 0.005494
grad_step = 000066, loss = 0.005119
grad_step = 000067, loss = 0.004775
grad_step = 000068, loss = 0.004464
grad_step = 000069, loss = 0.004192
grad_step = 000070, loss = 0.003945
grad_step = 000071, loss = 0.003721
grad_step = 000072, loss = 0.003525
grad_step = 000073, loss = 0.003352
grad_step = 000074, loss = 0.003194
grad_step = 000075, loss = 0.003053
grad_step = 000076, loss = 0.002932
grad_step = 000077, loss = 0.002825
grad_step = 000078, loss = 0.002727
grad_step = 000079, loss = 0.002644
grad_step = 000080, loss = 0.002574
grad_step = 000081, loss = 0.002514
grad_step = 000082, loss = 0.002460
grad_step = 000083, loss = 0.002415
grad_step = 000084, loss = 0.002379
grad_step = 000085, loss = 0.002347
grad_step = 000086, loss = 0.002319
grad_step = 000087, loss = 0.002298
grad_step = 000088, loss = 0.002282
grad_step = 000089, loss = 0.002268
grad_step = 000090, loss = 0.002256
grad_step = 000091, loss = 0.002248
grad_step = 000092, loss = 0.002242
grad_step = 000093, loss = 0.002236
grad_step = 000094, loss = 0.002232
grad_step = 000095, loss = 0.002229
grad_step = 000096, loss = 0.002226
grad_step = 000097, loss = 0.002223
grad_step = 000098, loss = 0.002221
grad_step = 000099, loss = 0.002219
grad_step = 000100, loss = 0.002216
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002214
grad_step = 000102, loss = 0.002211
grad_step = 000103, loss = 0.002208
grad_step = 000104, loss = 0.002205
grad_step = 000105, loss = 0.002201
grad_step = 000106, loss = 0.002197
grad_step = 000107, loss = 0.002193
grad_step = 000108, loss = 0.002189
grad_step = 000109, loss = 0.002185
grad_step = 000110, loss = 0.002180
grad_step = 000111, loss = 0.002175
grad_step = 000112, loss = 0.002171
grad_step = 000113, loss = 0.002166
grad_step = 000114, loss = 0.002161
grad_step = 000115, loss = 0.002157
grad_step = 000116, loss = 0.002152
grad_step = 000117, loss = 0.002148
grad_step = 000118, loss = 0.002143
grad_step = 000119, loss = 0.002139
grad_step = 000120, loss = 0.002135
grad_step = 000121, loss = 0.002131
grad_step = 000122, loss = 0.002127
grad_step = 000123, loss = 0.002122
grad_step = 000124, loss = 0.002119
grad_step = 000125, loss = 0.002115
grad_step = 000126, loss = 0.002111
grad_step = 000127, loss = 0.002107
grad_step = 000128, loss = 0.002103
grad_step = 000129, loss = 0.002100
grad_step = 000130, loss = 0.002096
grad_step = 000131, loss = 0.002092
grad_step = 000132, loss = 0.002089
grad_step = 000133, loss = 0.002085
grad_step = 000134, loss = 0.002081
grad_step = 000135, loss = 0.002078
grad_step = 000136, loss = 0.002074
grad_step = 000137, loss = 0.002070
grad_step = 000138, loss = 0.002066
grad_step = 000139, loss = 0.002062
grad_step = 000140, loss = 0.002058
grad_step = 000141, loss = 0.002054
grad_step = 000142, loss = 0.002050
grad_step = 000143, loss = 0.002045
grad_step = 000144, loss = 0.002041
grad_step = 000145, loss = 0.002036
grad_step = 000146, loss = 0.002032
grad_step = 000147, loss = 0.002027
grad_step = 000148, loss = 0.002022
grad_step = 000149, loss = 0.002017
grad_step = 000150, loss = 0.002012
grad_step = 000151, loss = 0.002007
grad_step = 000152, loss = 0.002002
grad_step = 000153, loss = 0.001996
grad_step = 000154, loss = 0.001990
grad_step = 000155, loss = 0.001985
grad_step = 000156, loss = 0.001979
grad_step = 000157, loss = 0.001973
grad_step = 000158, loss = 0.001968
grad_step = 000159, loss = 0.001964
grad_step = 000160, loss = 0.001965
grad_step = 000161, loss = 0.001975
grad_step = 000162, loss = 0.001985
grad_step = 000163, loss = 0.001984
grad_step = 000164, loss = 0.001951
grad_step = 000165, loss = 0.001921
grad_step = 000166, loss = 0.001915
grad_step = 000167, loss = 0.001927
grad_step = 000168, loss = 0.001934
grad_step = 000169, loss = 0.001915
grad_step = 000170, loss = 0.001889
grad_step = 000171, loss = 0.001874
grad_step = 000172, loss = 0.001875
grad_step = 000173, loss = 0.001881
grad_step = 000174, loss = 0.001877
grad_step = 000175, loss = 0.001863
grad_step = 000176, loss = 0.001841
grad_step = 000177, loss = 0.001824
grad_step = 000178, loss = 0.001816
grad_step = 000179, loss = 0.001813
grad_step = 000180, loss = 0.001814
grad_step = 000181, loss = 0.001818
grad_step = 000182, loss = 0.001828
grad_step = 000183, loss = 0.001827
grad_step = 000184, loss = 0.001823
grad_step = 000185, loss = 0.001790
grad_step = 000186, loss = 0.001755
grad_step = 000187, loss = 0.001732
grad_step = 000188, loss = 0.001722
grad_step = 000189, loss = 0.001723
grad_step = 000190, loss = 0.001740
grad_step = 000191, loss = 0.001797
grad_step = 000192, loss = 0.001823
grad_step = 000193, loss = 0.001841
grad_step = 000194, loss = 0.001737
grad_step = 000195, loss = 0.001663
grad_step = 000196, loss = 0.001674
grad_step = 000197, loss = 0.001707
grad_step = 000198, loss = 0.001733
grad_step = 000199, loss = 0.001716
grad_step = 000200, loss = 0.001664
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001617
grad_step = 000202, loss = 0.001612
grad_step = 000203, loss = 0.001623
grad_step = 000204, loss = 0.001656
grad_step = 000205, loss = 0.001702
grad_step = 000206, loss = 0.001700
grad_step = 000207, loss = 0.001715
grad_step = 000208, loss = 0.001636
grad_step = 000209, loss = 0.001579
grad_step = 000210, loss = 0.001555
grad_step = 000211, loss = 0.001571
grad_step = 000212, loss = 0.001620
grad_step = 000213, loss = 0.001650
grad_step = 000214, loss = 0.001700
grad_step = 000215, loss = 0.001666
grad_step = 000216, loss = 0.001622
grad_step = 000217, loss = 0.001547
grad_step = 000218, loss = 0.001514
grad_step = 000219, loss = 0.001537
grad_step = 000220, loss = 0.001577
grad_step = 000221, loss = 0.001614
grad_step = 000222, loss = 0.001588
grad_step = 000223, loss = 0.001556
grad_step = 000224, loss = 0.001507
grad_step = 000225, loss = 0.001485
grad_step = 000226, loss = 0.001486
grad_step = 000227, loss = 0.001509
grad_step = 000228, loss = 0.001552
grad_step = 000229, loss = 0.001584
grad_step = 000230, loss = 0.001650
grad_step = 000231, loss = 0.001604
grad_step = 000232, loss = 0.001554
grad_step = 000233, loss = 0.001476
grad_step = 000234, loss = 0.001468
grad_step = 000235, loss = 0.001517
grad_step = 000236, loss = 0.001537
grad_step = 000237, loss = 0.001535
grad_step = 000238, loss = 0.001474
grad_step = 000239, loss = 0.001447
grad_step = 000240, loss = 0.001463
grad_step = 000241, loss = 0.001494
grad_step = 000242, loss = 0.001532
grad_step = 000243, loss = 0.001513
grad_step = 000244, loss = 0.001492
grad_step = 000245, loss = 0.001453
grad_step = 000246, loss = 0.001433
grad_step = 000247, loss = 0.001435
grad_step = 000248, loss = 0.001449
grad_step = 000249, loss = 0.001464
grad_step = 000250, loss = 0.001459
grad_step = 000251, loss = 0.001452
grad_step = 000252, loss = 0.001434
grad_step = 000253, loss = 0.001421
grad_step = 000254, loss = 0.001417
grad_step = 000255, loss = 0.001422
grad_step = 000256, loss = 0.001430
grad_step = 000257, loss = 0.001432
grad_step = 000258, loss = 0.001434
grad_step = 000259, loss = 0.001427
grad_step = 000260, loss = 0.001422
grad_step = 000261, loss = 0.001412
grad_step = 000262, loss = 0.001405
grad_step = 000263, loss = 0.001400
grad_step = 000264, loss = 0.001397
grad_step = 000265, loss = 0.001396
grad_step = 000266, loss = 0.001395
grad_step = 000267, loss = 0.001395
grad_step = 000268, loss = 0.001397
grad_step = 000269, loss = 0.001403
grad_step = 000270, loss = 0.001412
grad_step = 000271, loss = 0.001437
grad_step = 000272, loss = 0.001461
grad_step = 000273, loss = 0.001524
grad_step = 000274, loss = 0.001516
grad_step = 000275, loss = 0.001522
grad_step = 000276, loss = 0.001434
grad_step = 000277, loss = 0.001382
grad_step = 000278, loss = 0.001389
grad_step = 000279, loss = 0.001429
grad_step = 000280, loss = 0.001475
grad_step = 000281, loss = 0.001440
grad_step = 000282, loss = 0.001406
grad_step = 000283, loss = 0.001370
grad_step = 000284, loss = 0.001365
grad_step = 000285, loss = 0.001382
grad_step = 000286, loss = 0.001402
grad_step = 000287, loss = 0.001426
grad_step = 000288, loss = 0.001408
grad_step = 000289, loss = 0.001392
grad_step = 000290, loss = 0.001364
grad_step = 000291, loss = 0.001351
grad_step = 000292, loss = 0.001353
grad_step = 000293, loss = 0.001363
grad_step = 000294, loss = 0.001374
grad_step = 000295, loss = 0.001372
grad_step = 000296, loss = 0.001368
grad_step = 000297, loss = 0.001353
grad_step = 000298, loss = 0.001343
grad_step = 000299, loss = 0.001336
grad_step = 000300, loss = 0.001335
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001337
grad_step = 000302, loss = 0.001341
grad_step = 000303, loss = 0.001349
grad_step = 000304, loss = 0.001354
grad_step = 000305, loss = 0.001367
grad_step = 000306, loss = 0.001371
grad_step = 000307, loss = 0.001388
grad_step = 000308, loss = 0.001379
grad_step = 000309, loss = 0.001381
grad_step = 000310, loss = 0.001354
grad_step = 000311, loss = 0.001335
grad_step = 000312, loss = 0.001318
grad_step = 000313, loss = 0.001312
grad_step = 000314, loss = 0.001317
grad_step = 000315, loss = 0.001326
grad_step = 000316, loss = 0.001339
grad_step = 000317, loss = 0.001342
grad_step = 000318, loss = 0.001353
grad_step = 000319, loss = 0.001347
grad_step = 000320, loss = 0.001352
grad_step = 000321, loss = 0.001339
grad_step = 000322, loss = 0.001336
grad_step = 000323, loss = 0.001320
grad_step = 000324, loss = 0.001309
grad_step = 000325, loss = 0.001297
grad_step = 000326, loss = 0.001290
grad_step = 000327, loss = 0.001286
grad_step = 000328, loss = 0.001285
grad_step = 000329, loss = 0.001286
grad_step = 000330, loss = 0.001289
grad_step = 000331, loss = 0.001296
grad_step = 000332, loss = 0.001307
grad_step = 000333, loss = 0.001335
grad_step = 000334, loss = 0.001360
grad_step = 000335, loss = 0.001431
grad_step = 000336, loss = 0.001434
grad_step = 000337, loss = 0.001467
grad_step = 000338, loss = 0.001367
grad_step = 000339, loss = 0.001295
grad_step = 000340, loss = 0.001269
grad_step = 000341, loss = 0.001302
grad_step = 000342, loss = 0.001349
grad_step = 000343, loss = 0.001328
grad_step = 000344, loss = 0.001288
grad_step = 000345, loss = 0.001258
grad_step = 000346, loss = 0.001262
grad_step = 000347, loss = 0.001290
grad_step = 000348, loss = 0.001310
grad_step = 000349, loss = 0.001329
grad_step = 000350, loss = 0.001302
grad_step = 000351, loss = 0.001275
grad_step = 000352, loss = 0.001249
grad_step = 000353, loss = 0.001241
grad_step = 000354, loss = 0.001250
grad_step = 000355, loss = 0.001262
grad_step = 000356, loss = 0.001270
grad_step = 000357, loss = 0.001262
grad_step = 000358, loss = 0.001250
grad_step = 000359, loss = 0.001234
grad_step = 000360, loss = 0.001226
grad_step = 000361, loss = 0.001226
grad_step = 000362, loss = 0.001231
grad_step = 000363, loss = 0.001238
grad_step = 000364, loss = 0.001241
grad_step = 000365, loss = 0.001246
grad_step = 000366, loss = 0.001240
grad_step = 000367, loss = 0.001236
grad_step = 000368, loss = 0.001227
grad_step = 000369, loss = 0.001219
grad_step = 000370, loss = 0.001210
grad_step = 000371, loss = 0.001203
grad_step = 000372, loss = 0.001198
grad_step = 000373, loss = 0.001194
grad_step = 000374, loss = 0.001190
grad_step = 000375, loss = 0.001187
grad_step = 000376, loss = 0.001185
grad_step = 000377, loss = 0.001182
grad_step = 000378, loss = 0.001181
grad_step = 000379, loss = 0.001180
grad_step = 000380, loss = 0.001185
grad_step = 000381, loss = 0.001201
grad_step = 000382, loss = 0.001256
grad_step = 000383, loss = 0.001364
grad_step = 000384, loss = 0.001681
grad_step = 000385, loss = 0.001836
grad_step = 000386, loss = 0.002000
grad_step = 000387, loss = 0.001338
grad_step = 000388, loss = 0.001239
grad_step = 000389, loss = 0.001595
grad_step = 000390, loss = 0.001362
grad_step = 000391, loss = 0.001201
grad_step = 000392, loss = 0.001390
grad_step = 000393, loss = 0.001296
grad_step = 000394, loss = 0.001173
grad_step = 000395, loss = 0.001275
grad_step = 000396, loss = 0.001258
grad_step = 000397, loss = 0.001163
grad_step = 000398, loss = 0.001185
grad_step = 000399, loss = 0.001232
grad_step = 000400, loss = 0.001225
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001166
grad_step = 000402, loss = 0.001150
grad_step = 000403, loss = 0.001199
grad_step = 000404, loss = 0.001200
grad_step = 000405, loss = 0.001157
grad_step = 000406, loss = 0.001135
grad_step = 000407, loss = 0.001149
grad_step = 000408, loss = 0.001172
grad_step = 000409, loss = 0.001168
grad_step = 000410, loss = 0.001137
grad_step = 000411, loss = 0.001120
grad_step = 000412, loss = 0.001130
grad_step = 000413, loss = 0.001142
grad_step = 000414, loss = 0.001140
grad_step = 000415, loss = 0.001128
grad_step = 000416, loss = 0.001110
grad_step = 000417, loss = 0.001111
grad_step = 000418, loss = 0.001118
grad_step = 000419, loss = 0.001120
grad_step = 000420, loss = 0.001119
grad_step = 000421, loss = 0.001107
grad_step = 000422, loss = 0.001099
grad_step = 000423, loss = 0.001098
grad_step = 000424, loss = 0.001099
grad_step = 000425, loss = 0.001103
grad_step = 000426, loss = 0.001101
grad_step = 000427, loss = 0.001096
grad_step = 000428, loss = 0.001090
grad_step = 000429, loss = 0.001086
grad_step = 000430, loss = 0.001084
grad_step = 000431, loss = 0.001086
grad_step = 000432, loss = 0.001087
grad_step = 000433, loss = 0.001086
grad_step = 000434, loss = 0.001085
grad_step = 000435, loss = 0.001081
grad_step = 000436, loss = 0.001077
grad_step = 000437, loss = 0.001073
grad_step = 000438, loss = 0.001070
grad_step = 000439, loss = 0.001069
grad_step = 000440, loss = 0.001068
grad_step = 000441, loss = 0.001066
grad_step = 000442, loss = 0.001066
grad_step = 000443, loss = 0.001065
grad_step = 000444, loss = 0.001064
grad_step = 000445, loss = 0.001063
grad_step = 000446, loss = 0.001062
grad_step = 000447, loss = 0.001061
grad_step = 000448, loss = 0.001060
grad_step = 000449, loss = 0.001059
grad_step = 000450, loss = 0.001058
grad_step = 000451, loss = 0.001058
grad_step = 000452, loss = 0.001057
grad_step = 000453, loss = 0.001056
grad_step = 000454, loss = 0.001055
grad_step = 000455, loss = 0.001057
grad_step = 000456, loss = 0.001057
grad_step = 000457, loss = 0.001062
grad_step = 000458, loss = 0.001066
grad_step = 000459, loss = 0.001078
grad_step = 000460, loss = 0.001086
grad_step = 000461, loss = 0.001113
grad_step = 000462, loss = 0.001123
grad_step = 000463, loss = 0.001160
grad_step = 000464, loss = 0.001146
grad_step = 000465, loss = 0.001151
grad_step = 000466, loss = 0.001097
grad_step = 000467, loss = 0.001055
grad_step = 000468, loss = 0.001028
grad_step = 000469, loss = 0.001032
grad_step = 000470, loss = 0.001056
grad_step = 000471, loss = 0.001069
grad_step = 000472, loss = 0.001073
grad_step = 000473, loss = 0.001049
grad_step = 000474, loss = 0.001027
grad_step = 000475, loss = 0.001015
grad_step = 000476, loss = 0.001018
grad_step = 000477, loss = 0.001029
grad_step = 000478, loss = 0.001039
grad_step = 000479, loss = 0.001048
grad_step = 000480, loss = 0.001046
grad_step = 000481, loss = 0.001044
grad_step = 000482, loss = 0.001032
grad_step = 000483, loss = 0.001023
grad_step = 000484, loss = 0.001012
grad_step = 000485, loss = 0.001004
grad_step = 000486, loss = 0.000999
grad_step = 000487, loss = 0.000996
grad_step = 000488, loss = 0.000996
grad_step = 000489, loss = 0.000997
grad_step = 000490, loss = 0.000999
grad_step = 000491, loss = 0.001002
grad_step = 000492, loss = 0.001008
grad_step = 000493, loss = 0.001013
grad_step = 000494, loss = 0.001025
grad_step = 000495, loss = 0.001036
grad_step = 000496, loss = 0.001069
grad_step = 000497, loss = 0.001098
grad_step = 000498, loss = 0.001172
grad_step = 000499, loss = 0.001172
grad_step = 000500, loss = 0.001206
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001105
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

  date_run                              2020-05-10 10:13:45.279970
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.217583
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 10:13:45.285436
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.118532
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 10:13:45.292559
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.130694
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 10:13:45.297461
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.801143
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
0   2020-05-10 10:13:19.083577  ...    mean_absolute_error
1   2020-05-10 10:13:19.087164  ...     mean_squared_error
2   2020-05-10 10:13:19.090163  ...  median_absolute_error
3   2020-05-10 10:13:19.093198  ...               r2_score
4   2020-05-10 10:13:27.876513  ...    mean_absolute_error
5   2020-05-10 10:13:27.880285  ...     mean_squared_error
6   2020-05-10 10:13:27.883407  ...  median_absolute_error
7   2020-05-10 10:13:27.886466  ...               r2_score
8   2020-05-10 10:13:45.279970  ...    mean_absolute_error
9   2020-05-10 10:13:45.285436  ...     mean_squared_error
10  2020-05-10 10:13:45.292559  ...  median_absolute_error
11  2020-05-10 10:13:45.297461  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 140818.31it/s] 52%|█████▏    | 5144576/9912422 [00:00<00:23, 200889.21it/s]9920512it [00:00, 37816119.17it/s]                           
0it [00:00, ?it/s]32768it [00:00, 568773.32it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 467407.36it/s]1654784it [00:00, 11864063.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 182412.35it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe0c41f9780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe06193c9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe0c41f9e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe0c41b0e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe06193e080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe06b05f4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe06b064438> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe06b05f4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe076babcc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe06b05f4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe076babcc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa460b481d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7df4f74abfba10ef30791eee33217820597c86cc2c7a9b63de088c11f67c8b21
  Stored in directory: /tmp/pip-ephem-wheel-cache-ccyad5hg/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa3f872c080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  401408/17464789 [..............................] - ETA: 2s
  860160/17464789 [>.............................] - ETA: 2s
 1220608/17464789 [=>............................] - ETA: 2s
 1605632/17464789 [=>............................] - ETA: 2s
 2056192/17464789 [==>...........................] - ETA: 2s
 2580480/17464789 [===>..........................] - ETA: 1s
 3153920/17464789 [====>.........................] - ETA: 1s
 3784704/17464789 [=====>........................] - ETA: 1s
 4472832/17464789 [======>.......................] - ETA: 1s
 5177344/17464789 [=======>......................] - ETA: 1s
 5857280/17464789 [=========>....................] - ETA: 1s
 6512640/17464789 [==========>...................] - ETA: 1s
 7069696/17464789 [===========>..................] - ETA: 1s
 7716864/17464789 [============>.................] - ETA: 0s
 8273920/17464789 [=============>................] - ETA: 0s
 9035776/17464789 [==============>...............] - ETA: 0s
 9838592/17464789 [===============>..............] - ETA: 0s
10690560/17464789 [=================>............] - ETA: 0s
11567104/17464789 [==================>...........] - ETA: 0s
12435456/17464789 [====================>.........] - ETA: 0s
13426688/17464789 [======================>.......] - ETA: 0s
14524416/17464789 [=======================>......] - ETA: 0s
15622144/17464789 [=========================>....] - ETA: 0s
16433152/17464789 [===========================>..] - ETA: 0s
17309696/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 10:15:11.960910: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 10:15:11.965018: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-10 10:15:11.965650: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b986cf5bd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 10:15:11.965671: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9273 - accuracy: 0.4830 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8097 - accuracy: 0.4907
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7816 - accuracy: 0.4925
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7770 - accuracy: 0.4928
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7510 - accuracy: 0.4945
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7214 - accuracy: 0.4964
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7314 - accuracy: 0.4958
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7295 - accuracy: 0.4959
11000/25000 [============>.................] - ETA: 3s - loss: 7.7112 - accuracy: 0.4971
12000/25000 [=============>................] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6442 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6502 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 2s - loss: 7.6411 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6379 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6387 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6351 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6344 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6352 - accuracy: 0.5020
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 7s 267us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 10:15:25.110790
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 10:15:25.110790  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 10:15:30.929135: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 10:15:30.935496: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-10 10:15:30.935703: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560c73a457f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 10:15:30.935717: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd7c356dcc0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7777 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.7119 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd7e512e0b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4520 - accuracy: 0.5140
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4980 - accuracy: 0.5110 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5184 - accuracy: 0.5097
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4903 - accuracy: 0.5115
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5388 - accuracy: 0.5083
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6250 - accuracy: 0.5027
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6551 - accuracy: 0.5008
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6804 - accuracy: 0.4991
11000/25000 [============>.................] - ETA: 3s - loss: 7.6680 - accuracy: 0.4999
12000/25000 [=============>................] - ETA: 3s - loss: 7.6756 - accuracy: 0.4994
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6867 - accuracy: 0.4987
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6783 - accuracy: 0.4992
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6346 - accuracy: 0.5021
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6480 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 269us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd7857b1390> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:03:57, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:40:36, 15.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:01:37, 21.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:43:37, 31.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:23:42, 44.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.93M/862M [00:01<3:45:16, 63.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:37:11, 90.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.6M/862M [00:01<1:49:35, 129kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<1:16:24, 183kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:01<53:18, 262kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:01<37:12, 373kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:02<26:03, 530kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<18:11, 754kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:02<12:48, 1.07MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:02<08:59, 1.51MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<06:57, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:45, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<06:56, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:23, 2.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:03, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:40, 2.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<05:50, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<07:10, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<05:47, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<04:12, 3.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<12:48, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<10:26, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<07:39, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<08:13, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<08:26, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<06:27, 2.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:13<04:41, 2.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<09:23, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<07:56, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<05:50, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:07, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<06:20, 2.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:17<04:46, 2.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<06:23, 2.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:19<07:07, 1.81MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<05:34, 2.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:19<04:00, 3.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<1:37:29, 132kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<1:09:32, 185kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:21<48:52, 263kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<37:07, 345kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<28:34, 448kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<20:38, 620kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<16:27, 774kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<12:50, 991kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<09:15, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:24, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:09, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:57, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<05:00, 2.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:49, 983kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<10:16, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<07:30, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:09, 1.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:15, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:20, 1.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:34, 2.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<13:42, 909kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<10:52, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:32<07:54, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:26, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:25, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:27, 1.92MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:37, 2.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<21:22, 577kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<16:13, 759kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<11:39, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:00, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:58, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<06:32, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:25, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:40, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:53, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:16, 2.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:18, 1.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:27, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:13, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:09, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:27, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:43, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:16, 2.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:10, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:32, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:08, 2.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:39, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:11, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:53, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:30, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:03, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<03:50, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:27, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:00, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:45, 3.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:25, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:47, 2.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:38, 3.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:18, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:05, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:51, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:15, 2.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:53, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:43, 3.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:16, 2.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:52, 2.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:41, 3.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:17, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<04:50, 2.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:38, 3.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:14, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:58, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:45, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:09, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:45, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:33, 3.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:08, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:52, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:35, 2.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:19, 3.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<13:10, 847kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:21, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:27, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:49, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:36, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:53, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:01, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:19, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:57, 2.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:22, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:00, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:45, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:26, 3.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:29:50, 17.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<7:21:45, 24.6kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<5:08:47, 35.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<3:37:56, 49.6kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<2:33:22, 70.5kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<1:47:21, 100kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<1:15:02, 143kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<1:33:08, 115kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<1:06:14, 162kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<46:31, 230kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<35:00, 305kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<25:33, 418kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<18:04, 589kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<15:08, 701kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<12:45, 832kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<09:23, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<06:42, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<09:28, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:42, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<05:38, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:20, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:34, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:06, 2.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<03:40, 2.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<14:43, 706kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<11:23, 913kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<08:11, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:07, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:46, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:53, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<04:14, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:25, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:57, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<05:07, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:58, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:14, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:48, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:31, 2.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:00, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:14, 1.93MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:55, 2.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:05, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:36, 1.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:26, 2.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:42, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:18, 2.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:16, 3.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:36, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:14, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:07, 2.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<02:59, 3.29MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:45, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:08, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:14, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:57, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:09, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:48, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:27, 2.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<28:07, 345kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<20:40, 470kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<14:40, 660kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<12:30, 771kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:44, 898kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:00, 1.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<05:41, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<9:11:49, 17.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<6:27:00, 24.7kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<4:30:24, 35.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<3:08:45, 50.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:33:54, 13.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<8:06:11, 19.5kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<5:39:26, 27.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<3:59:25, 39.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<2:49:38, 55.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:59:03, 79.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<1:23:16, 113kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<1:00:55, 154kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<43:32, 215kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<30:38, 304kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<23:33, 394kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<17:25, 533kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<12:24, 746kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<10:49, 851kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:31, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<06:10, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:28, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:27, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:02, 2.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:58, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:24, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:16, 2.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:25, 2.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:59, 2.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:01, 2.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:13, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:46, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:47, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:44, 3.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<8:34:50, 17.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<6:01:02, 24.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<4:12:13, 35.1kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<2:57:52, 49.5kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<2:06:16, 69.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:28:42, 99.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<1:03:09, 138kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<45:03, 194kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<31:38, 275kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<24:06, 360kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<18:37, 465kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<13:28, 642kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<09:29, 907kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<1:16:47, 112kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<54:34, 158kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<38:18, 224kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<28:42, 297kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<20:57, 407kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<14:51, 573kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<12:19, 687kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<09:28, 892kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:49, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:45, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<06:25, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:53, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:30, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<1:11:02, 117kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<50:31, 165kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<35:28, 234kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<26:39, 310kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<20:19, 406kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<14:35, 565kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<10:14, 801kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<20:58, 391kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<15:30, 528kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<11:01, 740kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<09:35, 846kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<08:21, 971kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<06:12, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:25, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:09, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:50, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<04:15, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:50, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:00, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:50, 2.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<02:46, 2.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:02, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<04:10, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:45, 1.65MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:55, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:50, 2.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:56, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:33, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:40, 2.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:40, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:12, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:20, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:25, 3.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<56:20, 136kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<40:10, 190kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<28:13, 270kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<21:26, 353kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<16:32, 458kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<11:57, 632kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<08:24, 893kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<59:32, 126kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<42:25, 177kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<29:47, 251kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<22:28, 331kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<16:28, 451kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<11:39, 635kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<09:52, 746kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<07:39, 961kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<05:30, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:34, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:23, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:04, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:56, 2.47MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:54, 1.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:52, 1.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:35, 2.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:22, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:25, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:31, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:12, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:23, 2.96MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:19, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:01, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:16, 3.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:14, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:40, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:53, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:04, 3.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<10:55, 631kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<08:21, 824kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:58, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:45, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:22, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:02, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:39, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:39, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:25, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:57, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:27, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:34, 2.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:21, 1.97MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:54, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:11, 3.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:05, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:26, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:43, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:56, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:44, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:03, 3.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:55, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:21, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:37, 2.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<01:53, 3.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<10:27, 606kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<07:57, 796kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<05:40, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:26, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:04, 1.24MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<03:48, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:43, 2.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:30, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:28, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:43, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:51, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:57, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<02:07, 2.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<08:56, 678kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<06:53, 880kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<04:57, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:50, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:36, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:29, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<02:29, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:43, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:36, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:21, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:42, 1.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:50, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:56, 1.99MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:07, 2.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:17, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:36, 1.60MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:38, 2.17MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:11, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:23, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:37, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<01:53, 2.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:53, 960kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:41, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:23, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:40, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:42, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:52, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:53, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:34, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:55, 2.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:37, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:45, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:12, 2.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:37, 3.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:17, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:51, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:07, 2.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:42, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:26, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:50, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:31, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:17, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<01:43, 3.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:25, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:44, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:09, 2.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:33, 3.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<06:00, 849kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<04:44, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:24, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:33, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:30, 1.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:40, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:22, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:47, 2.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:23, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:40, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:05, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:30, 3.21MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<04:01, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:19, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:24, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:47, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:25, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:48, 2.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:22, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:37, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:01, 2.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:29, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:35, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:16, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:41, 2.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:14, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:02, 2.24MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:31, 2.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:07, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:56, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:28, 3.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:04, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:20, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:49, 2.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:19, 3.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:00, 1.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:32, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:52, 2.30MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:18, 1.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:29, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:55, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<01:22, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:40, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:00, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:11, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:29, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:09, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:36, 2.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:04, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:17, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:46, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:18, 3.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:09, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:54, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:26, 2.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:54, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:10, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:41, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:13, 3.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:12, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:38, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:14, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:22, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:49, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:19, 2.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:44, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:18, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:41, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:02, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:48, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:20, 2.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:46, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:12, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:40, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:31, 2.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:09, 3.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:37, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:50, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:27, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:33, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:26, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:04, 3.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:31, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:43, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:20, 2.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<00:58, 3.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:08, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:49, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:21, 2.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:41, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:30, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:07, 2.79MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:30, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:41, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:19, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<00:56, 3.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<05:16, 578kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<04:00, 761kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:51, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:40, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:10, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:34, 1.87MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:46, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:50, 1.58MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:00, 2.82MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<11:05, 257kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<08:02, 353kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<05:38, 499kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<04:32, 611kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:27, 802kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:27, 1.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:20, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:11, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:38, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:09, 2.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<10:31, 251kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<07:37, 345kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<05:20, 488kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:17, 599kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:15, 788kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:19, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:11, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:02, 1.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:32, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<01:04, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:27, 702kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:39, 910kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:54, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:52, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:47, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:21, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:58, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:24, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:13, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:53, 2.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:08, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:15, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:58, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:01, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:15, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:56, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:02, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:09, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:54, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:56, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:52, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:38, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:54, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:01, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:48, 2.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:51, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:47, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:35, 3.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:49, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:45, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:34, 3.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:48, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:44, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:33, 3.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:46, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:42, 2.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:31, 3.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:44, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:50, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:40, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:42, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:29, 3.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:40, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:37, 2.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:27, 3.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:34, 2.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:36, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:33, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:25, 3.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:35, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:39, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:31, 2.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:32, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:30, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:22, 3.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:35, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:27, 2.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:19, 3.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:55, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:44, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:32, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:35, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:36, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:28, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:19, 2.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:39, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:33, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:24, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:28, 1.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:30, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:22, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:15, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:38, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:31, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:22, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:25, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:21, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:15, 2.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:19, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:16, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:11, 3.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:20, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:16, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:18, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:18, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:09, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 3.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:03, 2.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<01:15, 136kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:51, 190kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:30, 270kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:17, 353kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:12, 458kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:08, 633kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 789kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 1.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.39MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 883/400000 [00:00<00:45, 8829.59it/s]  0%|          | 1778/400000 [00:00<00:44, 8863.04it/s]  1%|          | 2687/400000 [00:00<00:44, 8928.74it/s]  1%|          | 3581/400000 [00:00<00:44, 8931.38it/s]  1%|          | 4459/400000 [00:00<00:44, 8882.84it/s]  1%|▏         | 5392/400000 [00:00<00:43, 9011.35it/s]  2%|▏         | 6293/400000 [00:00<00:43, 9008.89it/s]  2%|▏         | 7208/400000 [00:00<00:43, 9048.47it/s]  2%|▏         | 8103/400000 [00:00<00:43, 9017.61it/s]  2%|▏         | 9023/400000 [00:01<00:43, 9071.51it/s]  2%|▏         | 9954/400000 [00:01<00:42, 9139.72it/s]  3%|▎         | 10882/400000 [00:01<00:42, 9180.99it/s]  3%|▎         | 11797/400000 [00:01<00:42, 9169.54it/s]  3%|▎         | 12706/400000 [00:01<00:43, 8902.42it/s]  3%|▎         | 13625/400000 [00:01<00:42, 8986.09it/s]  4%|▎         | 14532/400000 [00:01<00:42, 9011.02it/s]  4%|▍         | 15433/400000 [00:01<00:42, 9009.61it/s]  4%|▍         | 16333/400000 [00:01<00:42, 8954.92it/s]  4%|▍         | 17228/400000 [00:01<00:42, 8924.08it/s]  5%|▍         | 18132/400000 [00:02<00:42, 8957.00it/s]  5%|▍         | 19028/400000 [00:02<00:42, 8888.73it/s]  5%|▍         | 19917/400000 [00:02<00:42, 8866.97it/s]  5%|▌         | 20841/400000 [00:02<00:42, 8974.08it/s]  5%|▌         | 21739/400000 [00:02<00:42, 8905.47it/s]  6%|▌         | 22643/400000 [00:02<00:42, 8943.57it/s]  6%|▌         | 23538/400000 [00:02<00:42, 8924.40it/s]  6%|▌         | 24455/400000 [00:02<00:41, 8993.74it/s]  6%|▋         | 25355/400000 [00:02<00:41, 8931.35it/s]  7%|▋         | 26249/400000 [00:02<00:41, 8902.42it/s]  7%|▋         | 27173/400000 [00:03<00:41, 9000.23it/s]  7%|▋         | 28085/400000 [00:03<00:41, 9033.97it/s]  7%|▋         | 29020/400000 [00:03<00:40, 9123.95it/s]  7%|▋         | 29951/400000 [00:03<00:40, 9178.64it/s]  8%|▊         | 30870/400000 [00:03<00:40, 9005.82it/s]  8%|▊         | 31800/400000 [00:03<00:40, 9091.68it/s]  8%|▊         | 32717/400000 [00:03<00:40, 9114.08it/s]  8%|▊         | 33656/400000 [00:03<00:39, 9194.84it/s]  9%|▊         | 34577/400000 [00:03<00:40, 9099.43it/s]  9%|▉         | 35488/400000 [00:03<00:40, 9053.92it/s]  9%|▉         | 36401/400000 [00:04<00:40, 9075.10it/s]  9%|▉         | 37327/400000 [00:04<00:39, 9127.24it/s] 10%|▉         | 38256/400000 [00:04<00:39, 9174.42it/s] 10%|▉         | 39174/400000 [00:04<00:39, 9073.07it/s] 10%|█         | 40102/400000 [00:04<00:39, 9132.23it/s] 10%|█         | 41017/400000 [00:04<00:39, 9135.04it/s] 10%|█         | 41931/400000 [00:04<00:39, 9099.00it/s] 11%|█         | 42864/400000 [00:04<00:38, 9166.72it/s] 11%|█         | 43781/400000 [00:04<00:39, 9112.03it/s] 11%|█         | 44698/400000 [00:04<00:38, 9127.12it/s] 11%|█▏        | 45612/400000 [00:05<00:38, 9130.12it/s] 12%|█▏        | 46526/400000 [00:05<00:38, 9121.49it/s] 12%|█▏        | 47446/400000 [00:05<00:38, 9143.94it/s] 12%|█▏        | 48361/400000 [00:05<00:38, 9127.02it/s] 12%|█▏        | 49277/400000 [00:05<00:38, 9135.12it/s] 13%|█▎        | 50216/400000 [00:05<00:37, 9207.62it/s] 13%|█▎        | 51137/400000 [00:05<00:37, 9183.20it/s] 13%|█▎        | 52066/400000 [00:05<00:37, 9214.13it/s] 13%|█▎        | 52988/400000 [00:05<00:38, 9122.89it/s] 13%|█▎        | 53935/400000 [00:05<00:37, 9221.44it/s] 14%|█▎        | 54858/400000 [00:06<00:37, 9203.67it/s] 14%|█▍        | 55802/400000 [00:06<00:37, 9272.24it/s] 14%|█▍        | 56757/400000 [00:06<00:36, 9352.59it/s] 14%|█▍        | 57694/400000 [00:06<00:36, 9356.39it/s] 15%|█▍        | 58641/400000 [00:06<00:36, 9389.30it/s] 15%|█▍        | 59590/400000 [00:06<00:36, 9418.86it/s] 15%|█▌        | 60533/400000 [00:06<00:36, 9355.05it/s] 15%|█▌        | 61469/400000 [00:06<00:36, 9274.97it/s] 16%|█▌        | 62397/400000 [00:06<00:36, 9185.11it/s] 16%|█▌        | 63316/400000 [00:06<00:36, 9118.81it/s] 16%|█▌        | 64229/400000 [00:07<00:36, 9075.30it/s] 16%|█▋        | 65137/400000 [00:07<00:37, 9046.14it/s] 17%|█▋        | 66070/400000 [00:07<00:36, 9126.90it/s] 17%|█▋        | 66996/400000 [00:07<00:36, 9164.66it/s] 17%|█▋        | 67926/400000 [00:07<00:36, 9201.91it/s] 17%|█▋        | 68847/400000 [00:07<00:37, 8890.90it/s] 17%|█▋        | 69739/400000 [00:07<00:37, 8793.59it/s] 18%|█▊        | 70654/400000 [00:07<00:37, 8897.26it/s] 18%|█▊        | 71556/400000 [00:07<00:36, 8931.21it/s] 18%|█▊        | 72501/400000 [00:07<00:36, 9078.40it/s] 18%|█▊        | 73446/400000 [00:08<00:35, 9185.13it/s] 19%|█▊        | 74390/400000 [00:08<00:35, 9259.66it/s] 19%|█▉        | 75323/400000 [00:08<00:34, 9278.56it/s] 19%|█▉        | 76252/400000 [00:08<00:35, 9194.36it/s] 19%|█▉        | 77185/400000 [00:08<00:34, 9234.26it/s] 20%|█▉        | 78120/400000 [00:08<00:34, 9267.53it/s] 20%|█▉        | 79048/400000 [00:08<00:34, 9250.24it/s] 20%|█▉        | 79974/400000 [00:08<00:35, 9050.39it/s] 20%|██        | 80881/400000 [00:08<00:35, 8983.17it/s] 20%|██        | 81817/400000 [00:08<00:34, 9091.21it/s] 21%|██        | 82751/400000 [00:09<00:34, 9163.42it/s] 21%|██        | 83696/400000 [00:09<00:34, 9245.24it/s] 21%|██        | 84628/400000 [00:09<00:34, 9267.40it/s] 21%|██▏       | 85556/400000 [00:09<00:34, 9157.26it/s] 22%|██▏       | 86473/400000 [00:09<00:34, 9122.79it/s] 22%|██▏       | 87406/400000 [00:09<00:34, 9183.14it/s] 22%|██▏       | 88353/400000 [00:09<00:33, 9265.64it/s] 22%|██▏       | 89286/400000 [00:09<00:33, 9282.48it/s] 23%|██▎       | 90215/400000 [00:09<00:33, 9210.57it/s] 23%|██▎       | 91137/400000 [00:10<00:33, 9118.45it/s] 23%|██▎       | 92050/400000 [00:10<00:34, 9037.21it/s] 23%|██▎       | 92963/400000 [00:10<00:33, 9063.40it/s] 23%|██▎       | 93880/400000 [00:10<00:33, 9095.00it/s] 24%|██▎       | 94790/400000 [00:10<00:33, 9059.57it/s] 24%|██▍       | 95718/400000 [00:10<00:33, 9122.50it/s] 24%|██▍       | 96649/400000 [00:10<00:33, 9175.51it/s] 24%|██▍       | 97578/400000 [00:10<00:32, 9207.11it/s] 25%|██▍       | 98499/400000 [00:10<00:32, 9197.94it/s] 25%|██▍       | 99419/400000 [00:10<00:33, 9075.10it/s] 25%|██▌       | 100340/400000 [00:11<00:32, 9112.48it/s] 25%|██▌       | 101276/400000 [00:11<00:32, 9184.48it/s] 26%|██▌       | 102218/400000 [00:11<00:32, 9253.48it/s] 26%|██▌       | 103148/400000 [00:11<00:32, 9265.95it/s] 26%|██▌       | 104075/400000 [00:11<00:32, 8971.77it/s] 26%|██▌       | 104982/400000 [00:11<00:32, 8985.50it/s] 26%|██▋       | 105883/400000 [00:11<00:32, 8982.10it/s] 27%|██▋       | 106783/400000 [00:11<00:32, 8942.10it/s] 27%|██▋       | 107679/400000 [00:11<00:33, 8664.77it/s] 27%|██▋       | 108573/400000 [00:11<00:33, 8743.03it/s] 27%|██▋       | 109492/400000 [00:12<00:32, 8872.04it/s] 28%|██▊       | 110402/400000 [00:12<00:32, 8937.65it/s] 28%|██▊       | 111326/400000 [00:12<00:31, 9025.43it/s] 28%|██▊       | 112270/400000 [00:12<00:31, 9144.08it/s] 28%|██▊       | 113199/400000 [00:12<00:31, 9184.98it/s] 29%|██▊       | 114139/400000 [00:12<00:30, 9245.67it/s] 29%|██▉       | 115065/400000 [00:12<00:30, 9220.33it/s] 29%|██▉       | 115988/400000 [00:12<00:30, 9168.30it/s] 29%|██▉       | 116906/400000 [00:12<00:31, 8975.97it/s] 29%|██▉       | 117805/400000 [00:12<00:31, 8897.78it/s] 30%|██▉       | 118730/400000 [00:13<00:31, 8997.99it/s] 30%|██▉       | 119638/400000 [00:13<00:31, 9022.12it/s] 30%|███       | 120563/400000 [00:13<00:30, 9087.72it/s] 30%|███       | 121473/400000 [00:13<00:30, 9047.55it/s] 31%|███       | 122379/400000 [00:13<00:30, 8975.86it/s] 31%|███       | 123284/400000 [00:13<00:30, 8996.09it/s] 31%|███       | 124208/400000 [00:13<00:30, 9066.36it/s] 31%|███▏      | 125135/400000 [00:13<00:30, 9126.33it/s] 32%|███▏      | 126062/400000 [00:13<00:29, 9166.20it/s] 32%|███▏      | 126979/400000 [00:13<00:29, 9125.69it/s] 32%|███▏      | 127902/400000 [00:14<00:29, 9153.77it/s] 32%|███▏      | 128825/400000 [00:14<00:29, 9174.35it/s] 32%|███▏      | 129743/400000 [00:14<00:29, 9152.73it/s] 33%|███▎      | 130672/400000 [00:14<00:29, 9192.13it/s] 33%|███▎      | 131592/400000 [00:14<00:29, 9162.33it/s] 33%|███▎      | 132514/400000 [00:14<00:29, 9179.41it/s] 33%|███▎      | 133433/400000 [00:14<00:29, 9094.67it/s] 34%|███▎      | 134355/400000 [00:14<00:29, 9131.39it/s] 34%|███▍      | 135269/400000 [00:14<00:29, 8883.25it/s] 34%|███▍      | 136159/400000 [00:14<00:29, 8865.07it/s] 34%|███▍      | 137055/400000 [00:15<00:29, 8892.51it/s] 34%|███▍      | 137973/400000 [00:15<00:29, 8975.96it/s] 35%|███▍      | 138882/400000 [00:15<00:28, 9007.55it/s] 35%|███▍      | 139795/400000 [00:15<00:28, 9043.77it/s] 35%|███▌      | 140700/400000 [00:15<00:28, 9045.35it/s] 35%|███▌      | 141613/400000 [00:15<00:28, 9069.29it/s] 36%|███▌      | 142540/400000 [00:15<00:28, 9126.39it/s] 36%|███▌      | 143467/400000 [00:15<00:27, 9168.85it/s] 36%|███▌      | 144385/400000 [00:15<00:28, 9094.65it/s] 36%|███▋      | 145295/400000 [00:15<00:28, 9082.32it/s] 37%|███▋      | 146207/400000 [00:16<00:27, 9091.09it/s] 37%|███▋      | 147125/400000 [00:16<00:27, 9115.67it/s] 37%|███▋      | 148037/400000 [00:16<00:27, 9037.31it/s] 37%|███▋      | 148955/400000 [00:16<00:27, 9079.49it/s] 37%|███▋      | 149864/400000 [00:16<00:28, 8832.63it/s] 38%|███▊      | 150768/400000 [00:16<00:28, 8891.19it/s] 38%|███▊      | 151665/400000 [00:16<00:27, 8912.50it/s] 38%|███▊      | 152558/400000 [00:16<00:28, 8721.80it/s] 38%|███▊      | 153493/400000 [00:16<00:27, 8899.50it/s] 39%|███▊      | 154397/400000 [00:17<00:27, 8940.20it/s] 39%|███▉      | 155331/400000 [00:17<00:27, 9055.21it/s] 39%|███▉      | 156271/400000 [00:17<00:26, 9153.61it/s] 39%|███▉      | 157205/400000 [00:17<00:26, 9206.57it/s] 40%|███▉      | 158127/400000 [00:17<00:26, 9007.72it/s] 40%|███▉      | 159030/400000 [00:17<00:27, 8887.31it/s] 40%|███▉      | 159929/400000 [00:17<00:26, 8915.59it/s] 40%|████      | 160828/400000 [00:17<00:26, 8936.08it/s] 40%|████      | 161742/400000 [00:17<00:26, 8995.47it/s] 41%|████      | 162643/400000 [00:17<00:26, 8976.05it/s] 41%|████      | 163542/400000 [00:18<00:26, 8939.74it/s] 41%|████      | 164460/400000 [00:18<00:26, 9009.30it/s] 41%|████▏     | 165407/400000 [00:18<00:25, 9141.97it/s] 42%|████▏     | 166348/400000 [00:18<00:25, 9218.92it/s] 42%|████▏     | 167273/400000 [00:18<00:25, 9225.95it/s] 42%|████▏     | 168197/400000 [00:18<00:25, 9200.85it/s] 42%|████▏     | 169118/400000 [00:18<00:25, 9150.80it/s] 43%|████▎     | 170034/400000 [00:18<00:25, 9114.17it/s] 43%|████▎     | 170946/400000 [00:18<00:25, 8924.45it/s] 43%|████▎     | 171857/400000 [00:18<00:25, 8978.29it/s] 43%|████▎     | 172777/400000 [00:19<00:25, 9040.57it/s] 43%|████▎     | 173702/400000 [00:19<00:24, 9101.99it/s] 44%|████▎     | 174639/400000 [00:19<00:24, 9179.81it/s] 44%|████▍     | 175558/400000 [00:19<00:24, 9139.42it/s] 44%|████▍     | 176495/400000 [00:19<00:24, 9204.65it/s] 44%|████▍     | 177416/400000 [00:19<00:24, 9193.37it/s] 45%|████▍     | 178336/400000 [00:19<00:24, 9120.28it/s] 45%|████▍     | 179258/400000 [00:19<00:24, 9148.07it/s] 45%|████▌     | 180176/400000 [00:19<00:24, 9156.43it/s] 45%|████▌     | 181092/400000 [00:19<00:23, 9136.78it/s] 46%|████▌     | 182009/400000 [00:20<00:23, 9144.33it/s] 46%|████▌     | 182939/400000 [00:20<00:23, 9188.12it/s] 46%|████▌     | 183858/400000 [00:20<00:23, 9166.71it/s] 46%|████▌     | 184775/400000 [00:20<00:23, 9118.79it/s] 46%|████▋     | 185688/400000 [00:20<00:23, 9085.88it/s] 47%|████▋     | 186608/400000 [00:20<00:23, 9117.67it/s] 47%|████▋     | 187524/400000 [00:20<00:23, 9128.89it/s] 47%|████▋     | 188437/400000 [00:20<00:23, 9104.70it/s] 47%|████▋     | 189348/400000 [00:20<00:23, 8860.84it/s] 48%|████▊     | 190256/400000 [00:20<00:23, 8923.22it/s] 48%|████▊     | 191164/400000 [00:21<00:23, 8969.15it/s] 48%|████▊     | 192073/400000 [00:21<00:23, 9002.63it/s] 48%|████▊     | 193009/400000 [00:21<00:22, 9105.93it/s] 48%|████▊     | 193947/400000 [00:21<00:22, 9185.21it/s] 49%|████▊     | 194883/400000 [00:21<00:22, 9236.00it/s] 49%|████▉     | 195814/400000 [00:21<00:22, 9254.36it/s] 49%|████▉     | 196743/400000 [00:21<00:21, 9262.30it/s] 49%|████▉     | 197670/400000 [00:21<00:21, 9217.01it/s] 50%|████▉     | 198592/400000 [00:21<00:22, 9143.38it/s] 50%|████▉     | 199525/400000 [00:21<00:21, 9197.25it/s] 50%|█████     | 200446/400000 [00:22<00:21, 9173.09it/s] 50%|█████     | 201364/400000 [00:22<00:21, 9115.39it/s] 51%|█████     | 202304/400000 [00:22<00:21, 9196.20it/s] 51%|█████     | 203224/400000 [00:22<00:21, 9186.37it/s] 51%|█████     | 204143/400000 [00:22<00:21, 9056.92it/s] 51%|█████▏    | 205072/400000 [00:22<00:21, 9122.77it/s] 51%|█████▏    | 205991/400000 [00:22<00:21, 9140.94it/s] 52%|█████▏    | 206922/400000 [00:22<00:21, 9161.68it/s] 52%|█████▏    | 207839/400000 [00:22<00:21, 8926.98it/s] 52%|█████▏    | 208769/400000 [00:22<00:21, 9032.78it/s] 52%|█████▏    | 209681/400000 [00:23<00:21, 9056.09it/s] 53%|█████▎    | 210605/400000 [00:23<00:20, 9107.83it/s] 53%|█████▎    | 211537/400000 [00:23<00:20, 9169.28it/s] 53%|█████▎    | 212478/400000 [00:23<00:20, 9238.27it/s] 53%|█████▎    | 213403/400000 [00:23<00:20, 9141.87it/s] 54%|█████▎    | 214318/400000 [00:23<00:20, 9028.67it/s] 54%|█████▍    | 215222/400000 [00:23<00:20, 9025.01it/s] 54%|█████▍    | 216138/400000 [00:23<00:20, 9063.50it/s] 54%|█████▍    | 217063/400000 [00:23<00:20, 9116.81it/s] 55%|█████▍    | 218005/400000 [00:23<00:19, 9203.36it/s] 55%|█████▍    | 218946/400000 [00:24<00:19, 9262.51it/s] 55%|█████▍    | 219873/400000 [00:24<00:19, 9250.19it/s] 55%|█████▌    | 220799/400000 [00:24<00:19, 9252.81it/s] 55%|█████▌    | 221731/400000 [00:24<00:19, 9270.27it/s] 56%|█████▌    | 222659/400000 [00:24<00:19, 9258.98it/s] 56%|█████▌    | 223586/400000 [00:24<00:19, 9204.97it/s] 56%|█████▌    | 224507/400000 [00:24<00:19, 9127.16it/s] 56%|█████▋    | 225447/400000 [00:24<00:18, 9206.42it/s] 57%|█████▋    | 226368/400000 [00:24<00:19, 9007.89it/s] 57%|█████▋    | 227304/400000 [00:24<00:18, 9109.64it/s] 57%|█████▋    | 228217/400000 [00:25<00:18, 9070.91it/s] 57%|█████▋    | 229125/400000 [00:25<00:18, 9034.52it/s] 58%|█████▊    | 230041/400000 [00:25<00:18, 9069.90it/s] 58%|█████▊    | 230949/400000 [00:25<00:18, 9034.07it/s] 58%|█████▊    | 231853/400000 [00:25<00:18, 8954.16it/s] 58%|█████▊    | 232795/400000 [00:25<00:18, 9087.74it/s] 58%|█████▊    | 233714/400000 [00:25<00:18, 9118.15it/s] 59%|█████▊    | 234643/400000 [00:25<00:18, 9168.80it/s] 59%|█████▉    | 235561/400000 [00:25<00:17, 9154.53it/s] 59%|█████▉    | 236501/400000 [00:26<00:17, 9226.49it/s] 59%|█████▉    | 237434/400000 [00:26<00:17, 9254.54it/s] 60%|█████▉    | 238362/400000 [00:26<00:17, 9260.14it/s] 60%|█████▉    | 239307/400000 [00:26<00:17, 9313.69it/s] 60%|██████    | 240239/400000 [00:26<00:17, 9277.14it/s] 60%|██████    | 241167/400000 [00:26<00:17, 9224.61it/s] 61%|██████    | 242090/400000 [00:26<00:17, 9108.81it/s] 61%|██████    | 243002/400000 [00:26<00:17, 9063.99it/s] 61%|██████    | 243933/400000 [00:26<00:17, 9134.81it/s] 61%|██████    | 244847/400000 [00:26<00:17, 8871.63it/s] 61%|██████▏   | 245757/400000 [00:27<00:17, 8936.27it/s] 62%|██████▏   | 246693/400000 [00:27<00:16, 9059.03it/s] 62%|██████▏   | 247601/400000 [00:27<00:16, 9051.64it/s] 62%|██████▏   | 248518/400000 [00:27<00:16, 9085.32it/s] 62%|██████▏   | 249433/400000 [00:27<00:16, 9104.27it/s] 63%|██████▎   | 250377/400000 [00:27<00:16, 9201.58it/s] 63%|██████▎   | 251317/400000 [00:27<00:16, 9257.99it/s] 63%|██████▎   | 252244/400000 [00:27<00:16, 9069.97it/s] 63%|██████▎   | 253156/400000 [00:27<00:16, 9083.60it/s] 64%|██████▎   | 254066/400000 [00:27<00:16, 9055.83it/s] 64%|██████▎   | 254973/400000 [00:28<00:16, 8932.47it/s] 64%|██████▍   | 255868/400000 [00:28<00:16, 8857.26it/s] 64%|██████▍   | 256755/400000 [00:28<00:16, 8851.42it/s] 64%|██████▍   | 257655/400000 [00:28<00:16, 8894.84it/s] 65%|██████▍   | 258545/400000 [00:28<00:15, 8882.66it/s] 65%|██████▍   | 259434/400000 [00:28<00:15, 8863.22it/s] 65%|██████▌   | 260338/400000 [00:28<00:15, 8913.81it/s] 65%|██████▌   | 261246/400000 [00:28<00:15, 8961.76it/s] 66%|██████▌   | 262152/400000 [00:28<00:15, 8989.01it/s] 66%|██████▌   | 263052/400000 [00:28<00:15, 8943.45it/s] 66%|██████▌   | 263974/400000 [00:29<00:15, 9024.17it/s] 66%|██████▌   | 264877/400000 [00:29<00:15, 8980.98it/s] 66%|██████▋   | 265776/400000 [00:29<00:14, 8977.79it/s] 67%|██████▋   | 266674/400000 [00:29<00:14, 8970.51it/s] 67%|██████▋   | 267572/400000 [00:29<00:14, 8957.88it/s] 67%|██████▋   | 268468/400000 [00:29<00:14, 8935.26it/s] 67%|██████▋   | 269362/400000 [00:29<00:14, 8906.56it/s] 68%|██████▊   | 270253/400000 [00:29<00:14, 8884.90it/s] 68%|██████▊   | 271142/400000 [00:29<00:14, 8826.38it/s] 68%|██████▊   | 272057/400000 [00:29<00:14, 8918.46it/s] 68%|██████▊   | 272956/400000 [00:30<00:14, 8937.93it/s] 68%|██████▊   | 273864/400000 [00:30<00:14, 8978.19it/s] 69%|██████▊   | 274763/400000 [00:30<00:14, 8817.95it/s] 69%|██████▉   | 275648/400000 [00:30<00:14, 8825.19it/s] 69%|██████▉   | 276539/400000 [00:30<00:13, 8848.80it/s] 69%|██████▉   | 277427/400000 [00:30<00:13, 8856.11it/s] 70%|██████▉   | 278320/400000 [00:30<00:13, 8877.83it/s] 70%|██████▉   | 279208/400000 [00:30<00:13, 8854.81it/s] 70%|███████   | 280094/400000 [00:30<00:13, 8625.15it/s] 70%|███████   | 280968/400000 [00:30<00:13, 8656.65it/s] 70%|███████   | 281860/400000 [00:31<00:13, 8733.63it/s] 71%|███████   | 282740/400000 [00:31<00:13, 8752.73it/s] 71%|███████   | 283616/400000 [00:31<00:13, 8740.94it/s] 71%|███████   | 284505/400000 [00:31<00:13, 8782.89it/s] 71%|███████▏  | 285384/400000 [00:31<00:13, 8759.22it/s] 72%|███████▏  | 286305/400000 [00:31<00:12, 8888.42it/s] 72%|███████▏  | 287199/400000 [00:31<00:12, 8902.19it/s] 72%|███████▏  | 288090/400000 [00:31<00:12, 8893.12it/s] 72%|███████▏  | 288990/400000 [00:31<00:12, 8923.27it/s] 72%|███████▏  | 289892/400000 [00:31<00:12, 8951.20it/s] 73%|███████▎  | 290788/400000 [00:32<00:12, 8928.73it/s] 73%|███████▎  | 291687/400000 [00:32<00:12, 8944.00it/s] 73%|███████▎  | 292589/400000 [00:32<00:11, 8965.47it/s] 73%|███████▎  | 293521/400000 [00:32<00:11, 9068.84it/s] 74%|███████▎  | 294429/400000 [00:32<00:12, 8792.05it/s] 74%|███████▍  | 295330/400000 [00:32<00:11, 8853.65it/s] 74%|███████▍  | 296235/400000 [00:32<00:11, 8911.46it/s] 74%|███████▍  | 297128/400000 [00:32<00:11, 8883.26it/s] 75%|███████▍  | 298018/400000 [00:32<00:11, 8563.15it/s] 75%|███████▍  | 298929/400000 [00:32<00:11, 8719.70it/s] 75%|███████▍  | 299863/400000 [00:33<00:11, 8895.37it/s] 75%|███████▌  | 300803/400000 [00:33<00:10, 9038.61it/s] 75%|███████▌  | 301725/400000 [00:33<00:10, 9090.13it/s] 76%|███████▌  | 302658/400000 [00:33<00:10, 9158.04it/s] 76%|███████▌  | 303576/400000 [00:33<00:10, 9146.04it/s] 76%|███████▌  | 304494/400000 [00:33<00:10, 9156.12it/s] 76%|███████▋  | 305411/400000 [00:33<00:10, 9146.24it/s] 77%|███████▋  | 306327/400000 [00:33<00:10, 9082.32it/s] 77%|███████▋  | 307239/400000 [00:33<00:10, 9093.04it/s] 77%|███████▋  | 308167/400000 [00:34<00:10, 9146.53it/s] 77%|███████▋  | 309104/400000 [00:34<00:09, 9209.85it/s] 78%|███████▊  | 310039/400000 [00:34<00:09, 9250.11it/s] 78%|███████▊  | 310965/400000 [00:34<00:09, 9209.88it/s] 78%|███████▊  | 311887/400000 [00:34<00:09, 9170.48it/s] 78%|███████▊  | 312805/400000 [00:34<00:09, 9167.85it/s] 78%|███████▊  | 313722/400000 [00:34<00:09, 9141.85it/s] 79%|███████▊  | 314637/400000 [00:34<00:09, 9057.96it/s] 79%|███████▉  | 315544/400000 [00:34<00:09, 9030.34it/s] 79%|███████▉  | 316448/400000 [00:34<00:09, 8882.33it/s] 79%|███████▉  | 317384/400000 [00:35<00:09, 9019.34it/s] 80%|███████▉  | 318318/400000 [00:35<00:08, 9111.83it/s] 80%|███████▉  | 319259/400000 [00:35<00:08, 9196.63it/s] 80%|████████  | 320186/400000 [00:35<00:08, 9216.78it/s] 80%|████████  | 321119/400000 [00:35<00:08, 9247.96it/s] 81%|████████  | 322049/400000 [00:35<00:08, 9262.02it/s] 81%|████████  | 322976/400000 [00:35<00:08, 9259.81it/s] 81%|████████  | 323906/400000 [00:35<00:08, 9270.09it/s] 81%|████████  | 324834/400000 [00:35<00:08, 9142.25it/s] 81%|████████▏ | 325749/400000 [00:35<00:08, 9137.45it/s] 82%|████████▏ | 326680/400000 [00:36<00:07, 9186.24it/s] 82%|████████▏ | 327612/400000 [00:36<00:07, 9223.92it/s] 82%|████████▏ | 328546/400000 [00:36<00:07, 9256.55it/s] 82%|████████▏ | 329472/400000 [00:36<00:07, 9253.49it/s] 83%|████████▎ | 330412/400000 [00:36<00:07, 9296.93it/s] 83%|████████▎ | 331342/400000 [00:36<00:07, 9089.06it/s] 83%|████████▎ | 332276/400000 [00:36<00:07, 9160.78it/s] 83%|████████▎ | 333213/400000 [00:36<00:07, 9220.25it/s] 84%|████████▎ | 334136/400000 [00:36<00:07, 9203.58it/s] 84%|████████▍ | 335058/400000 [00:36<00:07, 9208.48it/s] 84%|████████▍ | 335980/400000 [00:37<00:06, 9175.50it/s] 84%|████████▍ | 336898/400000 [00:37<00:06, 9149.40it/s] 84%|████████▍ | 337814/400000 [00:37<00:06, 9124.01it/s] 85%|████████▍ | 338737/400000 [00:37<00:06, 9155.23it/s] 85%|████████▍ | 339663/400000 [00:37<00:06, 9186.33it/s] 85%|████████▌ | 340585/400000 [00:37<00:06, 9194.61it/s] 85%|████████▌ | 341508/400000 [00:37<00:06, 9202.81it/s] 86%|████████▌ | 342444/400000 [00:37<00:06, 9247.19it/s] 86%|████████▌ | 343369/400000 [00:37<00:06, 9104.63it/s] 86%|████████▌ | 344298/400000 [00:37<00:06, 9159.36it/s] 86%|████████▋ | 345241/400000 [00:38<00:05, 9238.24it/s] 87%|████████▋ | 346184/400000 [00:38<00:05, 9292.53it/s] 87%|████████▋ | 347117/400000 [00:38<00:05, 9301.08it/s] 87%|████████▋ | 348048/400000 [00:38<00:05, 9219.97it/s] 87%|████████▋ | 348971/400000 [00:38<00:05, 9212.70it/s] 87%|████████▋ | 349893/400000 [00:38<00:05, 9175.53it/s] 88%|████████▊ | 350811/400000 [00:38<00:05, 9144.27it/s] 88%|████████▊ | 351726/400000 [00:38<00:05, 9140.48it/s] 88%|████████▊ | 352641/400000 [00:38<00:05, 9023.88it/s] 88%|████████▊ | 353546/400000 [00:38<00:05, 9029.72it/s] 89%|████████▊ | 354464/400000 [00:39<00:05, 9071.85it/s] 89%|████████▉ | 355372/400000 [00:39<00:04, 8966.73it/s] 89%|████████▉ | 356270/400000 [00:39<00:04, 8880.86it/s] 89%|████████▉ | 357202/400000 [00:39<00:04, 9006.81it/s] 90%|████████▉ | 358136/400000 [00:39<00:04, 9102.82it/s] 90%|████████▉ | 359070/400000 [00:39<00:04, 9170.03it/s] 90%|████████▉ | 359988/400000 [00:39<00:04, 9168.21it/s] 90%|█████████ | 360906/400000 [00:39<00:04, 9131.94it/s] 90%|█████████ | 361820/400000 [00:39<00:04, 9074.56it/s] 91%|█████████ | 362728/400000 [00:39<00:04, 9068.36it/s] 91%|█████████ | 363645/400000 [00:40<00:03, 9097.24it/s] 91%|█████████ | 364555/400000 [00:40<00:03, 9091.01it/s] 91%|█████████▏| 365502/400000 [00:40<00:03, 9199.00it/s] 92%|█████████▏| 366423/400000 [00:40<00:03, 9173.82it/s] 92%|█████████▏| 367351/400000 [00:40<00:03, 9203.74it/s] 92%|█████████▏| 368272/400000 [00:40<00:03, 8916.34it/s] 92%|█████████▏| 369188/400000 [00:40<00:03, 8985.61it/s] 93%|█████████▎| 370105/400000 [00:40<00:03, 9038.92it/s] 93%|█████████▎| 371011/400000 [00:40<00:03, 9011.48it/s] 93%|█████████▎| 371933/400000 [00:40<00:03, 9072.92it/s] 93%|█████████▎| 372862/400000 [00:41<00:02, 9136.78it/s] 93%|█████████▎| 373777/400000 [00:41<00:02, 9058.53it/s] 94%|█████████▎| 374702/400000 [00:41<00:02, 9113.21it/s] 94%|█████████▍| 375615/400000 [00:41<00:02, 9115.68it/s] 94%|█████████▍| 376554/400000 [00:41<00:02, 9195.41it/s] 94%|█████████▍| 377484/400000 [00:41<00:02, 9225.79it/s] 95%|█████████▍| 378418/400000 [00:41<00:02, 9257.27it/s] 95%|█████████▍| 379360/400000 [00:41<00:02, 9303.23it/s] 95%|█████████▌| 380291/400000 [00:41<00:02, 9208.48it/s] 95%|█████████▌| 381230/400000 [00:41<00:02, 9260.29it/s] 96%|█████████▌| 382157/400000 [00:42<00:01, 9197.63it/s] 96%|█████████▌| 383078/400000 [00:42<00:01, 9013.71it/s] 96%|█████████▌| 384013/400000 [00:42<00:01, 9110.87it/s] 96%|█████████▌| 384932/400000 [00:42<00:01, 9134.37it/s] 96%|█████████▋| 385871/400000 [00:42<00:01, 9206.81it/s] 97%|█████████▋| 386809/400000 [00:42<00:01, 9256.61it/s] 97%|█████████▋| 387745/400000 [00:42<00:01, 9285.63it/s] 97%|█████████▋| 388683/400000 [00:42<00:01, 9312.60it/s] 97%|█████████▋| 389615/400000 [00:42<00:01, 9208.16it/s] 98%|█████████▊| 390539/400000 [00:42<00:01, 9217.41it/s] 98%|█████████▊| 391476/400000 [00:43<00:00, 9261.32it/s] 98%|█████████▊| 392408/400000 [00:43<00:00, 9276.61it/s] 98%|█████████▊| 393350/400000 [00:43<00:00, 9318.67it/s] 99%|█████████▊| 394283/400000 [00:43<00:00, 9301.88it/s] 99%|█████████▉| 395214/400000 [00:43<00:00, 9229.03it/s] 99%|█████████▉| 396149/400000 [00:43<00:00, 9264.53it/s] 99%|█████████▉| 397076/400000 [00:43<00:00, 9232.10it/s]100%|█████████▉| 398000/400000 [00:43<00:00, 9136.26it/s]100%|█████████▉| 398914/400000 [00:43<00:00, 9048.12it/s]100%|█████████▉| 399820/400000 [00:44<00:00, 9040.04it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9085.34it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd77d6b5f98> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011306846517008943 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010946807255314345 	 Accuracy: 56

  model saves at 56% accuracy 

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

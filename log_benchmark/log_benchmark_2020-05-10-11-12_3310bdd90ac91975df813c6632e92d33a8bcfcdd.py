
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fc9131e74a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 11:12:55.467531
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 11:12:55.471204
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 11:12:55.474256
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 11:12:55.477386
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fc8ec267978> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355728.7812
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 280984.1250
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 183520.5625
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 105947.1484
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 55674.0859
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 29024.6074
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 16281.3057
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 9896.4434
Epoch 9/10

1/1 [==============================] - 0s 92ms/step - loss: 6507.8535
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 4606.8164

  #### Inference Need return ypred, ytrue ######################### 
[[ 7.78683901e-01  2.89916563e+00  1.75411558e+00 -9.35888350e-01
   8.01086724e-01  3.11387062e-01 -1.08161139e+00  3.87562186e-01
  -6.50189519e-01  2.36142612e+00  3.04684579e-01 -4.19119120e-01
  -1.48740196e+00 -1.65941429e+00  7.78308988e-01 -2.83496618e+00
  -9.62220550e-01 -1.11748362e+00  1.13392866e+00 -6.19520247e-01
  -1.57440498e-01 -1.18907523e+00  1.68671596e+00  8.16205740e-01
  -2.85378903e-01  3.93825769e-03 -1.60609961e+00 -6.74881041e-02
  -1.46052170e+00  1.77191591e+00 -1.10048652e-01 -3.21630299e-01
  -1.05885172e+00  5.63928246e-01  5.08304954e-01 -2.20131204e-01
   1.79329216e-01  7.18476415e-01 -1.72142231e+00 -3.56362373e-01
   6.41179681e-01 -6.55792534e-01  1.20620728e+00  2.44016266e+00
   2.17091823e+00 -2.85019875e-02 -2.10473561e+00  3.63070071e-01
  -1.41491032e+00  1.07802749e+00 -6.03468657e-01 -2.24095523e-01
  -8.17660570e-01  1.00649035e+00  1.64422750e-01  1.43816340e+00
   3.22318554e-01 -1.23086882e+00 -7.08754420e-01 -2.97646970e-01
   1.82820296e+00 -1.20299149e+00  2.63535947e-01  6.63519859e-01
  -2.34233737e-02 -1.64562416e+00 -1.27276510e-01  4.41307485e-01
  -4.66698050e-01  6.58100247e-01 -9.63943839e-01 -2.02208281e-01
  -2.76171684e-01 -1.57554865e+00  8.75858247e-01 -6.25997901e-01
  -2.17033207e-01 -2.27661327e-01  1.17029250e+00 -5.98146617e-02
   2.47540188e+00 -2.93819696e-01 -4.40097481e-01 -4.09768343e-01
  -1.47252226e+00  1.28419518e+00  1.25243711e+00  3.80706340e-02
  -4.38316196e-01  7.95381188e-01  6.32408261e-01 -1.59065127e+00
   3.54280502e-01  8.79494250e-01 -2.92214006e-01  4.68887269e-01
   1.68803644e+00  1.99374974e-01  6.01164818e-01 -9.80891109e-01
  -8.76724601e-01  1.42116398e-02  1.16768897e+00 -6.95355773e-01
   6.67094707e-01 -4.84230518e-02 -3.87336105e-01 -5.24280488e-01
   7.35412598e-01  2.40984130e+00 -1.23856115e+00 -8.68042886e-01
  -1.30225456e+00 -7.77974725e-01 -8.77440810e-01  1.98788679e+00
   6.07671976e-01 -4.10922289e-01 -9.88886476e-01  1.73999560e+00
   7.05932081e-03  9.74897957e+00  1.08144064e+01  1.09524899e+01
   1.08029585e+01  1.11608496e+01  1.27177782e+01  1.06723166e+01
   1.14951401e+01  9.43292141e+00  1.12690029e+01  9.69840717e+00
   1.04651184e+01  9.71942043e+00  1.07759686e+01  9.45102978e+00
   8.78768921e+00  1.18033676e+01  1.10318441e+01  1.00569506e+01
   1.31719160e+01  9.01356983e+00  9.51511860e+00  1.11323690e+01
   1.00208950e+01  1.27277184e+01  1.19236460e+01  9.16524887e+00
   1.19480124e+01  1.06898661e+01  1.07703171e+01  1.09625587e+01
   1.07578630e+01  1.02306576e+01  1.14084396e+01  1.18891506e+01
   1.14399376e+01  9.93538380e+00  1.28670578e+01  1.19688473e+01
   1.05729818e+01  1.32605486e+01  9.94166660e+00  1.21503725e+01
   1.10432711e+01  1.26871233e+01  7.84138107e+00  9.79800034e+00
   1.10241079e+01  1.14042006e+01  1.26252565e+01  1.33863325e+01
   1.16794415e+01  1.06785908e+01  9.87812424e+00  1.12840605e+01
   1.13831310e+01  1.01340561e+01  1.05759745e+01  9.86515808e+00
   1.51106429e+00  1.33289313e+00  4.70616221e-01  1.61950231e-01
   1.38982415e+00  2.42951727e+00  1.70572805e+00  2.37375975e-01
   9.73148942e-01  1.16111398e-01  7.92396009e-01  3.36563706e-01
   7.28963614e-01  8.47900510e-01  6.36599958e-01  1.52085900e-01
   1.78191543e-01  3.82764041e-01  5.35439134e-01  2.60652423e-01
   7.39469528e-02  7.68449426e-01  3.25776637e-01  3.13575745e-01
   6.59622908e-01  8.06843877e-01  8.39833200e-01  9.58400130e-01
   1.48260593e-01  7.12088764e-01  6.40268922e-01  1.42130017e+00
   3.07102442e-01  4.40429688e-01  2.25990415e-01  2.59229136e+00
   4.23362613e-01  1.85275686e+00  8.40953410e-01  8.64070833e-01
   1.24812305e-01  2.75246048e+00  3.54019761e-01  2.92245722e+00
   1.32718444e-01  1.68282795e+00  3.43541241e+00  1.26510167e+00
   3.14599419e+00  1.18269956e+00  1.56794727e-01  3.22140098e-01
   5.33661783e-01  6.27862632e-01  3.21718311e+00  2.14190149e+00
   4.38264012e-02  3.18784118e-01  2.32196212e-01  5.38798213e-01
   8.71353090e-01  2.97712564e-01  1.98608255e+00  3.14404011e-01
   7.68178701e-01  1.03188872e+00  3.47059774e+00  2.05001330e+00
   1.12552786e+00  5.90708137e-01  2.19827604e+00  6.89091682e-02
   8.48802626e-01  2.41307020e+00  1.50148392e-01  9.98473763e-01
   2.21973479e-01  8.05377364e-02  8.28911304e-01  1.82713199e+00
   2.19134212e+00  4.55980897e-01  2.14955258e+00  1.04935718e+00
   1.30329919e+00  2.68472493e-01  1.49473774e+00  1.65250134e+00
   1.51307940e+00  4.90255356e-01  1.16973698e-01  1.71330988e+00
   1.92471683e-01  1.47322237e+00  2.55799294e-01  6.35620236e-01
   3.79058838e-01  1.60762095e+00  6.78073108e-01  1.46107340e+00
   6.87106073e-01  7.42791235e-01  6.48402750e-01  1.53959787e+00
   1.41298234e+00  4.51267362e-01  2.40882111e+00  1.12845898e+00
   1.48654866e+00  2.51647353e+00  1.08375490e-01  3.46565485e-01
   2.36817741e+00  2.03884935e+00  6.55947149e-01  1.49120331e+00
   2.61655951e+00  3.30337000e+00  3.59615088e-01  5.24537623e-01
   2.75887132e-01  1.09760866e+01  1.10254488e+01  1.00917139e+01
   1.09580202e+01  1.06426306e+01  1.17720556e+01  9.43818569e+00
   1.14116783e+01  9.78680325e+00  1.10973644e+01  9.57161045e+00
   9.41388798e+00  1.00436621e+01  1.22000351e+01  9.69560909e+00
   1.05697432e+01  1.15416002e+01  1.22225723e+01  1.05928497e+01
   9.41425991e+00  1.11522980e+01  9.62268639e+00  1.06533899e+01
   9.47469425e+00  1.16015539e+01  1.03678856e+01  9.75893688e+00
   9.31346130e+00  1.00394335e+01  1.18420477e+01  1.08102884e+01
   1.05632486e+01  9.08105087e+00  8.23834705e+00  1.24690180e+01
   9.32343483e+00  1.07545195e+01  9.79234219e+00  1.03201456e+01
   9.90309048e+00  9.87323952e+00  1.21709547e+01  9.30890179e+00
   9.82856464e+00  1.04562206e+01  1.06634245e+01  9.97798824e+00
   1.08820400e+01  9.90352345e+00  1.14032297e+01  1.01996365e+01
   1.18440933e+01  1.13030586e+01  1.14965458e+01  9.72832203e+00
   9.75798035e+00  1.03560753e+01  1.06012020e+01  1.16753168e+01
  -7.87237263e+00 -9.21586990e+00  7.17531586e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 11:13:03.740438
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.4142
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 11:13:03.744198
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8380.57
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 11:13:03.747594
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.9516
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 11:13:03.750757
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -749.539
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140500896985552
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140499687199632
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140499687200136
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140499687200640
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140499687201144
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140499687201648

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fc8ff772b00> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.591915
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.560887
grad_step = 000002, loss = 0.538566
grad_step = 000003, loss = 0.518717
grad_step = 000004, loss = 0.494353
grad_step = 000005, loss = 0.463739
grad_step = 000006, loss = 0.429150
grad_step = 000007, loss = 0.394807
grad_step = 000008, loss = 0.368683
grad_step = 000009, loss = 0.351118
grad_step = 000010, loss = 0.339559
grad_step = 000011, loss = 0.322974
grad_step = 000012, loss = 0.303858
grad_step = 000013, loss = 0.288691
grad_step = 000014, loss = 0.278021
grad_step = 000015, loss = 0.268693
grad_step = 000016, loss = 0.257810
grad_step = 000017, loss = 0.244795
grad_step = 000018, loss = 0.231167
grad_step = 000019, loss = 0.218961
grad_step = 000020, loss = 0.208683
grad_step = 000021, loss = 0.198372
grad_step = 000022, loss = 0.186681
grad_step = 000023, loss = 0.174759
grad_step = 000024, loss = 0.164403
grad_step = 000025, loss = 0.154793
grad_step = 000026, loss = 0.145150
grad_step = 000027, loss = 0.135783
grad_step = 000028, loss = 0.127157
grad_step = 000029, loss = 0.119468
grad_step = 000030, loss = 0.112624
grad_step = 000031, loss = 0.106148
grad_step = 000032, loss = 0.099623
grad_step = 000033, loss = 0.093198
grad_step = 000034, loss = 0.087236
grad_step = 000035, loss = 0.081704
grad_step = 000036, loss = 0.076325
grad_step = 000037, loss = 0.071028
grad_step = 000038, loss = 0.066014
grad_step = 000039, loss = 0.061461
grad_step = 000040, loss = 0.057298
grad_step = 000041, loss = 0.053374
grad_step = 000042, loss = 0.049725
grad_step = 000043, loss = 0.046413
grad_step = 000044, loss = 0.043344
grad_step = 000045, loss = 0.040410
grad_step = 000046, loss = 0.037628
grad_step = 000047, loss = 0.035057
grad_step = 000048, loss = 0.032695
grad_step = 000049, loss = 0.030467
grad_step = 000050, loss = 0.028340
grad_step = 000051, loss = 0.026344
grad_step = 000052, loss = 0.024490
grad_step = 000053, loss = 0.022768
grad_step = 000054, loss = 0.021176
grad_step = 000055, loss = 0.019721
grad_step = 000056, loss = 0.018393
grad_step = 000057, loss = 0.017152
grad_step = 000058, loss = 0.015972
grad_step = 000059, loss = 0.014865
grad_step = 000060, loss = 0.013838
grad_step = 000061, loss = 0.012883
grad_step = 000062, loss = 0.011993
grad_step = 000063, loss = 0.011179
grad_step = 000064, loss = 0.010437
grad_step = 000065, loss = 0.009741
grad_step = 000066, loss = 0.009079
grad_step = 000067, loss = 0.008464
grad_step = 000068, loss = 0.007903
grad_step = 000069, loss = 0.007387
grad_step = 000070, loss = 0.006915
grad_step = 000071, loss = 0.006489
grad_step = 000072, loss = 0.006098
grad_step = 000073, loss = 0.005726
grad_step = 000074, loss = 0.005381
grad_step = 000075, loss = 0.005070
grad_step = 000076, loss = 0.004785
grad_step = 000077, loss = 0.004520
grad_step = 000078, loss = 0.004278
grad_step = 000079, loss = 0.004059
grad_step = 000080, loss = 0.003855
grad_step = 000081, loss = 0.003668
grad_step = 000082, loss = 0.003503
grad_step = 000083, loss = 0.003359
grad_step = 000084, loss = 0.003226
grad_step = 000085, loss = 0.003106
grad_step = 000086, loss = 0.002998
grad_step = 000087, loss = 0.002899
grad_step = 000088, loss = 0.002809
grad_step = 000089, loss = 0.002731
grad_step = 000090, loss = 0.002663
grad_step = 000091, loss = 0.002601
grad_step = 000092, loss = 0.002547
grad_step = 000093, loss = 0.002499
grad_step = 000094, loss = 0.002456
grad_step = 000095, loss = 0.002419
grad_step = 000096, loss = 0.002389
grad_step = 000097, loss = 0.002362
grad_step = 000098, loss = 0.002339
grad_step = 000099, loss = 0.002319
grad_step = 000100, loss = 0.002301
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002285
grad_step = 000102, loss = 0.002273
grad_step = 000103, loss = 0.002262
grad_step = 000104, loss = 0.002252
grad_step = 000105, loss = 0.002244
grad_step = 000106, loss = 0.002237
grad_step = 000107, loss = 0.002231
grad_step = 000108, loss = 0.002226
grad_step = 000109, loss = 0.002222
grad_step = 000110, loss = 0.002219
grad_step = 000111, loss = 0.002215
grad_step = 000112, loss = 0.002212
grad_step = 000113, loss = 0.002209
grad_step = 000114, loss = 0.002207
grad_step = 000115, loss = 0.002205
grad_step = 000116, loss = 0.002202
grad_step = 000117, loss = 0.002200
grad_step = 000118, loss = 0.002198
grad_step = 000119, loss = 0.002196
grad_step = 000120, loss = 0.002194
grad_step = 000121, loss = 0.002192
grad_step = 000122, loss = 0.002190
grad_step = 000123, loss = 0.002187
grad_step = 000124, loss = 0.002185
grad_step = 000125, loss = 0.002183
grad_step = 000126, loss = 0.002180
grad_step = 000127, loss = 0.002178
grad_step = 000128, loss = 0.002175
grad_step = 000129, loss = 0.002173
grad_step = 000130, loss = 0.002170
grad_step = 000131, loss = 0.002167
grad_step = 000132, loss = 0.002165
grad_step = 000133, loss = 0.002162
grad_step = 000134, loss = 0.002159
grad_step = 000135, loss = 0.002156
grad_step = 000136, loss = 0.002154
grad_step = 000137, loss = 0.002151
grad_step = 000138, loss = 0.002148
grad_step = 000139, loss = 0.002145
grad_step = 000140, loss = 0.002142
grad_step = 000141, loss = 0.002140
grad_step = 000142, loss = 0.002137
grad_step = 000143, loss = 0.002134
grad_step = 000144, loss = 0.002131
grad_step = 000145, loss = 0.002128
grad_step = 000146, loss = 0.002126
grad_step = 000147, loss = 0.002124
grad_step = 000148, loss = 0.002125
grad_step = 000149, loss = 0.002132
grad_step = 000150, loss = 0.002154
grad_step = 000151, loss = 0.002207
grad_step = 000152, loss = 0.002217
grad_step = 000153, loss = 0.002164
grad_step = 000154, loss = 0.002104
grad_step = 000155, loss = 0.002150
grad_step = 000156, loss = 0.002173
grad_step = 000157, loss = 0.002102
grad_step = 000158, loss = 0.002119
grad_step = 000159, loss = 0.002150
grad_step = 000160, loss = 0.002094
grad_step = 000161, loss = 0.002106
grad_step = 000162, loss = 0.002128
grad_step = 000163, loss = 0.002082
grad_step = 000164, loss = 0.002097
grad_step = 000165, loss = 0.002108
grad_step = 000166, loss = 0.002072
grad_step = 000167, loss = 0.002089
grad_step = 000168, loss = 0.002091
grad_step = 000169, loss = 0.002062
grad_step = 000170, loss = 0.002080
grad_step = 000171, loss = 0.002074
grad_step = 000172, loss = 0.002054
grad_step = 000173, loss = 0.002070
grad_step = 000174, loss = 0.002060
grad_step = 000175, loss = 0.002046
grad_step = 000176, loss = 0.002058
grad_step = 000177, loss = 0.002046
grad_step = 000178, loss = 0.002038
grad_step = 000179, loss = 0.002046
grad_step = 000180, loss = 0.002034
grad_step = 000181, loss = 0.002029
grad_step = 000182, loss = 0.002034
grad_step = 000183, loss = 0.002024
grad_step = 000184, loss = 0.002020
grad_step = 000185, loss = 0.002022
grad_step = 000186, loss = 0.002013
grad_step = 000187, loss = 0.002010
grad_step = 000188, loss = 0.002011
grad_step = 000189, loss = 0.002003
grad_step = 000190, loss = 0.002000
grad_step = 000191, loss = 0.001999
grad_step = 000192, loss = 0.001993
grad_step = 000193, loss = 0.001989
grad_step = 000194, loss = 0.001989
grad_step = 000195, loss = 0.001984
grad_step = 000196, loss = 0.001979
grad_step = 000197, loss = 0.001977
grad_step = 000198, loss = 0.001973
grad_step = 000199, loss = 0.001968
grad_step = 000200, loss = 0.001966
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001963
grad_step = 000202, loss = 0.001958
grad_step = 000203, loss = 0.001954
grad_step = 000204, loss = 0.001952
grad_step = 000205, loss = 0.001948
grad_step = 000206, loss = 0.001943
grad_step = 000207, loss = 0.001940
grad_step = 000208, loss = 0.001937
grad_step = 000209, loss = 0.001932
grad_step = 000210, loss = 0.001927
grad_step = 000211, loss = 0.001924
grad_step = 000212, loss = 0.001920
grad_step = 000213, loss = 0.001916
grad_step = 000214, loss = 0.001911
grad_step = 000215, loss = 0.001907
grad_step = 000216, loss = 0.001903
grad_step = 000217, loss = 0.001900
grad_step = 000218, loss = 0.001895
grad_step = 000219, loss = 0.001890
grad_step = 000220, loss = 0.001886
grad_step = 000221, loss = 0.001882
grad_step = 000222, loss = 0.001877
grad_step = 000223, loss = 0.001873
grad_step = 000224, loss = 0.001868
grad_step = 000225, loss = 0.001863
grad_step = 000226, loss = 0.001858
grad_step = 000227, loss = 0.001854
grad_step = 000228, loss = 0.001849
grad_step = 000229, loss = 0.001844
grad_step = 000230, loss = 0.001840
grad_step = 000231, loss = 0.001836
grad_step = 000232, loss = 0.001832
grad_step = 000233, loss = 0.001829
grad_step = 000234, loss = 0.001826
grad_step = 000235, loss = 0.001825
grad_step = 000236, loss = 0.001823
grad_step = 000237, loss = 0.001821
grad_step = 000238, loss = 0.001816
grad_step = 000239, loss = 0.001811
grad_step = 000240, loss = 0.001802
grad_step = 000241, loss = 0.001793
grad_step = 000242, loss = 0.001785
grad_step = 000243, loss = 0.001778
grad_step = 000244, loss = 0.001774
grad_step = 000245, loss = 0.001771
grad_step = 000246, loss = 0.001769
grad_step = 000247, loss = 0.001768
grad_step = 000248, loss = 0.001768
grad_step = 000249, loss = 0.001770
grad_step = 000250, loss = 0.001776
grad_step = 000251, loss = 0.001783
grad_step = 000252, loss = 0.001794
grad_step = 000253, loss = 0.001793
grad_step = 000254, loss = 0.001784
grad_step = 000255, loss = 0.001758
grad_step = 000256, loss = 0.001735
grad_step = 000257, loss = 0.001727
grad_step = 000258, loss = 0.001734
grad_step = 000259, loss = 0.001747
grad_step = 000260, loss = 0.001752
grad_step = 000261, loss = 0.001748
grad_step = 000262, loss = 0.001730
grad_step = 000263, loss = 0.001715
grad_step = 000264, loss = 0.001707
grad_step = 000265, loss = 0.001708
grad_step = 000266, loss = 0.001714
grad_step = 000267, loss = 0.001718
grad_step = 000268, loss = 0.001718
grad_step = 000269, loss = 0.001711
grad_step = 000270, loss = 0.001704
grad_step = 000271, loss = 0.001695
grad_step = 000272, loss = 0.001689
grad_step = 000273, loss = 0.001686
grad_step = 000274, loss = 0.001685
grad_step = 000275, loss = 0.001686
grad_step = 000276, loss = 0.001689
grad_step = 000277, loss = 0.001695
grad_step = 000278, loss = 0.001705
grad_step = 000279, loss = 0.001723
grad_step = 000280, loss = 0.001743
grad_step = 000281, loss = 0.001768
grad_step = 000282, loss = 0.001762
grad_step = 000283, loss = 0.001731
grad_step = 000284, loss = 0.001685
grad_step = 000285, loss = 0.001666
grad_step = 000286, loss = 0.001683
grad_step = 000287, loss = 0.001707
grad_step = 000288, loss = 0.001710
grad_step = 000289, loss = 0.001686
grad_step = 000290, loss = 0.001663
grad_step = 000291, loss = 0.001659
grad_step = 000292, loss = 0.001672
grad_step = 000293, loss = 0.001688
grad_step = 000294, loss = 0.001693
grad_step = 000295, loss = 0.001685
grad_step = 000296, loss = 0.001668
grad_step = 000297, loss = 0.001654
grad_step = 000298, loss = 0.001648
grad_step = 000299, loss = 0.001652
grad_step = 000300, loss = 0.001660
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001664
grad_step = 000302, loss = 0.001664
grad_step = 000303, loss = 0.001657
grad_step = 000304, loss = 0.001648
grad_step = 000305, loss = 0.001641
grad_step = 000306, loss = 0.001639
grad_step = 000307, loss = 0.001640
grad_step = 000308, loss = 0.001644
grad_step = 000309, loss = 0.001648
grad_step = 000310, loss = 0.001650
grad_step = 000311, loss = 0.001653
grad_step = 000312, loss = 0.001653
grad_step = 000313, loss = 0.001653
grad_step = 000314, loss = 0.001651
grad_step = 000315, loss = 0.001649
grad_step = 000316, loss = 0.001645
grad_step = 000317, loss = 0.001640
grad_step = 000318, loss = 0.001635
grad_step = 000319, loss = 0.001631
grad_step = 000320, loss = 0.001627
grad_step = 000321, loss = 0.001623
grad_step = 000322, loss = 0.001621
grad_step = 000323, loss = 0.001620
grad_step = 000324, loss = 0.001618
grad_step = 000325, loss = 0.001617
grad_step = 000326, loss = 0.001616
grad_step = 000327, loss = 0.001615
grad_step = 000328, loss = 0.001615
grad_step = 000329, loss = 0.001615
grad_step = 000330, loss = 0.001617
grad_step = 000331, loss = 0.001626
grad_step = 000332, loss = 0.001651
grad_step = 000333, loss = 0.001711
grad_step = 000334, loss = 0.001839
grad_step = 000335, loss = 0.001991
grad_step = 000336, loss = 0.002036
grad_step = 000337, loss = 0.001770
grad_step = 000338, loss = 0.001613
grad_step = 000339, loss = 0.001777
grad_step = 000340, loss = 0.001818
grad_step = 000341, loss = 0.001639
grad_step = 000342, loss = 0.001680
grad_step = 000343, loss = 0.001771
grad_step = 000344, loss = 0.001661
grad_step = 000345, loss = 0.001639
grad_step = 000346, loss = 0.001733
grad_step = 000347, loss = 0.001645
grad_step = 000348, loss = 0.001625
grad_step = 000349, loss = 0.001679
grad_step = 000350, loss = 0.001646
grad_step = 000351, loss = 0.001603
grad_step = 000352, loss = 0.001647
grad_step = 000353, loss = 0.001636
grad_step = 000354, loss = 0.001596
grad_step = 000355, loss = 0.001621
grad_step = 000356, loss = 0.001633
grad_step = 000357, loss = 0.001600
grad_step = 000358, loss = 0.001597
grad_step = 000359, loss = 0.001620
grad_step = 000360, loss = 0.001606
grad_step = 000361, loss = 0.001588
grad_step = 000362, loss = 0.001597
grad_step = 000363, loss = 0.001605
grad_step = 000364, loss = 0.001591
grad_step = 000365, loss = 0.001584
grad_step = 000366, loss = 0.001593
grad_step = 000367, loss = 0.001593
grad_step = 000368, loss = 0.001582
grad_step = 000369, loss = 0.001581
grad_step = 000370, loss = 0.001587
grad_step = 000371, loss = 0.001583
grad_step = 000372, loss = 0.001576
grad_step = 000373, loss = 0.001576
grad_step = 000374, loss = 0.001580
grad_step = 000375, loss = 0.001576
grad_step = 000376, loss = 0.001571
grad_step = 000377, loss = 0.001571
grad_step = 000378, loss = 0.001573
grad_step = 000379, loss = 0.001572
grad_step = 000380, loss = 0.001568
grad_step = 000381, loss = 0.001566
grad_step = 000382, loss = 0.001566
grad_step = 000383, loss = 0.001567
grad_step = 000384, loss = 0.001565
grad_step = 000385, loss = 0.001562
grad_step = 000386, loss = 0.001560
grad_step = 000387, loss = 0.001560
grad_step = 000388, loss = 0.001560
grad_step = 000389, loss = 0.001559
grad_step = 000390, loss = 0.001556
grad_step = 000391, loss = 0.001554
grad_step = 000392, loss = 0.001554
grad_step = 000393, loss = 0.001553
grad_step = 000394, loss = 0.001552
grad_step = 000395, loss = 0.001550
grad_step = 000396, loss = 0.001548
grad_step = 000397, loss = 0.001547
grad_step = 000398, loss = 0.001546
grad_step = 000399, loss = 0.001545
grad_step = 000400, loss = 0.001544
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001543
grad_step = 000402, loss = 0.001541
grad_step = 000403, loss = 0.001539
grad_step = 000404, loss = 0.001538
grad_step = 000405, loss = 0.001537
grad_step = 000406, loss = 0.001535
grad_step = 000407, loss = 0.001534
grad_step = 000408, loss = 0.001533
grad_step = 000409, loss = 0.001532
grad_step = 000410, loss = 0.001531
grad_step = 000411, loss = 0.001529
grad_step = 000412, loss = 0.001528
grad_step = 000413, loss = 0.001527
grad_step = 000414, loss = 0.001525
grad_step = 000415, loss = 0.001524
grad_step = 000416, loss = 0.001523
grad_step = 000417, loss = 0.001522
grad_step = 000418, loss = 0.001521
grad_step = 000419, loss = 0.001520
grad_step = 000420, loss = 0.001519
grad_step = 000421, loss = 0.001519
grad_step = 000422, loss = 0.001519
grad_step = 000423, loss = 0.001521
grad_step = 000424, loss = 0.001525
grad_step = 000425, loss = 0.001533
grad_step = 000426, loss = 0.001544
grad_step = 000427, loss = 0.001561
grad_step = 000428, loss = 0.001581
grad_step = 000429, loss = 0.001605
grad_step = 000430, loss = 0.001612
grad_step = 000431, loss = 0.001592
grad_step = 000432, loss = 0.001550
grad_step = 000433, loss = 0.001513
grad_step = 000434, loss = 0.001503
grad_step = 000435, loss = 0.001520
grad_step = 000436, loss = 0.001541
grad_step = 000437, loss = 0.001546
grad_step = 000438, loss = 0.001530
grad_step = 000439, loss = 0.001507
grad_step = 000440, loss = 0.001495
grad_step = 000441, loss = 0.001500
grad_step = 000442, loss = 0.001512
grad_step = 000443, loss = 0.001518
grad_step = 000444, loss = 0.001514
grad_step = 000445, loss = 0.001502
grad_step = 000446, loss = 0.001491
grad_step = 000447, loss = 0.001486
grad_step = 000448, loss = 0.001488
grad_step = 000449, loss = 0.001493
grad_step = 000450, loss = 0.001498
grad_step = 000451, loss = 0.001501
grad_step = 000452, loss = 0.001502
grad_step = 000453, loss = 0.001500
grad_step = 000454, loss = 0.001497
grad_step = 000455, loss = 0.001492
grad_step = 000456, loss = 0.001486
grad_step = 000457, loss = 0.001480
grad_step = 000458, loss = 0.001475
grad_step = 000459, loss = 0.001473
grad_step = 000460, loss = 0.001472
grad_step = 000461, loss = 0.001472
grad_step = 000462, loss = 0.001474
grad_step = 000463, loss = 0.001475
grad_step = 000464, loss = 0.001477
grad_step = 000465, loss = 0.001480
grad_step = 000466, loss = 0.001483
grad_step = 000467, loss = 0.001487
grad_step = 000468, loss = 0.001492
grad_step = 000469, loss = 0.001497
grad_step = 000470, loss = 0.001502
grad_step = 000471, loss = 0.001504
grad_step = 000472, loss = 0.001503
grad_step = 000473, loss = 0.001497
grad_step = 000474, loss = 0.001485
grad_step = 000475, loss = 0.001472
grad_step = 000476, loss = 0.001460
grad_step = 000477, loss = 0.001453
grad_step = 000478, loss = 0.001452
grad_step = 000479, loss = 0.001454
grad_step = 000480, loss = 0.001459
grad_step = 000481, loss = 0.001465
grad_step = 000482, loss = 0.001470
grad_step = 000483, loss = 0.001475
grad_step = 000484, loss = 0.001479
grad_step = 000485, loss = 0.001481
grad_step = 000486, loss = 0.001479
grad_step = 000487, loss = 0.001473
grad_step = 000488, loss = 0.001463
grad_step = 000489, loss = 0.001452
grad_step = 000490, loss = 0.001443
grad_step = 000491, loss = 0.001437
grad_step = 000492, loss = 0.001436
grad_step = 000493, loss = 0.001438
grad_step = 000494, loss = 0.001441
grad_step = 000495, loss = 0.001446
grad_step = 000496, loss = 0.001452
grad_step = 000497, loss = 0.001459
grad_step = 000498, loss = 0.001467
grad_step = 000499, loss = 0.001471
grad_step = 000500, loss = 0.001474
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001470
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

  date_run                              2020-05-10 11:13:22.370897
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.274099
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 11:13:22.377345
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.177047
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 11:13:22.383892
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.164318
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 11:13:22.388961
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.69029
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
0   2020-05-10 11:12:55.467531  ...    mean_absolute_error
1   2020-05-10 11:12:55.471204  ...     mean_squared_error
2   2020-05-10 11:12:55.474256  ...  median_absolute_error
3   2020-05-10 11:12:55.477386  ...               r2_score
4   2020-05-10 11:13:03.740438  ...    mean_absolute_error
5   2020-05-10 11:13:03.744198  ...     mean_squared_error
6   2020-05-10 11:13:03.747594  ...  median_absolute_error
7   2020-05-10 11:13:03.750757  ...               r2_score
8   2020-05-10 11:13:22.370897  ...    mean_absolute_error
9   2020-05-10 11:13:22.377345  ...     mean_squared_error
10  2020-05-10 11:13:22.383892  ...  median_absolute_error
11  2020-05-10 11:13:22.388961  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 43%|████▎     | 4300800/9912422 [00:00<00:00, 42983035.98it/s]9920512it [00:00, 37555964.97it/s]                             
0it [00:00, ?it/s]32768it [00:00, 577404.24it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 447627.51it/s]1654784it [00:00, 11315481.07it/s]                         
0it [00:00, ?it/s]8192it [00:00, 152095.48it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f862da780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f23a1ec88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f86291e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f23a1ee80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f86291e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f38c8ccc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f862daf98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f38c9dda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f862daf98> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f38c9deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f86291e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5dd883c208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=149b964d3b67ea2b3e2a9977ef40927d87e95a31949b99b55d73c54b54a203b0
  Stored in directory: /tmp/pip-ephem-wheel-cache-pnu46wiu/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5dd06bdfd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2891776/17464789 [===>..........................] - ETA: 0s
11313152/17464789 [==================>...........] - ETA: 0s
16326656/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 11:14:46.700865: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 11:14:46.705563: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 11:14:46.705688: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558add16a540 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 11:14:46.705700: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6710 - accuracy: 0.4997
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6551 - accuracy: 0.5008
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6496 - accuracy: 0.5011
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7096 - accuracy: 0.4972
11000/25000 [============>.................] - ETA: 3s - loss: 7.7015 - accuracy: 0.4977
12000/25000 [=============>................] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6784 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6633 - accuracy: 0.5002
15000/25000 [=================>............] - ETA: 2s - loss: 7.6697 - accuracy: 0.4998
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6693 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 7s 263us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 11:14:59.429678
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 11:14:59.429678  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 11:15:05.071516: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 11:15:05.076877: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 11:15:05.077447: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b49d0ecb60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 11:15:05.077570: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f39c0aebd30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1068 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.0177 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f39b5e93f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 8.1573 - accuracy: 0.4680
 2000/25000 [=>............................] - ETA: 7s - loss: 7.9580 - accuracy: 0.4810 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7801 - accuracy: 0.4926
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8200 - accuracy: 0.4900
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7783 - accuracy: 0.4927
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7510 - accuracy: 0.4945
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7382 - accuracy: 0.4953
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6611 - accuracy: 0.5004
15000/25000 [=================>............] - ETA: 2s - loss: 7.6871 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6602 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6551 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 271us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f3971c09048> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:32:44, 10.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:00:58, 15.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:15:54, 21.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:53:29, 30.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.24M/862M [00:01<5:30:43, 43.3kB/s].vector_cache/glove.6B.zip:   1%|          | 6.59M/862M [00:01<3:50:45, 61.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:01<2:40:41, 88.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<1:52:07, 126kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.3M/862M [00:01<1:18:05, 180kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:01<54:32, 256kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:01<38:03, 365kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:02<26:36, 519kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.6M/862M [00:02<18:37, 738kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:02<13:04, 1.05MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<09:11, 1.48MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<06:30, 2.08MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<05:04, 2.67MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<05:27, 2.47MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:42, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<04:20, 3.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:04<03:17, 4.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<08:04, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<05:20, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<04:27, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:08<10:20:33, 21.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<7:14:39, 30.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:08<5:03:31, 43.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<3:39:14, 60.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<2:36:43, 84.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<1:50:26, 120kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:10<1:17:13, 171kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<1:02:14, 212kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<45:07, 292kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<31:53, 412kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<24:57, 525kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<20:31, 638kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<15:03, 869kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.7M/862M [00:14<10:42, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<16:28, 791kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<13:09, 990kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<09:30, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:16<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<20:00, 648kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<17:10, 754kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<12:41, 1.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:18<09:01, 1.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<12:00, 1.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<09:56, 1.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<07:16, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<07:45, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<08:35, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<06:41, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<04:54, 2.60MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:07, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<06:27, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<04:53, 2.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:03, 2.09MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:12, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:48, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:12, 3.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<11:23, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:28, 1.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:56, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<05:00, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<31:22, 400kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<25:01, 501kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<18:09, 690kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<12:49, 974kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<16:22, 762kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:56, 963kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:25, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:08, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:26, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:21, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:17, 2.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:57, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:48, 1.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:13, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:35, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:46, 1.81MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:02, 2.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:03, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:04, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:35, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<04:03, 2.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:03, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:47, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:48, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:32, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:31, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:53, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:17, 2.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:45, 1.54MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:53, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:10, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:05, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:40, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:19, 2.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:27, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:43, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:18, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<03:51, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:03, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:06, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:16, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:49, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<2:22:18, 82.3kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:40:59, 116kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<1:10:54, 165kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<51:51, 225kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<39:00, 298kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<27:52, 417kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<19:37, 591kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<18:12, 636kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<14:06, 819kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:09, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:28, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:00, 1.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:56, 1.93MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:30, 1.76MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:13, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:45, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:09, 2.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<10:30, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:43, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:24, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:32, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:50, 1.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:16, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:26, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:48, 1.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:23, 2.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:25, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:24, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:03, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:40, 3.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:19, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:27, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:50, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:39, 1.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<06:38, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:13, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:48, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:51, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:08, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:34, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:26, 2.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:07, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:50, 2.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:55, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:58, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:46, 2.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:27, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:09, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:59, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:12, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:50, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:41, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:13, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:46, 2.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:41, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<07:23, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:28, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:00, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:39, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:18, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:49, 2.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:49, 1.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<08:06, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:59, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:18, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:58, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:54, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:58, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:44, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:56, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:42, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:30, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:05, 2.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:42, 2.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:34, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:50, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:21, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:57, 1.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:23, 2.96MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:54, 1.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:26, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:29, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:53, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:36, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:13, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:47, 2.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<09:43, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:59, 1.24MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<05:52, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:07, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:28, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:03, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:39, 2.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<17:54, 545kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<13:41, 713kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<09:51, 988kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<08:52, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<08:36, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:36, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:44, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:12, 943kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:17, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:04, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:12, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:42, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:15, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:48, 2.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<09:32, 995kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:48, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:44, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:56, 1.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:22, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:01, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:38, 2.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<09:19, 1.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:33, 1.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<05:29, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:52, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:13, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:55, 2.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:38, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:30, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:08, 2.92MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:52, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:56, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:25, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:57, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:35, 1.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:23, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:12, 2.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:51, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:34, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:49, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:12, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:33, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:22, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<03:09, 2.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<08:19, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<6:26:49, 22.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<4:30:57, 32.7kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<3:08:50, 46.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<2:17:28, 64.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<1:38:14, 89.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:09:08, 127kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<48:19, 181kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<37:13, 234kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<27:05, 322kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<19:08, 454kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<15:07, 572kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:30, 692kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<09:12, 938kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<06:31, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<18:04, 475kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<13:40, 628kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<09:48, 873kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<08:35, 992kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:59, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:02, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<04:18, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:01, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:37, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:50, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:07, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:39, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:22, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:12, 2.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:37, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:13, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:10, 2.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:54, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:45, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:45, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:43, 3.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:44, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:59, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:43, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:52, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:50, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:47, 2.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:23, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:18, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:43, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:42, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:42, 2.92MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:29, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:07, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:07, 2.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:47, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:28, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:36, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<02:36, 2.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:11, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:16, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:52, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:17, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:52, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:45, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:11, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:10, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:49, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:19, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:54, 1.54MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:52, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:48, 2.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<07:18, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:02, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:24, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:37, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:48, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:44, 1.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<02:41, 2.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:21, 997kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:58, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:22, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:38, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:02, 1.44MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:53, 1.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:50, 2.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:18, 1.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:52, 1.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:54, 2.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:30, 2.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:12, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:23, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:26, 2.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<06:07, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:07, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:45, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:04, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:28, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:29, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:31, 2.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:54, 1.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:15, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:10, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:37, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:09, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:19, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:24, 2.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<06:36, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:25, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:59, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:09, 1.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:28, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:31, 2.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<06:15, 1.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:07, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:43, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:02, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:26, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:30, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:31, 2.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:14, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:08, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:46, 1.71MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:57, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:20, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:25, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:28, 2.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:58, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:56, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:36, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:48, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:08, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:13, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:20, 2.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:59, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:32, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:38, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:06, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:54, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:11, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:46, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:26, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:46, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:00, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:25, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:27, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:16, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:35, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:03, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:39<02:17, 2.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:41, 3.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<08:10, 720kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<07:09, 822kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:18, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<03:45, 1.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<06:04, 956kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:56, 1.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:36, 1.60MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:41, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:55, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:02, 1.89MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:11, 2.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:07, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:33, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:59, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:27, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:42, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:57, 2.83MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:33, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:09, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:21, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:45, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:12, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:49, 2.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:00, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:27, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:32, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:52, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<01:58, 2.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:27, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:01, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:22, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<01:43, 3.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:04, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:47, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:04, 2.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:30, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:56, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:21, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:42, 2.95MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:51, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:00, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:55, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:03, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:21, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:37, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:52, 2.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:20, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:37, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:40, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:51, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:10, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:30, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:48, 2.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:31, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:44, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:44, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:52, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:06, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:25, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<01:44, 2.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:27, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:58, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:11, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:27, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:50, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:12, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:36, 2.81MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:48, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:30, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:52, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:13, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:35, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:04, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:29, 2.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:52, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:13, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:32, 1.69MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:46, 1.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:09, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<01:33, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:28, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:55, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:10, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:43, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<3:10:55, 21.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<2:13:32, 31.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<1:32:31, 44.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<1:07:10, 60.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<47:56, 85.0kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<33:43, 121kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<23:24, 172kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<19:02, 211kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<13:47, 290kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<09:42, 411kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<07:32, 523kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<06:14, 632kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:33, 861kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:14, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:28, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:53, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:06, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:14, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:30, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:58, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<01:25, 2.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:42, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:59, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:10, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:18, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:28, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:57, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:23, 2.58MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<03:13, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:40, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:57, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:05, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:19, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:49, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:19, 2.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:23, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:45, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:00, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:08, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:53, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:25, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:40, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:56, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:33, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:07, 2.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:48, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:21, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:43, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:50, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:04, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:36, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:09, 2.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:16, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:57, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:27, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:38, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:30, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:08, 2.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:24, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:20, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:00, 2.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:17, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:34, 1.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:16, 2.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:55, 3.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:24, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:01, 1.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:28, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:35, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:47, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:23, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:00, 2.71MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:39, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:28, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:06, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:18, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:33, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:13, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:52, 2.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:21, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:57, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:26, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:30, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:38, 1.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:17, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:55, 2.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:15, 1.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:51, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:21, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:24, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:33, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:13, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:52, 2.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:14, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:50, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:19, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:56, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<04:03, 544kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:05, 710kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:12, 988kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:57, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:51, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:25, 1.49MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<01:00, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:04, 993kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:41, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:15, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:07, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:50, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:58, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:08, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:39, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:05, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:59, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:43, 2.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:52, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:02, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:50, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:35, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:32, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:16, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:55, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:58, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:01, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:48, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:34, 2.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<02:48, 564kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:08, 733kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:31, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:21, 1.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:17, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:59, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:41, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:23, 1.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:08, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:49, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:50, 1.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:54, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:42, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:29, 2.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:05, 1.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:54, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:40, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:42, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:47, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:37, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:25, 2.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:54, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:46, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:34, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:36, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:40, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:32, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:22, 2.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:59, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:48, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:35, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:38, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:29, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:20, 2.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:36, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:31, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:16, 2.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:43, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:35, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:16, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:18, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:12, 2.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:14, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:18, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:14, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:09, 3.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:20, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:12, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 2.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:12, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 2.95MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.80MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 2.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:25<00:01, 3.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.51MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.27MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1046/400000 [00:00<00:38, 10455.45it/s]  1%|          | 2134/400000 [00:00<00:37, 10579.29it/s]  1%|          | 3191/400000 [00:00<00:37, 10574.12it/s]  1%|          | 4291/400000 [00:00<00:36, 10696.04it/s]  1%|▏         | 5399/400000 [00:00<00:36, 10805.93it/s]  2%|▏         | 6472/400000 [00:00<00:36, 10782.12it/s]  2%|▏         | 7514/400000 [00:00<00:36, 10669.55it/s]  2%|▏         | 8505/400000 [00:00<00:37, 10428.95it/s]  2%|▏         | 9495/400000 [00:00<00:38, 10261.81it/s]  3%|▎         | 10500/400000 [00:01<00:38, 10196.51it/s]  3%|▎         | 11523/400000 [00:01<00:38, 10204.68it/s]  3%|▎         | 12524/400000 [00:01<00:38, 10082.87it/s]  3%|▎         | 13563/400000 [00:01<00:37, 10172.05it/s]  4%|▎         | 14572/400000 [00:01<00:37, 10145.60it/s]  4%|▍         | 15622/400000 [00:01<00:37, 10248.67it/s]  4%|▍         | 16675/400000 [00:01<00:37, 10331.18it/s]  4%|▍         | 17806/400000 [00:01<00:36, 10605.24it/s]  5%|▍         | 18867/400000 [00:01<00:36, 10541.24it/s]  5%|▍         | 19922/400000 [00:01<00:36, 10383.45it/s]  5%|▌         | 20961/400000 [00:02<00:36, 10368.76it/s]  5%|▌         | 21999/400000 [00:02<00:37, 10128.82it/s]  6%|▌         | 23142/400000 [00:02<00:35, 10486.69it/s]  6%|▌         | 24195/400000 [00:02<00:35, 10492.27it/s]  6%|▋         | 25283/400000 [00:02<00:35, 10602.64it/s]  7%|▋         | 26346/400000 [00:02<00:35, 10472.80it/s]  7%|▋         | 27396/400000 [00:02<00:35, 10470.80it/s]  7%|▋         | 28453/400000 [00:02<00:35, 10498.64it/s]  7%|▋         | 29504/400000 [00:02<00:35, 10424.10it/s]  8%|▊         | 30614/400000 [00:02<00:34, 10614.34it/s]  8%|▊         | 31677/400000 [00:03<00:35, 10291.86it/s]  8%|▊         | 32780/400000 [00:03<00:34, 10502.57it/s]  8%|▊         | 33834/400000 [00:03<00:34, 10498.80it/s]  9%|▊         | 34887/400000 [00:03<00:34, 10502.88it/s]  9%|▉         | 35983/400000 [00:03<00:34, 10635.12it/s]  9%|▉         | 37081/400000 [00:03<00:33, 10735.19it/s] 10%|▉         | 38156/400000 [00:03<00:34, 10598.63it/s] 10%|▉         | 39218/400000 [00:03<00:34, 10479.42it/s] 10%|█         | 40268/400000 [00:03<00:34, 10391.30it/s] 10%|█         | 41339/400000 [00:03<00:34, 10481.92it/s] 11%|█         | 42389/400000 [00:04<00:34, 10457.94it/s] 11%|█         | 43436/400000 [00:04<00:35, 10187.36it/s] 11%|█         | 44468/400000 [00:04<00:34, 10225.37it/s] 11%|█▏        | 45492/400000 [00:04<00:34, 10177.38it/s] 12%|█▏        | 46529/400000 [00:04<00:34, 10233.77it/s] 12%|█▏        | 47606/400000 [00:04<00:33, 10388.04it/s] 12%|█▏        | 48704/400000 [00:04<00:33, 10558.43it/s] 12%|█▏        | 49792/400000 [00:04<00:32, 10651.56it/s] 13%|█▎        | 50859/400000 [00:04<00:33, 10465.30it/s] 13%|█▎        | 51917/400000 [00:04<00:33, 10498.09it/s] 13%|█▎        | 52989/400000 [00:05<00:32, 10562.20it/s] 14%|█▎        | 54047/400000 [00:05<00:33, 10430.60it/s] 14%|█▍        | 55092/400000 [00:05<00:33, 10249.30it/s] 14%|█▍        | 56141/400000 [00:05<00:33, 10318.25it/s] 14%|█▍        | 57251/400000 [00:05<00:32, 10539.32it/s] 15%|█▍        | 58307/400000 [00:05<00:32, 10377.38it/s] 15%|█▍        | 59347/400000 [00:05<00:34, 10015.69it/s] 15%|█▌        | 60379/400000 [00:05<00:33, 10102.90it/s] 15%|█▌        | 61393/400000 [00:05<00:33, 9993.57it/s]  16%|█▌        | 62468/400000 [00:05<00:33, 10206.80it/s] 16%|█▌        | 63561/400000 [00:06<00:32, 10413.17it/s] 16%|█▌        | 64631/400000 [00:06<00:31, 10496.80it/s] 16%|█▋        | 65683/400000 [00:06<00:32, 10327.48it/s] 17%|█▋        | 66721/400000 [00:06<00:32, 10341.34it/s] 17%|█▋        | 67815/400000 [00:06<00:31, 10512.35it/s] 17%|█▋        | 68885/400000 [00:06<00:31, 10567.34it/s] 17%|█▋        | 69975/400000 [00:06<00:30, 10663.76it/s] 18%|█▊        | 71043/400000 [00:06<00:31, 10556.82it/s] 18%|█▊        | 72124/400000 [00:06<00:30, 10630.20it/s] 18%|█▊        | 73188/400000 [00:07<00:30, 10610.14it/s] 19%|█▊        | 74250/400000 [00:07<00:31, 10349.40it/s] 19%|█▉        | 75304/400000 [00:07<00:31, 10404.42it/s] 19%|█▉        | 76431/400000 [00:07<00:30, 10648.58it/s] 19%|█▉        | 77499/400000 [00:07<00:30, 10606.11it/s] 20%|█▉        | 78562/400000 [00:07<00:31, 10337.62it/s] 20%|█▉        | 79599/400000 [00:07<00:31, 10331.21it/s] 20%|██        | 80634/400000 [00:07<00:31, 10194.42it/s] 20%|██        | 81726/400000 [00:07<00:30, 10401.23it/s] 21%|██        | 82815/400000 [00:07<00:30, 10540.93it/s] 21%|██        | 83885/400000 [00:08<00:29, 10585.56it/s] 21%|██        | 84946/400000 [00:08<00:30, 10492.01it/s] 22%|██▏       | 86061/400000 [00:08<00:29, 10680.76it/s] 22%|██▏       | 87135/400000 [00:08<00:29, 10698.05it/s] 22%|██▏       | 88206/400000 [00:08<00:29, 10658.20it/s] 22%|██▏       | 89273/400000 [00:08<00:29, 10609.69it/s] 23%|██▎       | 90364/400000 [00:08<00:28, 10696.01it/s] 23%|██▎       | 91448/400000 [00:08<00:28, 10735.73it/s] 23%|██▎       | 92523/400000 [00:08<00:29, 10558.84it/s] 23%|██▎       | 93580/400000 [00:08<00:29, 10451.39it/s] 24%|██▎       | 94627/400000 [00:09<00:29, 10302.91it/s] 24%|██▍       | 95659/400000 [00:09<00:29, 10273.26it/s] 24%|██▍       | 96688/400000 [00:09<00:31, 9592.39it/s]  24%|██▍       | 97713/400000 [00:09<00:30, 9779.53it/s] 25%|██▍       | 98740/400000 [00:09<00:30, 9920.40it/s] 25%|██▍       | 99832/400000 [00:09<00:29, 10199.40it/s] 25%|██▌       | 100941/400000 [00:09<00:28, 10450.26it/s] 26%|██▌       | 102056/400000 [00:09<00:27, 10649.61it/s] 26%|██▌       | 103126/400000 [00:09<00:27, 10608.68it/s] 26%|██▌       | 104197/400000 [00:09<00:27, 10636.40it/s] 26%|██▋       | 105287/400000 [00:10<00:27, 10709.96it/s] 27%|██▋       | 106366/400000 [00:10<00:27, 10732.45it/s] 27%|██▋       | 107456/400000 [00:10<00:27, 10780.08it/s] 27%|██▋       | 108551/400000 [00:10<00:26, 10828.30it/s] 27%|██▋       | 109635/400000 [00:10<00:27, 10539.05it/s] 28%|██▊       | 110702/400000 [00:10<00:27, 10575.91it/s] 28%|██▊       | 111781/400000 [00:10<00:27, 10639.01it/s] 28%|██▊       | 112852/400000 [00:10<00:26, 10658.55it/s] 28%|██▊       | 113919/400000 [00:10<00:27, 10240.37it/s] 29%|██▊       | 114948/400000 [00:11<00:28, 9994.74it/s]  29%|██▉       | 115988/400000 [00:11<00:28, 10109.04it/s] 29%|██▉       | 117003/400000 [00:11<00:27, 10117.61it/s] 30%|██▉       | 118076/400000 [00:11<00:27, 10291.74it/s] 30%|██▉       | 119165/400000 [00:11<00:26, 10462.27it/s] 30%|███       | 120254/400000 [00:11<00:26, 10585.76it/s] 30%|███       | 121380/400000 [00:11<00:25, 10777.67it/s] 31%|███       | 122460/400000 [00:11<00:26, 10466.30it/s] 31%|███       | 123511/400000 [00:11<00:26, 10401.34it/s] 31%|███       | 124554/400000 [00:11<00:26, 10215.53it/s] 31%|███▏      | 125579/400000 [00:12<00:27, 9858.34it/s]  32%|███▏      | 126570/400000 [00:12<00:28, 9635.79it/s] 32%|███▏      | 127538/400000 [00:12<00:28, 9417.24it/s] 32%|███▏      | 128584/400000 [00:12<00:27, 9706.67it/s] 32%|███▏      | 129631/400000 [00:12<00:27, 9923.08it/s] 33%|███▎      | 130629/400000 [00:12<00:27, 9913.73it/s] 33%|███▎      | 131624/400000 [00:12<00:27, 9865.88it/s] 33%|███▎      | 132708/400000 [00:12<00:26, 10138.39it/s] 33%|███▎      | 133813/400000 [00:12<00:25, 10394.53it/s] 34%|███▎      | 134857/400000 [00:12<00:26, 10174.97it/s] 34%|███▍      | 135906/400000 [00:13<00:25, 10265.68it/s] 34%|███▍      | 136973/400000 [00:13<00:25, 10381.75it/s] 35%|███▍      | 138064/400000 [00:13<00:24, 10532.32it/s] 35%|███▍      | 139139/400000 [00:13<00:24, 10594.31it/s] 35%|███▌      | 140235/400000 [00:13<00:24, 10698.20it/s] 35%|███▌      | 141307/400000 [00:13<00:24, 10644.90it/s] 36%|███▌      | 142373/400000 [00:13<00:24, 10620.52it/s] 36%|███▌      | 143436/400000 [00:13<00:24, 10599.61it/s] 36%|███▌      | 144572/400000 [00:13<00:23, 10815.00it/s] 36%|███▋      | 145655/400000 [00:13<00:23, 10752.73it/s] 37%|███▋      | 146732/400000 [00:14<00:23, 10613.37it/s] 37%|███▋      | 147795/400000 [00:14<00:24, 10416.42it/s] 37%|███▋      | 148839/400000 [00:14<00:24, 10338.67it/s] 37%|███▋      | 149875/400000 [00:14<00:24, 10044.33it/s] 38%|███▊      | 150883/400000 [00:14<00:25, 9914.60it/s]  38%|███▊      | 152004/400000 [00:14<00:24, 10268.26it/s] 38%|███▊      | 153143/400000 [00:14<00:23, 10578.59it/s] 39%|███▊      | 154235/400000 [00:14<00:23, 10678.03it/s] 39%|███▉      | 155308/400000 [00:14<00:22, 10672.28it/s] 39%|███▉      | 156379/400000 [00:15<00:23, 10390.50it/s] 39%|███▉      | 157422/400000 [00:15<00:23, 10130.90it/s] 40%|███▉      | 158440/400000 [00:15<00:25, 9560.73it/s]  40%|███▉      | 159484/400000 [00:15<00:24, 9807.73it/s] 40%|████      | 160473/400000 [00:15<00:24, 9775.63it/s] 40%|████      | 161466/400000 [00:15<00:24, 9821.30it/s] 41%|████      | 162522/400000 [00:15<00:23, 10030.71it/s] 41%|████      | 163683/400000 [00:15<00:22, 10456.24it/s] 41%|████      | 164782/400000 [00:15<00:22, 10609.74it/s] 41%|████▏     | 165860/400000 [00:15<00:21, 10658.44it/s] 42%|████▏     | 166930/400000 [00:16<00:22, 10241.64it/s] 42%|████▏     | 167961/400000 [00:16<00:23, 10004.85it/s] 42%|████▏     | 169110/400000 [00:16<00:22, 10405.58it/s] 43%|████▎     | 170159/400000 [00:16<00:22, 10371.38it/s] 43%|████▎     | 171202/400000 [00:16<00:22, 10377.76it/s] 43%|████▎     | 172249/400000 [00:16<00:21, 10403.11it/s] 43%|████▎     | 173293/400000 [00:16<00:22, 10126.48it/s] 44%|████▎     | 174402/400000 [00:16<00:21, 10397.47it/s] 44%|████▍     | 175523/400000 [00:16<00:21, 10626.85it/s] 44%|████▍     | 176638/400000 [00:16<00:20, 10776.17it/s] 44%|████▍     | 177720/400000 [00:17<00:20, 10640.95it/s] 45%|████▍     | 178787/400000 [00:17<00:21, 10508.94it/s] 45%|████▍     | 179841/400000 [00:17<00:21, 10368.66it/s] 45%|████▌     | 180880/400000 [00:17<00:21, 10339.36it/s] 45%|████▌     | 181916/400000 [00:17<00:22, 9841.47it/s]  46%|████▌     | 182907/400000 [00:17<00:22, 9728.87it/s] 46%|████▌     | 183944/400000 [00:17<00:21, 9912.45it/s] 46%|████▌     | 184952/400000 [00:17<00:21, 9961.69it/s] 46%|████▋     | 185967/400000 [00:17<00:21, 10016.97it/s] 47%|████▋     | 187054/400000 [00:18<00:20, 10255.45it/s] 47%|████▋     | 188083/400000 [00:18<00:20, 10168.08it/s] 47%|████▋     | 189150/400000 [00:18<00:20, 10313.27it/s] 48%|████▊     | 190190/400000 [00:18<00:20, 10337.01it/s] 48%|████▊     | 191226/400000 [00:18<00:20, 10150.09it/s] 48%|████▊     | 192253/400000 [00:18<00:20, 10183.42it/s] 48%|████▊     | 193273/400000 [00:18<00:20, 9975.08it/s]  49%|████▊     | 194294/400000 [00:18<00:20, 10044.34it/s] 49%|████▉     | 195300/400000 [00:18<00:20, 9945.00it/s]  49%|████▉     | 196296/400000 [00:18<00:20, 9717.44it/s] 49%|████▉     | 197270/400000 [00:19<00:20, 9697.18it/s] 50%|████▉     | 198310/400000 [00:19<00:20, 9896.08it/s] 50%|████▉     | 199389/400000 [00:19<00:19, 10146.87it/s] 50%|█████     | 200407/400000 [00:19<00:20, 9939.14it/s]  50%|█████     | 201452/400000 [00:19<00:19, 10084.81it/s] 51%|█████     | 202468/400000 [00:19<00:19, 10105.49it/s] 51%|█████     | 203502/400000 [00:19<00:19, 10171.84it/s] 51%|█████     | 204542/400000 [00:19<00:19, 10238.67it/s] 51%|█████▏    | 205610/400000 [00:19<00:18, 10364.82it/s] 52%|█████▏    | 206648/400000 [00:19<00:18, 10365.09it/s] 52%|█████▏    | 207686/400000 [00:20<00:18, 10331.91it/s] 52%|█████▏    | 208720/400000 [00:20<00:18, 10306.04it/s] 52%|█████▏    | 209752/400000 [00:20<00:18, 10173.25it/s] 53%|█████▎    | 210778/400000 [00:20<00:18, 10198.53it/s] 53%|█████▎    | 211824/400000 [00:20<00:18, 10274.05it/s] 53%|█████▎    | 212852/400000 [00:20<00:18, 10203.13it/s] 53%|█████▎    | 213873/400000 [00:20<00:18, 9911.04it/s]  54%|█████▎    | 214876/400000 [00:20<00:18, 9944.20it/s] 54%|█████▍    | 215985/400000 [00:20<00:17, 10260.34it/s] 54%|█████▍    | 217015/400000 [00:20<00:17, 10218.25it/s] 55%|█████▍    | 218040/400000 [00:21<00:17, 10225.71it/s] 55%|█████▍    | 219065/400000 [00:21<00:17, 10080.50it/s] 55%|█████▌    | 220075/400000 [00:21<00:18, 9933.89it/s]  55%|█████▌    | 221071/400000 [00:21<00:18, 9895.22it/s] 56%|█████▌    | 222062/400000 [00:21<00:18, 9883.62it/s] 56%|█████▌    | 223090/400000 [00:21<00:17, 9998.12it/s] 56%|█████▌    | 224135/400000 [00:21<00:17, 10129.23it/s] 56%|█████▋    | 225232/400000 [00:21<00:16, 10365.71it/s] 57%|█████▋    | 226290/400000 [00:21<00:16, 10428.59it/s] 57%|█████▋    | 227353/400000 [00:21<00:16, 10485.63it/s] 57%|█████▋    | 228403/400000 [00:22<00:16, 10473.63it/s] 57%|█████▋    | 229452/400000 [00:22<00:16, 10241.77it/s] 58%|█████▊    | 230574/400000 [00:22<00:16, 10515.17it/s] 58%|█████▊    | 231648/400000 [00:22<00:15, 10579.06it/s] 58%|█████▊    | 232739/400000 [00:22<00:15, 10674.67it/s] 58%|█████▊    | 233809/400000 [00:22<00:15, 10417.95it/s] 59%|█████▊    | 234854/400000 [00:22<00:16, 9974.89it/s]  59%|█████▉    | 235858/400000 [00:22<00:16, 9956.06it/s] 59%|█████▉    | 236858/400000 [00:22<00:16, 9907.51it/s] 59%|█████▉    | 237852/400000 [00:23<00:16, 9787.63it/s] 60%|█████▉    | 238834/400000 [00:23<00:16, 9539.07it/s] 60%|█████▉    | 239791/400000 [00:23<00:16, 9455.91it/s] 60%|██████    | 240739/400000 [00:23<00:17, 9365.18it/s] 60%|██████    | 241678/400000 [00:23<00:16, 9333.15it/s] 61%|██████    | 242613/400000 [00:23<00:16, 9337.50it/s] 61%|██████    | 243548/400000 [00:23<00:17, 9152.48it/s] 61%|██████    | 244500/400000 [00:23<00:16, 9257.87it/s] 61%|██████▏   | 245493/400000 [00:23<00:16, 9448.73it/s] 62%|██████▏   | 246498/400000 [00:23<00:15, 9620.51it/s] 62%|██████▏   | 247478/400000 [00:24<00:15, 9673.13it/s] 62%|██████▏   | 248447/400000 [00:24<00:15, 9562.18it/s] 62%|██████▏   | 249405/400000 [00:24<00:16, 9199.42it/s] 63%|██████▎   | 250367/400000 [00:24<00:16, 9320.46it/s] 63%|██████▎   | 251303/400000 [00:24<00:16, 9209.88it/s] 63%|██████▎   | 252265/400000 [00:24<00:15, 9326.43it/s] 63%|██████▎   | 253200/400000 [00:24<00:16, 9150.37it/s] 64%|██████▎   | 254118/400000 [00:24<00:15, 9141.29it/s] 64%|██████▍   | 255107/400000 [00:24<00:15, 9352.37it/s] 64%|██████▍   | 256087/400000 [00:24<00:15, 9480.29it/s] 64%|██████▍   | 257038/400000 [00:25<00:15, 9280.81it/s] 65%|██████▍   | 258011/400000 [00:25<00:15, 9408.85it/s] 65%|██████▍   | 258954/400000 [00:25<00:15, 9315.38it/s] 65%|██████▍   | 259905/400000 [00:25<00:14, 9370.30it/s] 65%|██████▌   | 260844/400000 [00:25<00:15, 8850.41it/s] 65%|██████▌   | 261736/400000 [00:25<00:16, 8571.46it/s] 66%|██████▌   | 262600/400000 [00:25<00:16, 8273.77it/s] 66%|██████▌   | 263535/400000 [00:25<00:15, 8569.41it/s] 66%|██████▌   | 264486/400000 [00:25<00:15, 8829.87it/s] 66%|██████▋   | 265441/400000 [00:26<00:14, 9031.71it/s] 67%|██████▋   | 266423/400000 [00:26<00:14, 9253.32it/s] 67%|██████▋   | 267355/400000 [00:26<00:14, 9191.03it/s] 67%|██████▋   | 268392/400000 [00:26<00:13, 9513.37it/s] 67%|██████▋   | 269418/400000 [00:26<00:13, 9723.18it/s] 68%|██████▊   | 270401/400000 [00:26<00:13, 9753.82it/s] 68%|██████▊   | 271381/400000 [00:26<00:13, 9703.19it/s] 68%|██████▊   | 272354/400000 [00:26<00:14, 9016.27it/s] 68%|██████▊   | 273317/400000 [00:26<00:13, 9190.58it/s] 69%|██████▊   | 274309/400000 [00:26<00:13, 9392.85it/s] 69%|██████▉   | 275275/400000 [00:27<00:13, 9468.80it/s] 69%|██████▉   | 276228/400000 [00:27<00:13, 9379.18it/s] 69%|██████▉   | 277170/400000 [00:27<00:13, 9158.05it/s] 70%|██████▉   | 278248/400000 [00:27<00:12, 9590.96it/s] 70%|██████▉   | 279372/400000 [00:27<00:12, 10031.93it/s] 70%|███████   | 280455/400000 [00:27<00:11, 10255.90it/s] 70%|███████   | 281509/400000 [00:27<00:11, 10338.49it/s] 71%|███████   | 282550/400000 [00:27<00:11, 9976.29it/s]  71%|███████   | 283574/400000 [00:27<00:11, 10053.59it/s] 71%|███████   | 284692/400000 [00:27<00:11, 10365.44it/s] 71%|███████▏  | 285770/400000 [00:28<00:10, 10483.41it/s] 72%|███████▏  | 286863/400000 [00:28<00:10, 10609.61it/s] 72%|███████▏  | 287928/400000 [00:28<00:10, 10301.52it/s] 72%|███████▏  | 289005/400000 [00:28<00:10, 10436.39it/s] 73%|███████▎  | 290053/400000 [00:28<00:10, 10424.65it/s] 73%|███████▎  | 291098/400000 [00:28<00:10, 10231.69it/s] 73%|███████▎  | 292124/400000 [00:28<00:10, 10113.58it/s] 73%|███████▎  | 293138/400000 [00:28<00:10, 9940.74it/s]  74%|███████▎  | 294163/400000 [00:28<00:10, 10029.64it/s] 74%|███████▍  | 295168/400000 [00:29<00:10, 9996.88it/s]  74%|███████▍  | 296169/400000 [00:29<00:10, 9980.14it/s] 74%|███████▍  | 297185/400000 [00:29<00:10, 10032.18it/s] 75%|███████▍  | 298189/400000 [00:29<00:10, 9982.96it/s]  75%|███████▍  | 299188/400000 [00:29<00:10, 9824.69it/s] 75%|███████▌  | 300172/400000 [00:29<00:10, 9779.66it/s] 75%|███████▌  | 301168/400000 [00:29<00:10, 9830.78it/s] 76%|███████▌  | 302172/400000 [00:29<00:09, 9892.27it/s] 76%|███████▌  | 303162/400000 [00:29<00:10, 9646.71it/s] 76%|███████▌  | 304155/400000 [00:29<00:09, 9729.19it/s] 76%|███████▋  | 305150/400000 [00:30<00:09, 9792.90it/s] 77%|███████▋  | 306157/400000 [00:30<00:09, 9867.65it/s] 77%|███████▋  | 307163/400000 [00:30<00:09, 9923.72it/s] 77%|███████▋  | 308191/400000 [00:30<00:09, 10027.12it/s] 77%|███████▋  | 309221/400000 [00:30<00:08, 10107.34it/s] 78%|███████▊  | 310233/400000 [00:30<00:09, 9918.24it/s]  78%|███████▊  | 311227/400000 [00:30<00:09, 9794.12it/s] 78%|███████▊  | 312208/400000 [00:30<00:08, 9774.25it/s] 78%|███████▊  | 313187/400000 [00:30<00:08, 9684.97it/s] 79%|███████▊  | 314215/400000 [00:30<00:08, 9855.14it/s] 79%|███████▉  | 315202/400000 [00:31<00:08, 9496.30it/s] 79%|███████▉  | 316156/400000 [00:31<00:09, 9239.48it/s] 79%|███████▉  | 317085/400000 [00:31<00:08, 9251.68it/s] 80%|███████▉  | 318036/400000 [00:31<00:08, 9326.64it/s] 80%|███████▉  | 319038/400000 [00:31<00:08, 9522.55it/s] 80%|████████  | 320031/400000 [00:31<00:08, 9639.19it/s] 80%|████████  | 320998/400000 [00:31<00:08, 9635.87it/s] 80%|████████  | 321971/400000 [00:31<00:08, 9662.00it/s] 81%|████████  | 322939/400000 [00:31<00:08, 9472.75it/s] 81%|████████  | 323888/400000 [00:32<00:08, 9101.43it/s] 81%|████████  | 324803/400000 [00:32<00:08, 9053.65it/s] 81%|████████▏ | 325782/400000 [00:32<00:08, 9262.26it/s] 82%|████████▏ | 326872/400000 [00:32<00:07, 9698.41it/s] 82%|████████▏ | 327870/400000 [00:32<00:07, 9778.82it/s] 82%|████████▏ | 328870/400000 [00:32<00:07, 9842.02it/s] 82%|████████▏ | 329859/400000 [00:32<00:07, 9737.45it/s] 83%|████████▎ | 330846/400000 [00:32<00:07, 9775.17it/s] 83%|████████▎ | 331826/400000 [00:32<00:07, 9712.27it/s] 83%|████████▎ | 332799/400000 [00:32<00:06, 9684.51it/s] 83%|████████▎ | 333829/400000 [00:33<00:06, 9860.16it/s] 84%|████████▎ | 334924/400000 [00:33<00:06, 10162.86it/s] 84%|████████▍ | 335955/400000 [00:33<00:06, 10204.02it/s] 84%|████████▍ | 336978/400000 [00:33<00:06, 9951.44it/s]  84%|████████▍ | 337977/400000 [00:33<00:06, 9823.80it/s] 85%|████████▍ | 338977/400000 [00:33<00:06, 9873.58it/s] 85%|████████▌ | 340001/400000 [00:33<00:06, 9977.42it/s] 85%|████████▌ | 341001/400000 [00:33<00:06, 9814.87it/s] 85%|████████▌ | 341985/400000 [00:33<00:05, 9807.10it/s] 86%|████████▌ | 342967/400000 [00:33<00:05, 9768.18it/s] 86%|████████▌ | 343945/400000 [00:34<00:05, 9721.52it/s] 86%|████████▌ | 344968/400000 [00:34<00:05, 9866.26it/s] 86%|████████▋ | 345997/400000 [00:34<00:05, 9987.64it/s] 87%|████████▋ | 346997/400000 [00:34<00:05, 9958.41it/s] 87%|████████▋ | 347994/400000 [00:34<00:05, 9906.75it/s] 87%|████████▋ | 348991/400000 [00:34<00:05, 9925.31it/s] 87%|████████▋ | 349984/400000 [00:34<00:05, 9789.32it/s] 88%|████████▊ | 350964/400000 [00:34<00:05, 9710.13it/s] 88%|████████▊ | 351936/400000 [00:34<00:05, 9564.55it/s] 88%|████████▊ | 352902/400000 [00:34<00:04, 9590.66it/s] 88%|████████▊ | 353862/400000 [00:35<00:04, 9540.96it/s] 89%|████████▊ | 354822/400000 [00:35<00:04, 9557.05it/s] 89%|████████▉ | 355784/400000 [00:35<00:04, 9575.55it/s] 89%|████████▉ | 356742/400000 [00:35<00:04, 9556.43it/s] 89%|████████▉ | 357698/400000 [00:35<00:04, 9495.49it/s] 90%|████████▉ | 358675/400000 [00:35<00:04, 9574.96it/s] 90%|████████▉ | 359633/400000 [00:35<00:04, 9279.51it/s] 90%|█████████ | 360605/400000 [00:35<00:04, 9404.62it/s] 90%|█████████ | 361561/400000 [00:35<00:04, 9449.03it/s] 91%|█████████ | 362508/400000 [00:35<00:03, 9409.79it/s] 91%|█████████ | 363558/400000 [00:36<00:03, 9711.49it/s] 91%|█████████ | 364560/400000 [00:36<00:03, 9800.72it/s] 91%|█████████▏| 365543/400000 [00:36<00:03, 9750.28it/s] 92%|█████████▏| 366571/400000 [00:36<00:03, 9901.12it/s] 92%|█████████▏| 367572/400000 [00:36<00:03, 9931.41it/s] 92%|█████████▏| 368592/400000 [00:36<00:03, 10010.48it/s] 92%|█████████▏| 369647/400000 [00:36<00:02, 10164.22it/s] 93%|█████████▎| 370665/400000 [00:36<00:02, 10137.41it/s] 93%|█████████▎| 371718/400000 [00:36<00:02, 10250.30it/s] 93%|█████████▎| 372744/400000 [00:36<00:02, 10213.52it/s] 93%|█████████▎| 373767/400000 [00:37<00:02, 10105.66it/s] 94%|█████████▎| 374779/400000 [00:37<00:02, 9970.34it/s]  94%|█████████▍| 375777/400000 [00:37<00:02, 9918.38it/s] 94%|█████████▍| 376770/400000 [00:37<00:02, 9920.98it/s] 94%|█████████▍| 377763/400000 [00:37<00:02, 9596.65it/s] 95%|█████████▍| 378766/400000 [00:37<00:02, 9720.35it/s] 95%|█████████▍| 379788/400000 [00:37<00:02, 9864.04it/s] 95%|█████████▌| 380777/400000 [00:37<00:01, 9793.10it/s] 95%|█████████▌| 381758/400000 [00:37<00:01, 9673.72it/s] 96%|█████████▌| 382727/400000 [00:38<00:01, 9609.51it/s] 96%|█████████▌| 383690/400000 [00:38<00:01, 9602.60it/s] 96%|█████████▌| 384652/400000 [00:38<00:01, 9528.16it/s] 96%|█████████▋| 385613/400000 [00:38<00:01, 9550.33it/s] 97%|█████████▋| 386576/400000 [00:38<00:01, 9572.65it/s] 97%|█████████▋| 387534/400000 [00:38<00:01, 9473.17it/s] 97%|█████████▋| 388498/400000 [00:38<00:01, 9519.85it/s] 97%|█████████▋| 389451/400000 [00:38<00:01, 9441.23it/s] 98%|█████████▊| 390443/400000 [00:38<00:00, 9579.01it/s] 98%|█████████▊| 391420/400000 [00:38<00:00, 9632.92it/s] 98%|█████████▊| 392388/400000 [00:39<00:00, 9646.08it/s] 98%|█████████▊| 393354/400000 [00:39<00:00, 9647.12it/s] 99%|█████████▊| 394320/400000 [00:39<00:00, 9630.76it/s] 99%|█████████▉| 395284/400000 [00:39<00:00, 9429.93it/s] 99%|█████████▉| 396229/400000 [00:39<00:00, 9385.57it/s] 99%|█████████▉| 397169/400000 [00:39<00:00, 9227.45it/s]100%|█████████▉| 398093/400000 [00:39<00:00, 9184.01it/s]100%|█████████▉| 399030/400000 [00:39<00:00, 9238.83it/s]100%|█████████▉| 399955/400000 [00:39<00:00, 9228.18it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10039.72it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f397ac154e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011255804817052358 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.010953981541470939 	 Accuracy: 68

  model saves at 68% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15702 out of table with 15596 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15702 out of table with 15596 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

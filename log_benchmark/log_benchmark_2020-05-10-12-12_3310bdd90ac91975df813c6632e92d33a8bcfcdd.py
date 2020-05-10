
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fe9727a4470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 12:12:42.620616
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 12:12:42.624013
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 12:12:42.626791
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 12:12:42.629941
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fe967ef8b70> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 356678.1875
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 296854.0312
Epoch 3/10

1/1 [==============================] - 0s 89ms/step - loss: 214792.0781
Epoch 4/10

1/1 [==============================] - 0s 89ms/step - loss: 138750.3438
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 80687.7812
Epoch 6/10

1/1 [==============================] - 0s 85ms/step - loss: 44814.7578
Epoch 7/10

1/1 [==============================] - 0s 83ms/step - loss: 25691.2910
Epoch 8/10

1/1 [==============================] - 0s 86ms/step - loss: 15952.6504
Epoch 9/10

1/1 [==============================] - 0s 86ms/step - loss: 10727.2998
Epoch 10/10

1/1 [==============================] - 0s 114ms/step - loss: 7748.4126

  #### Inference Need return ypred, ytrue ######################### 
[[ -1.9558698    0.50431657   1.2602959   -0.68770003   0.08899506
   -0.01367244   0.539356    -0.2313261    1.2135811    0.12469003
    2.1344936    0.6264138   -1.525276    -0.9585695   -0.6817538
   -0.09242159   0.04427847  -0.08965099  -1.8229053   -0.15951389
   -0.13145077   1.046623    -0.11071026  -1.5491121   -1.2730072
   -0.85888207   0.5578727    0.34176713  -0.14150622  -0.92281395
   -0.31577867   0.12564152   0.07803918  -0.02398306   0.210026
    0.57050574  -0.12419358  -0.23501146  -1.7668202   -0.69498885
    0.8739959   -1.1370304    0.35398015   0.03376088  -0.36168373
    0.27636173   1.5196173   -0.0891628    0.8334358   -1.0436265
   -1.2691637   -0.9983189   -1.4710308    0.575806    -0.8132874
    0.5883226   -0.15552786  -1.1223042   -0.77553886  -1.2649975
    1.6797149    1.3199525    1.0202136    2.0025592    0.24274448
   -1.8404886   -1.1932809    0.54893154   0.9830023    1.7652324
   -1.5580318   -0.58402485   0.79119843   1.7732615    0.67769206
   -0.13726656  -0.7310396   -0.08785623   1.3128077    0.10287328
   -1.8232627    0.69965374  -1.1691751    1.1853948   -1.917424
    0.48244688  -1.2881098    0.0270406   -0.6256786    1.5713071
   -0.14561546  -0.5379732    0.99398005   0.4577757   -1.8234243
    0.48898286   0.89509344   0.46706998  -0.1619084    1.2589824
    0.49400747   0.38185108   0.2026313    0.21672848  -0.86580503
    1.797242     0.8498484   -0.02346191  -0.4218711    0.04298735
    0.80041945  -2.0959692    1.1407019   -1.1308434    0.78323114
    0.29033303   1.2782642   -0.23408258  -2.0317903   -1.8678107
    0.08820084   7.786519     7.0524116    9.081469     7.8958893
    7.7566724    8.800009     7.05153      8.146209     8.814031
    7.8166647    8.07566      8.516322     6.7369823    6.2152205
    8.234625     7.6428757    7.372898     7.193915     7.518365
    9.309524     6.4164004    8.566816     6.246636     8.126948
    8.735587     6.0235567    7.998315     8.340016     7.184125
    7.6287565    8.347113     7.913019     7.8442006    7.2419424
    6.08701      7.8672667    8.348346     7.3724055    6.7282643
    8.381743     7.3886356    8.767461     7.5998397    7.845066
    7.0911746    6.201779     7.5246806    5.8645024    6.6692057
    7.694255     5.9557447    9.194492     7.348995     8.138478
    7.3892593    8.117373     7.9086843    6.8021035    7.501957
    0.69064736   1.2854704    1.7018653    0.7058118    2.3211102
    1.5298455    2.062341     2.162059     0.53956693   0.8071885
    2.4352174    2.5437465    0.6243762    0.8709538    2.4683733
    1.2995317    1.9711862    0.22151858   2.0839837    0.4556129
    0.45282412   0.91691345   0.56745875   0.8660342    0.16959715
    1.0562305    0.32750154   2.912735     0.30634886   1.3724926
    1.1192095    0.1634103    1.1396725    0.47572064   0.30546045
    0.59138525   0.8515798    1.5400625    2.4872303    1.919741
    0.19550157   0.55430216   1.2739425    0.18809319   2.5440722
    1.556953     0.36949927   0.71469235   1.9175959    2.1325586
    2.810855     2.205927     0.4725349    1.6328932    0.8903898
    0.7036692    0.32338452   1.5226719    0.8378818    0.37496525
    2.2363977    1.7685183    0.6617244    1.7407837    0.25340545
    0.66943574   1.5977715    2.19837      2.270362     1.9244286
    0.49746764   2.0521278    1.1533439    0.2738508    1.4123396
    0.30783975   1.1813385    0.6658085    1.7974364    0.9545569
    0.2795825    1.5500014    0.296592     1.8359351    0.76349837
    0.6435233    1.7286106    0.2568336    1.1219286    1.5935477
    0.78322715   0.5703174    1.1646196    0.86095166   2.1810198
    0.8953741    0.23801196   0.6184202    0.6195       1.1036694
    1.714231     0.6941899    0.33211643   0.5242772    1.3706095
    2.3185515    0.54053885   1.2051758    1.4624007    0.43208086
    0.35826677   0.6637889    2.436998     1.2440326    1.1233897
    0.18567342   0.34853423   1.902899     0.84808254   1.0244169
    0.09564221   7.766102     7.1088624    7.993352     7.071132
    8.788033     6.6325083    7.3881626    8.545798     7.7086344
    8.330343     8.446623     7.6713853    9.031154     7.422517
    7.4436035    7.0337462    7.5866957    8.453329     8.438617
    6.9898233    8.292904     7.5486555    9.047402     8.86933
    6.7320147    8.294184     7.7819767   10.195427     7.599905
    8.125622     6.486693     9.589676     7.795        7.644248
    7.103975     7.552113     6.2843056    8.717088     8.2048235
    8.759504     8.103477     8.194974     8.433297     8.256232
    8.264767     7.566213     7.668063     7.9965835    6.530051
    7.787766     8.403881     5.8653045    7.3003607    8.269654
    8.395256     9.344011     7.42837      8.864693     7.3640947
  -10.733737    -4.1215277    5.779909  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 12:12:50.015055
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.709
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 12:12:50.018374
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8990.38
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 12:12:50.020973
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.1483
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 12:12:50.023911
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -804.152
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140639935926400
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140638994310032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140638994310536
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140638994311040
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140638994311544
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140638994312048

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fe966975f60> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.505154
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.476992
grad_step = 000002, loss = 0.460242
grad_step = 000003, loss = 0.442596
grad_step = 000004, loss = 0.422855
grad_step = 000005, loss = 0.400872
grad_step = 000006, loss = 0.378361
grad_step = 000007, loss = 0.359637
grad_step = 000008, loss = 0.346945
grad_step = 000009, loss = 0.338142
grad_step = 000010, loss = 0.326714
grad_step = 000011, loss = 0.311342
grad_step = 000012, loss = 0.297499
grad_step = 000013, loss = 0.287673
grad_step = 000014, loss = 0.279734
grad_step = 000015, loss = 0.271035
grad_step = 000016, loss = 0.261165
grad_step = 000017, loss = 0.250544
grad_step = 000018, loss = 0.239700
grad_step = 000019, loss = 0.229407
grad_step = 000020, loss = 0.219945
grad_step = 000021, loss = 0.210889
grad_step = 000022, loss = 0.201635
grad_step = 000023, loss = 0.190914
grad_step = 000024, loss = 0.180258
grad_step = 000025, loss = 0.170454
grad_step = 000026, loss = 0.161573
grad_step = 000027, loss = 0.153416
grad_step = 000028, loss = 0.145621
grad_step = 000029, loss = 0.137681
grad_step = 000030, loss = 0.129491
grad_step = 000031, loss = 0.121349
grad_step = 000032, loss = 0.113625
grad_step = 000033, loss = 0.106595
grad_step = 000034, loss = 0.099981
grad_step = 000035, loss = 0.093405
grad_step = 000036, loss = 0.086866
grad_step = 000037, loss = 0.080579
grad_step = 000038, loss = 0.074633
grad_step = 000039, loss = 0.069000
grad_step = 000040, loss = 0.063661
grad_step = 000041, loss = 0.058623
grad_step = 000042, loss = 0.053804
grad_step = 000043, loss = 0.049237
grad_step = 000044, loss = 0.044941
grad_step = 000045, loss = 0.040962
grad_step = 000046, loss = 0.037280
grad_step = 000047, loss = 0.033778
grad_step = 000048, loss = 0.030530
grad_step = 000049, loss = 0.027505
grad_step = 000050, loss = 0.024733
grad_step = 000051, loss = 0.022217
grad_step = 000052, loss = 0.019945
grad_step = 000053, loss = 0.017886
grad_step = 000054, loss = 0.015967
grad_step = 000055, loss = 0.014212
grad_step = 000056, loss = 0.012646
grad_step = 000057, loss = 0.011260
grad_step = 000058, loss = 0.010033
grad_step = 000059, loss = 0.008942
grad_step = 000060, loss = 0.007979
grad_step = 000061, loss = 0.007143
grad_step = 000062, loss = 0.006440
grad_step = 000063, loss = 0.005829
grad_step = 000064, loss = 0.005286
grad_step = 000065, loss = 0.004819
grad_step = 000066, loss = 0.004441
grad_step = 000067, loss = 0.004133
grad_step = 000068, loss = 0.003876
grad_step = 000069, loss = 0.003652
grad_step = 000070, loss = 0.003458
grad_step = 000071, loss = 0.003301
grad_step = 000072, loss = 0.003177
grad_step = 000073, loss = 0.003078
grad_step = 000074, loss = 0.002991
grad_step = 000075, loss = 0.002920
grad_step = 000076, loss = 0.002859
grad_step = 000077, loss = 0.002809
grad_step = 000078, loss = 0.002767
grad_step = 000079, loss = 0.002725
grad_step = 000080, loss = 0.002689
grad_step = 000081, loss = 0.002659
grad_step = 000082, loss = 0.002636
grad_step = 000083, loss = 0.002613
grad_step = 000084, loss = 0.002590
grad_step = 000085, loss = 0.002568
grad_step = 000086, loss = 0.002557
grad_step = 000087, loss = 0.002564
grad_step = 000088, loss = 0.002599
grad_step = 000089, loss = 0.002679
grad_step = 000090, loss = 0.002792
grad_step = 000091, loss = 0.002870
grad_step = 000092, loss = 0.002738
grad_step = 000093, loss = 0.002488
grad_step = 000094, loss = 0.002380
grad_step = 000095, loss = 0.002491
grad_step = 000096, loss = 0.002606
grad_step = 000097, loss = 0.002515
grad_step = 000098, loss = 0.002353
grad_step = 000099, loss = 0.002336
grad_step = 000100, loss = 0.002432
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002445
grad_step = 000102, loss = 0.002336
grad_step = 000103, loss = 0.002276
grad_step = 000104, loss = 0.002326
grad_step = 000105, loss = 0.002362
grad_step = 000106, loss = 0.002311
grad_step = 000107, loss = 0.002248
grad_step = 000108, loss = 0.002257
grad_step = 000109, loss = 0.002295
grad_step = 000110, loss = 0.002285
grad_step = 000111, loss = 0.002237
grad_step = 000112, loss = 0.002216
grad_step = 000113, loss = 0.002236
grad_step = 000114, loss = 0.002252
grad_step = 000115, loss = 0.002234
grad_step = 000116, loss = 0.002203
grad_step = 000117, loss = 0.002193
grad_step = 000118, loss = 0.002206
grad_step = 000119, loss = 0.002216
grad_step = 000120, loss = 0.002206
grad_step = 000121, loss = 0.002186
grad_step = 000122, loss = 0.002174
grad_step = 000123, loss = 0.002176
grad_step = 000124, loss = 0.002183
grad_step = 000125, loss = 0.002185
grad_step = 000126, loss = 0.002178
grad_step = 000127, loss = 0.002166
grad_step = 000128, loss = 0.002156
grad_step = 000129, loss = 0.002151
grad_step = 000130, loss = 0.002152
grad_step = 000131, loss = 0.002154
grad_step = 000132, loss = 0.002155
grad_step = 000133, loss = 0.002153
grad_step = 000134, loss = 0.002149
grad_step = 000135, loss = 0.002143
grad_step = 000136, loss = 0.002136
grad_step = 000137, loss = 0.002130
grad_step = 000138, loss = 0.002124
grad_step = 000139, loss = 0.002119
grad_step = 000140, loss = 0.002114
grad_step = 000141, loss = 0.002110
grad_step = 000142, loss = 0.002107
grad_step = 000143, loss = 0.002104
grad_step = 000144, loss = 0.002101
grad_step = 000145, loss = 0.002101
grad_step = 000146, loss = 0.002107
grad_step = 000147, loss = 0.002126
grad_step = 000148, loss = 0.002178
grad_step = 000149, loss = 0.002298
grad_step = 000150, loss = 0.002547
grad_step = 000151, loss = 0.002912
grad_step = 000152, loss = 0.003116
grad_step = 000153, loss = 0.002817
grad_step = 000154, loss = 0.002201
grad_step = 000155, loss = 0.002127
grad_step = 000156, loss = 0.002548
grad_step = 000157, loss = 0.002651
grad_step = 000158, loss = 0.002254
grad_step = 000159, loss = 0.002067
grad_step = 000160, loss = 0.002340
grad_step = 000161, loss = 0.002436
grad_step = 000162, loss = 0.002146
grad_step = 000163, loss = 0.002060
grad_step = 000164, loss = 0.002260
grad_step = 000165, loss = 0.002256
grad_step = 000166, loss = 0.002057
grad_step = 000167, loss = 0.002067
grad_step = 000168, loss = 0.002198
grad_step = 000169, loss = 0.002138
grad_step = 000170, loss = 0.002019
grad_step = 000171, loss = 0.002067
grad_step = 000172, loss = 0.002135
grad_step = 000173, loss = 0.002061
grad_step = 000174, loss = 0.002000
grad_step = 000175, loss = 0.002051
grad_step = 000176, loss = 0.002077
grad_step = 000177, loss = 0.002016
grad_step = 000178, loss = 0.001986
grad_step = 000179, loss = 0.002026
grad_step = 000180, loss = 0.002034
grad_step = 000181, loss = 0.001989
grad_step = 000182, loss = 0.001971
grad_step = 000183, loss = 0.001996
grad_step = 000184, loss = 0.002001
grad_step = 000185, loss = 0.001970
grad_step = 000186, loss = 0.001953
grad_step = 000187, loss = 0.001966
grad_step = 000188, loss = 0.001973
grad_step = 000189, loss = 0.001955
grad_step = 000190, loss = 0.001937
grad_step = 000191, loss = 0.001939
grad_step = 000192, loss = 0.001944
grad_step = 000193, loss = 0.001938
grad_step = 000194, loss = 0.001927
grad_step = 000195, loss = 0.001923
grad_step = 000196, loss = 0.001927
grad_step = 000197, loss = 0.001924
grad_step = 000198, loss = 0.001915
grad_step = 000199, loss = 0.001906
grad_step = 000200, loss = 0.001903
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001901
grad_step = 000202, loss = 0.001895
grad_step = 000203, loss = 0.001887
grad_step = 000204, loss = 0.001880
grad_step = 000205, loss = 0.001877
grad_step = 000206, loss = 0.001879
grad_step = 000207, loss = 0.001886
grad_step = 000208, loss = 0.001919
grad_step = 000209, loss = 0.001951
grad_step = 000210, loss = 0.001997
grad_step = 000211, loss = 0.001912
grad_step = 000212, loss = 0.001929
grad_step = 000213, loss = 0.001934
grad_step = 000214, loss = 0.001849
grad_step = 000215, loss = 0.001883
grad_step = 000216, loss = 0.001903
grad_step = 000217, loss = 0.001842
grad_step = 000218, loss = 0.001912
grad_step = 000219, loss = 0.001954
grad_step = 000220, loss = 0.001875
grad_step = 000221, loss = 0.001966
grad_step = 000222, loss = 0.001909
grad_step = 000223, loss = 0.001856
grad_step = 000224, loss = 0.001897
grad_step = 000225, loss = 0.001832
grad_step = 000226, loss = 0.001838
grad_step = 000227, loss = 0.001839
grad_step = 000228, loss = 0.001810
grad_step = 000229, loss = 0.001824
grad_step = 000230, loss = 0.001805
grad_step = 000231, loss = 0.001805
grad_step = 000232, loss = 0.001800
grad_step = 000233, loss = 0.001795
grad_step = 000234, loss = 0.001788
grad_step = 000235, loss = 0.001790
grad_step = 000236, loss = 0.001786
grad_step = 000237, loss = 0.001789
grad_step = 000238, loss = 0.001818
grad_step = 000239, loss = 0.001894
grad_step = 000240, loss = 0.002101
grad_step = 000241, loss = 0.002736
grad_step = 000242, loss = 0.003496
grad_step = 000243, loss = 0.004224
grad_step = 000244, loss = 0.002956
grad_step = 000245, loss = 0.001795
grad_step = 000246, loss = 0.002485
grad_step = 000247, loss = 0.003091
grad_step = 000248, loss = 0.002299
grad_step = 000249, loss = 0.001850
grad_step = 000250, loss = 0.002639
grad_step = 000251, loss = 0.002403
grad_step = 000252, loss = 0.001791
grad_step = 000253, loss = 0.002273
grad_step = 000254, loss = 0.002323
grad_step = 000255, loss = 0.001788
grad_step = 000256, loss = 0.002078
grad_step = 000257, loss = 0.002186
grad_step = 000258, loss = 0.001755
grad_step = 000259, loss = 0.001979
grad_step = 000260, loss = 0.002052
grad_step = 000261, loss = 0.001747
grad_step = 000262, loss = 0.001907
grad_step = 000263, loss = 0.001949
grad_step = 000264, loss = 0.001724
grad_step = 000265, loss = 0.001862
grad_step = 000266, loss = 0.001871
grad_step = 000267, loss = 0.001722
grad_step = 000268, loss = 0.001815
grad_step = 000269, loss = 0.001830
grad_step = 000270, loss = 0.001703
grad_step = 000271, loss = 0.001784
grad_step = 000272, loss = 0.001786
grad_step = 000273, loss = 0.001696
grad_step = 000274, loss = 0.001752
grad_step = 000275, loss = 0.001762
grad_step = 000276, loss = 0.001686
grad_step = 000277, loss = 0.001726
grad_step = 000278, loss = 0.001742
grad_step = 000279, loss = 0.001678
grad_step = 000280, loss = 0.001702
grad_step = 000281, loss = 0.001723
grad_step = 000282, loss = 0.001675
grad_step = 000283, loss = 0.001679
grad_step = 000284, loss = 0.001706
grad_step = 000285, loss = 0.001669
grad_step = 000286, loss = 0.001662
grad_step = 000287, loss = 0.001684
grad_step = 000288, loss = 0.001666
grad_step = 000289, loss = 0.001648
grad_step = 000290, loss = 0.001661
grad_step = 000291, loss = 0.001660
grad_step = 000292, loss = 0.001641
grad_step = 000293, loss = 0.001641
grad_step = 000294, loss = 0.001648
grad_step = 000295, loss = 0.001639
grad_step = 000296, loss = 0.001628
grad_step = 000297, loss = 0.001632
grad_step = 000298, loss = 0.001633
grad_step = 000299, loss = 0.001623
grad_step = 000300, loss = 0.001617
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001620
grad_step = 000302, loss = 0.001619
grad_step = 000303, loss = 0.001611
grad_step = 000304, loss = 0.001606
grad_step = 000305, loss = 0.001607
grad_step = 000306, loss = 0.001606
grad_step = 000307, loss = 0.001600
grad_step = 000308, loss = 0.001595
grad_step = 000309, loss = 0.001594
grad_step = 000310, loss = 0.001593
grad_step = 000311, loss = 0.001591
grad_step = 000312, loss = 0.001586
grad_step = 000313, loss = 0.001581
grad_step = 000314, loss = 0.001579
grad_step = 000315, loss = 0.001578
grad_step = 000316, loss = 0.001576
grad_step = 000317, loss = 0.001573
grad_step = 000318, loss = 0.001569
grad_step = 000319, loss = 0.001565
grad_step = 000320, loss = 0.001562
grad_step = 000321, loss = 0.001560
grad_step = 000322, loss = 0.001558
grad_step = 000323, loss = 0.001556
grad_step = 000324, loss = 0.001554
grad_step = 000325, loss = 0.001552
grad_step = 000326, loss = 0.001549
grad_step = 000327, loss = 0.001547
grad_step = 000328, loss = 0.001544
grad_step = 000329, loss = 0.001542
grad_step = 000330, loss = 0.001540
grad_step = 000331, loss = 0.001539
grad_step = 000332, loss = 0.001539
grad_step = 000333, loss = 0.001542
grad_step = 000334, loss = 0.001546
grad_step = 000335, loss = 0.001560
grad_step = 000336, loss = 0.001573
grad_step = 000337, loss = 0.001606
grad_step = 000338, loss = 0.001613
grad_step = 000339, loss = 0.001633
grad_step = 000340, loss = 0.001602
grad_step = 000341, loss = 0.001570
grad_step = 000342, loss = 0.001530
grad_step = 000343, loss = 0.001505
grad_step = 000344, loss = 0.001504
grad_step = 000345, loss = 0.001521
grad_step = 000346, loss = 0.001540
grad_step = 000347, loss = 0.001545
grad_step = 000348, loss = 0.001544
grad_step = 000349, loss = 0.001524
grad_step = 000350, loss = 0.001507
grad_step = 000351, loss = 0.001488
grad_step = 000352, loss = 0.001479
grad_step = 000353, loss = 0.001478
grad_step = 000354, loss = 0.001482
grad_step = 000355, loss = 0.001490
grad_step = 000356, loss = 0.001499
grad_step = 000357, loss = 0.001512
grad_step = 000358, loss = 0.001524
grad_step = 000359, loss = 0.001543
grad_step = 000360, loss = 0.001554
grad_step = 000361, loss = 0.001575
grad_step = 000362, loss = 0.001569
grad_step = 000363, loss = 0.001564
grad_step = 000364, loss = 0.001536
grad_step = 000365, loss = 0.001503
grad_step = 000366, loss = 0.001469
grad_step = 000367, loss = 0.001444
grad_step = 000368, loss = 0.001435
grad_step = 000369, loss = 0.001440
grad_step = 000370, loss = 0.001452
grad_step = 000371, loss = 0.001465
grad_step = 000372, loss = 0.001478
grad_step = 000373, loss = 0.001487
grad_step = 000374, loss = 0.001497
grad_step = 000375, loss = 0.001498
grad_step = 000376, loss = 0.001501
grad_step = 000377, loss = 0.001493
grad_step = 000378, loss = 0.001484
grad_step = 000379, loss = 0.001468
grad_step = 000380, loss = 0.001450
grad_step = 000381, loss = 0.001431
grad_step = 000382, loss = 0.001414
grad_step = 000383, loss = 0.001402
grad_step = 000384, loss = 0.001393
grad_step = 000385, loss = 0.001387
grad_step = 000386, loss = 0.001383
grad_step = 000387, loss = 0.001381
grad_step = 000388, loss = 0.001381
grad_step = 000389, loss = 0.001384
grad_step = 000390, loss = 0.001391
grad_step = 000391, loss = 0.001410
grad_step = 000392, loss = 0.001450
grad_step = 000393, loss = 0.001549
grad_step = 000394, loss = 0.001716
grad_step = 000395, loss = 0.002029
grad_step = 000396, loss = 0.002222
grad_step = 000397, loss = 0.002240
grad_step = 000398, loss = 0.001858
grad_step = 000399, loss = 0.001442
grad_step = 000400, loss = 0.001412
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001691
grad_step = 000402, loss = 0.001805
grad_step = 000403, loss = 0.001563
grad_step = 000404, loss = 0.001350
grad_step = 000405, loss = 0.001456
grad_step = 000406, loss = 0.001629
grad_step = 000407, loss = 0.001571
grad_step = 000408, loss = 0.001383
grad_step = 000409, loss = 0.001344
grad_step = 000410, loss = 0.001465
grad_step = 000411, loss = 0.001510
grad_step = 000412, loss = 0.001409
grad_step = 000413, loss = 0.001324
grad_step = 000414, loss = 0.001356
grad_step = 000415, loss = 0.001420
grad_step = 000416, loss = 0.001401
grad_step = 000417, loss = 0.001340
grad_step = 000418, loss = 0.001315
grad_step = 000419, loss = 0.001333
grad_step = 000420, loss = 0.001348
grad_step = 000421, loss = 0.001339
grad_step = 000422, loss = 0.001324
grad_step = 000423, loss = 0.001302
grad_step = 000424, loss = 0.001289
grad_step = 000425, loss = 0.001295
grad_step = 000426, loss = 0.001307
grad_step = 000427, loss = 0.001309
grad_step = 000428, loss = 0.001280
grad_step = 000429, loss = 0.001260
grad_step = 000430, loss = 0.001265
grad_step = 000431, loss = 0.001275
grad_step = 000432, loss = 0.001276
grad_step = 000433, loss = 0.001262
grad_step = 000434, loss = 0.001252
grad_step = 000435, loss = 0.001247
grad_step = 000436, loss = 0.001241
grad_step = 000437, loss = 0.001238
grad_step = 000438, loss = 0.001240
grad_step = 000439, loss = 0.001241
grad_step = 000440, loss = 0.001236
grad_step = 000441, loss = 0.001227
grad_step = 000442, loss = 0.001219
grad_step = 000443, loss = 0.001217
grad_step = 000444, loss = 0.001214
grad_step = 000445, loss = 0.001210
grad_step = 000446, loss = 0.001203
grad_step = 000447, loss = 0.001198
grad_step = 000448, loss = 0.001196
grad_step = 000449, loss = 0.001195
grad_step = 000450, loss = 0.001193
grad_step = 000451, loss = 0.001191
grad_step = 000452, loss = 0.001190
grad_step = 000453, loss = 0.001193
grad_step = 000454, loss = 0.001206
grad_step = 000455, loss = 0.001231
grad_step = 000456, loss = 0.001289
grad_step = 000457, loss = 0.001379
grad_step = 000458, loss = 0.001571
grad_step = 000459, loss = 0.001665
grad_step = 000460, loss = 0.001788
grad_step = 000461, loss = 0.001707
grad_step = 000462, loss = 0.001506
grad_step = 000463, loss = 0.001263
grad_step = 000464, loss = 0.001153
grad_step = 000465, loss = 0.001258
grad_step = 000466, loss = 0.001411
grad_step = 000467, loss = 0.001404
grad_step = 000468, loss = 0.001258
grad_step = 000469, loss = 0.001146
grad_step = 000470, loss = 0.001165
grad_step = 000471, loss = 0.001252
grad_step = 000472, loss = 0.001285
grad_step = 000473, loss = 0.001219
grad_step = 000474, loss = 0.001139
grad_step = 000475, loss = 0.001125
grad_step = 000476, loss = 0.001168
grad_step = 000477, loss = 0.001201
grad_step = 000478, loss = 0.001176
grad_step = 000479, loss = 0.001128
grad_step = 000480, loss = 0.001100
grad_step = 000481, loss = 0.001110
grad_step = 000482, loss = 0.001135
grad_step = 000483, loss = 0.001139
grad_step = 000484, loss = 0.001118
grad_step = 000485, loss = 0.001087
grad_step = 000486, loss = 0.001073
grad_step = 000487, loss = 0.001082
grad_step = 000488, loss = 0.001095
grad_step = 000489, loss = 0.001100
grad_step = 000490, loss = 0.001085
grad_step = 000491, loss = 0.001066
grad_step = 000492, loss = 0.001053
grad_step = 000493, loss = 0.001049
grad_step = 000494, loss = 0.001054
grad_step = 000495, loss = 0.001059
grad_step = 000496, loss = 0.001061
grad_step = 000497, loss = 0.001057
grad_step = 000498, loss = 0.001050
grad_step = 000499, loss = 0.001039
grad_step = 000500, loss = 0.001030
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001022
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

  date_run                              2020-05-10 12:13:09.379272
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.214779
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 12:13:09.384981
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.10088
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 12:13:09.391237
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13195
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 12:13:09.395361
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.532912
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
0   2020-05-10 12:12:42.620616  ...    mean_absolute_error
1   2020-05-10 12:12:42.624013  ...     mean_squared_error
2   2020-05-10 12:12:42.626791  ...  median_absolute_error
3   2020-05-10 12:12:42.629941  ...               r2_score
4   2020-05-10 12:12:50.015055  ...    mean_absolute_error
5   2020-05-10 12:12:50.018374  ...     mean_squared_error
6   2020-05-10 12:12:50.020973  ...  median_absolute_error
7   2020-05-10 12:12:50.023911  ...               r2_score
8   2020-05-10 12:13:09.379272  ...    mean_absolute_error
9   2020-05-10 12:13:09.384981  ...     mean_squared_error
10  2020-05-10 12:13:09.391237  ...  median_absolute_error
11  2020-05-10 12:13:09.395361  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147580.38it/s] 83%|████████▎ | 8232960/9912422 [00:00<00:07, 210666.06it/s]9920512it [00:00, 44435235.34it/s]                           
0it [00:00, ?it/s]32768it [00:00, 615983.94it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 451869.94it/s]1654784it [00:00, 11512824.04it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186229.63it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53c360d5c0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5354d0f9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53b7584e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5354d0fda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53b75cd6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53b75cdf60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5354d12080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53b75cdf60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5354d12080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53b75cdf60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53b75cd6d8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fac9b9ea208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=63ecb544137ed79bd419657136466ee1a6c9687b186e2451e3d7f5a735a293e4
  Stored in directory: /tmp/pip-ephem-wheel-cache-pzsb367c/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fac346c6f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3866624/17464789 [=====>........................] - ETA: 0s
11313152/17464789 [==================>...........] - ETA: 0s
16556032/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 12:14:32.895109: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 12:14:32.899263: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-05-10 12:14:32.899382: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5580148861e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 12:14:32.899394: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6641 - accuracy: 0.5002
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6688 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6905 - accuracy: 0.4984
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6375 - accuracy: 0.5019
11000/25000 [============>.................] - ETA: 3s - loss: 7.6931 - accuracy: 0.4983
12000/25000 [=============>................] - ETA: 2s - loss: 7.6935 - accuracy: 0.4983
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6961 - accuracy: 0.4981
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6776 - accuracy: 0.4993
15000/25000 [=================>............] - ETA: 2s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6632 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6384 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 12:14:45.517297
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 12:14:45.517297  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 12:14:51.228546: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 12:14:51.233639: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-05-10 12:14:51.234098: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56010b70c860 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 12:14:51.234374: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6906e82ba8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2938 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.2526 - val_crf_viterbi_accuracy: 0.6667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f68e1dc7f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6155 - accuracy: 0.5033
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6145 - accuracy: 0.5034
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5951 - accuracy: 0.5047
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5681 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5593 - accuracy: 0.5070
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5814 - accuracy: 0.5056
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6268 - accuracy: 0.5026
11000/25000 [============>.................] - ETA: 3s - loss: 7.6346 - accuracy: 0.5021
12000/25000 [=============>................] - ETA: 3s - loss: 7.6347 - accuracy: 0.5021
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6171 - accuracy: 0.5032
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6338 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 2s - loss: 7.6176 - accuracy: 0.5032
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6101 - accuracy: 0.5037
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6035 - accuracy: 0.5041
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6257 - accuracy: 0.5027
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6400 - accuracy: 0.5017
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6659 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6798 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6757 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6781 - accuracy: 0.4992
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f68af246240> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:01:53, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:21:13, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:30:03, 20.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 778k/862M [00:01<8:05:36, 29.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.81M/862M [00:01<5:40:05, 42.2kB/s].vector_cache/glove.6B.zip:   1%|          | 5.94M/862M [00:01<3:57:03, 60.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.7M/862M [00:01<2:44:55, 85.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:01<1:54:45, 123kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:02<1:19:58, 175kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:02<55:50, 250kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.3M/862M [00:02<38:55, 356kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:02<27:15, 506kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.8M/862M [00:02<19:02, 720kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:02<13:37, 1.00MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:02<09:32, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<06:45, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.5M/862M [00:02<04:50, 2.78MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.0M/862M [00:04<17:38, 763kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.3M/862M [00:04<14:11, 947kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<10:23, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:06<09:47, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<08:16, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<06:08, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.3M/862M [00:08<07:15, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<06:29, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<04:49, 2.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:10<06:26, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:10<07:13, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<05:37, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:10<04:07, 3.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.5M/862M [00:12<08:24, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<07:13, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<05:21, 2.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:14<06:49, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<06:06, 2.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<04:35, 2.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:16<06:18, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:16<07:05, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<05:37, 2.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:16<04:04, 3.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:18<14:39, 885kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<11:34, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<08:25, 1.54MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:20<08:55, 1.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<07:40, 1.68MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<05:43, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:22<06:48, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:22<07:36, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:22<04:22, 2.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:24<16:41, 764kB/s] .vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<13:05, 975kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<09:29, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:25, 1.35MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:24, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:17, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:14, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<17:11, 735kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<13:25, 940kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:41, 1.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<09:30, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<08:02, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:58, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:54, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<07:37, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:55, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:21, 2.86MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<07:04, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:19, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:43, 2.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<06:01, 2.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:57, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:32, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<04:02, 3.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<17:41, 694kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:42, 895kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:52, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:34, 1.27MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:25, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:10, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:09, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<10:41, 1.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<08:49, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:29, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<07:11, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<07:42, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:04, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:23, 2.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<17:36, 682kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:37, 881kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:50, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<09:29, 1.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:17, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:09, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<05:09, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<17:58, 660kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<13:51, 855kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:00, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<07:20, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<05:29, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<19:34:56, 9.99kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<13:53:50, 14.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<9:46:24, 20.0kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<6:50:40, 28.5kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<4:46:56, 40.7kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<3:23:58, 57.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<2:23:58, 81.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:40:53, 115kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<1:12:56, 159kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<53:38, 216kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<38:09, 304kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<26:45, 431kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<32:41, 353kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<24:08, 477kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<17:09, 670kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<14:28, 792kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<12:40, 904kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:26, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<06:44, 1.69MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<16:52, 675kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<13:03, 871kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<09:26, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<09:04, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<08:52, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:43, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:53, 2.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:01, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:11, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:35, 2.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:37, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:24, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:06, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<03:41, 3.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<11:44, 946kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:25, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:53, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:13, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:25, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:41, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:07, 2.66MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:38, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:33, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:52, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:47, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:29, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:08, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<03:43, 2.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<15:44, 688kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<12:11, 888kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<08:46, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<08:29, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<08:20, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:26, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:36, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<14:52, 718kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<11:34, 924kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<08:22, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:10, 1.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:04, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:14, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<04:29, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<15:55, 662kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<12:15, 860kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<08:50, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<08:32, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<08:11, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:16, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<04:29, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<25:26, 409kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<18:56, 550kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<13:30, 769kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<11:41, 885kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:18, 1.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:46, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<06:57, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:09, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:34, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:01, 2.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<16:38, 613kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<13:48, 738kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<10:06, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<07:11, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<09:34, 1.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<07:47, 1.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:43, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:12, 1.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:29, 1.55MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:04, 1.98MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:47, 2.65MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:41, 1.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:18, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:53, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<03:34, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:00, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:30, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:01, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<03:38, 2.71MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:25, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:41, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:10, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:43, 2.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:46, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:12, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:15, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:28, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<19:09, 507kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<15:22, 632kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<11:10, 868kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<07:58, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:39, 1.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:12, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:16, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:29, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<13:54, 689kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<11:40, 820kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<08:38, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<06:07, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<13:52, 686kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:33, 823kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<08:33, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<06:04, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<38:03, 248kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<28:43, 329kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<20:31, 459kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<14:26, 650kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<14:02, 668kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<11:54, 787kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:44, 1.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<06:14, 1.49MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:48, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:26, 1.25MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:39, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<04:02, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<12:13, 756kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<10:16, 898kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:36, 1.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<06:50, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<06:54, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:16, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:47, 2.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<07:48, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:24, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:39, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:03, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<17:42, 510kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<14:01, 644kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<10:09, 887kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<07:10, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<16:32, 542kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<15:34, 575kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<11:57, 749kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:36, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<07:46, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:59, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:17, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:10, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<07:57, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<06:35, 1.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:48, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:07, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:26, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:16, 2.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:05, 2.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<17:40, 491kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<14:12, 611kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<10:20, 839kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<07:18, 1.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<10:08, 849kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<08:55, 965kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:37, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:44, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<06:49, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:12, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:42, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:44, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<05:16, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:10, 2.03MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:01, 2.78MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<10:50, 776kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<10:32, 797kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<08:05, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:47, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:20, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:22, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:51, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:30, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<05:39, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<05:24, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:08, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<04:17, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<04:55, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:54, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:49, 2.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<09:59, 813kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<08:45, 928kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<06:32, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:40, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<14:49, 544kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<12:06, 666kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<08:52, 906kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<06:16, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<09:21, 854kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<08:14, 969kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:41<06:10, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:24, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<17:15, 459kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<13:51, 572kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<10:04, 786kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:08, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<07:58, 986kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<07:25, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:34, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:59, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<05:47, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<05:52, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:28, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:13, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<05:35, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<05:42, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<04:22, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:08, 2.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:47, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<06:26, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<04:52, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:02, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:12, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<03:59, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<02:54, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:30, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:53, 1.53MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:46, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:44, 2.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:01, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<05:14, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:53, 2.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:45, 949kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<07:09, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:21, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:50, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<06:01, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<05:49, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<04:28, 1.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:12, 2.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<10:07, 714kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<08:41, 831kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<06:28, 1.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:36, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<11:23, 629kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<09:32, 750kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<07:00, 1.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:00, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:57, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<05:44, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:24, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:09, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:44, 721kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<08:26, 831kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<06:17, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:28, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<11:03, 628kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<09:18, 747kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<06:52, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:51, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<12:00, 573kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<09:51, 697kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<07:15, 945kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:12, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:32, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:19, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<04:01, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:55, 2.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:20, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:32, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<03:28, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:31, 2.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:22, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:28, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:26, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:33, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:23, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:50, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<02:59, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:09, 3.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<07:27, 877kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:36, 988kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<04:58, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:32, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<10:50, 596kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<09:01, 717kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<06:40, 967kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:42, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<10:08, 630kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<08:27, 756kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<06:15, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:25, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<11:06, 570kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<09:10, 689kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<06:41, 943kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:48, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:01, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:54, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:43, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:39, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:39, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:19, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:00, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:53, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:16, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:21, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:19, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:23, 2.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:39, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:32, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:28, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<07:30, 797kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:31, 917kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:49, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:02, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:50, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:42, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:39, 2.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<08:31, 685kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<07:15, 804kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<05:23, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:49, 1.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<09:15, 623kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:42, 748kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:40, 1.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:01, 1.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<16:23, 348kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<12:43, 448kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<09:10, 620kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<06:27, 876kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<07:03, 799kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<06:06, 923kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:32, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<03:13, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<10:20, 538kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<08:31, 652kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:15, 886kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<04:25, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<09:21, 587kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<07:46, 707kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:40, 965kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<04:01, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:26, 997kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:57, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:45, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:40, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<08:48, 608kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<07:21, 727kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:23, 992kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:49, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:43, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:17, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:14, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:19, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<51:20, 102kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<37:04, 141kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<26:09, 199kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<18:14, 283kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<15:06, 341kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<11:39, 441kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<08:22, 613kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<05:54, 864kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<06:03, 839kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:09, 985kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:29, 1.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:34, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:43, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:58, 2.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:02, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:10, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:26, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:45, 2.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:18, 1.47MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:08, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:21, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:41, 2.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<08:08, 590kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<06:49, 704kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:01, 952kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:32, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<06:37, 714kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<05:24, 874kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:58, 1.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<03:34, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<04:33, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:42, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:41, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:57, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:15, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:37, 2.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<04:04, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:48, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:51, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:03, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:17, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:58, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:13, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:35, 2.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<11:35, 379kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<09:00, 487kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<06:29, 673kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<05:12, 831kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<14:09, 305kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<12:15, 353kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<09:06, 474kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<06:30, 660kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<04:34, 932kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<10:43, 396kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<08:02, 528kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<05:44, 736kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<04:48, 871kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<05:38, 742kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:28, 933kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:15, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<03:01, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<03:10, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:29, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:47, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:05, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:12, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:29, 1.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:47, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:04, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<03:10, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:28, 1.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:46, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:58, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<03:05, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:22, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:43, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:22, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:39, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:04, 1.85MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:29, 2.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:36, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:47, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:11, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:35, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:46, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<02:56, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:17, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:39, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:47, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:53, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:14, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:36, 2.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:43, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:49, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:09, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:33, 2.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:06, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:25, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:23, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:29, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:37, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:02, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:28, 2.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:32, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:38, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:01, 1.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:02, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:15, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:47, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:17, 2.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:21, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:29, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:54, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:22, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:01, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:13, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:43, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:15, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:56, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:32, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:06, 2.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:11, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:18, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:48, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:17, 2.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:52, 1.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:45, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:17, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:40, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:42, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:51, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:27, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:03, 2.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:06, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:13, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:44, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:14, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:00, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:33, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:07, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:46, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:49, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:23, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:00, 2.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:48, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:50, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:23, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:59, 2.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:52, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:50, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:23, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:00, 2.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:32, 1.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:40, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:18, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:56, 2.55MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:51, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:52, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:27, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:02, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:51, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:55, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:29, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:03, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:42, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:36, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:13, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:51, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<31:50, 67.9kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<22:38, 95.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<15:50, 135kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<10:51, 193kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<47:35, 44.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<33:37, 62.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<23:28, 88.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<16:22, 124kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<12:21, 164kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<08:48, 229kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<06:15, 319kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<04:38, 421kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<03:43, 524kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<02:41, 721kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:52, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:03, 916kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:51, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:23, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:58, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:23, 761kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:05, 868kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:33, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:05, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:10, 806kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:47, 974kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:18, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:12, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:35, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:17, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:55, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:59, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:00, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:47, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:33, 2.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<12:58, 119kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<09:16, 166kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<06:27, 235kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<04:39, 316kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<03:48, 386kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<02:47, 524kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:56, 737kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:40, 834kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:25, 981kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<01:03, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:55, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:53, 1.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:41, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:28, 2.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<02:58, 425kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<02:18, 545kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:39, 755kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:08, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:16, 941kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:06, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:49, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:33, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<08:37, 131kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<06:13, 180kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<04:21, 254kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<02:55, 362kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<04:10, 253kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<03:06, 339kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<02:11, 473kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<01:28, 671kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<07:27, 133kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<05:22, 183kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<03:45, 258kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<02:38, 347kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<02:01, 454kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:25, 629kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:05, 782kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:54, 929kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:39, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:34, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:32, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:24, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:23, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:32, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:26, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:21, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:22, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:16, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:17, 1.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:13, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:13, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:15, 1.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:11, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:07, 3.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:23, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:20, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:15, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:13, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:21, 851kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:17, 990kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:12, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:09, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:09, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:06, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:05, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:05, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:03, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:01, 2.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.98MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.49MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 804/400000 [00:00<00:49, 8032.69it/s]  0%|          | 1630/400000 [00:00<00:49, 8097.13it/s]  1%|          | 2433/400000 [00:00<00:49, 8075.05it/s]  1%|          | 3315/400000 [00:00<00:47, 8283.20it/s]  1%|          | 4203/400000 [00:00<00:46, 8452.32it/s]  1%|▏         | 5082/400000 [00:00<00:46, 8514.71it/s]  1%|▏         | 5899/400000 [00:00<00:46, 8407.41it/s]  2%|▏         | 6801/400000 [00:00<00:45, 8580.71it/s]  2%|▏         | 7695/400000 [00:00<00:45, 8683.41it/s]  2%|▏         | 8533/400000 [00:01<00:52, 7512.70it/s]  2%|▏         | 9367/400000 [00:01<00:50, 7741.82it/s]  3%|▎         | 10290/400000 [00:01<00:47, 8133.48it/s]  3%|▎         | 11216/400000 [00:01<00:46, 8441.21it/s]  3%|▎         | 12070/400000 [00:01<00:46, 8317.44it/s]  3%|▎         | 12989/400000 [00:01<00:45, 8558.96it/s]  3%|▎         | 13920/400000 [00:01<00:44, 8769.52it/s]  4%|▎         | 14891/400000 [00:01<00:42, 9031.55it/s]  4%|▍         | 15856/400000 [00:01<00:41, 9206.95it/s]  4%|▍         | 16785/400000 [00:01<00:41, 9228.94it/s]  4%|▍         | 17722/400000 [00:02<00:41, 9270.32it/s]  5%|▍         | 18652/400000 [00:02<00:41, 9237.50it/s]  5%|▍         | 19578/400000 [00:02<00:41, 9211.68it/s]  5%|▌         | 20501/400000 [00:02<00:41, 9140.27it/s]  5%|▌         | 21418/400000 [00:02<00:41, 9146.11it/s]  6%|▌         | 22376/400000 [00:02<00:40, 9271.23it/s]  6%|▌         | 23305/400000 [00:02<00:40, 9190.86it/s]  6%|▌         | 24225/400000 [00:02<00:40, 9193.38it/s]  6%|▋         | 25160/400000 [00:02<00:40, 9237.71it/s]  7%|▋         | 26088/400000 [00:02<00:40, 9247.61it/s]  7%|▋         | 27028/400000 [00:03<00:40, 9291.38it/s]  7%|▋         | 27958/400000 [00:03<00:48, 7679.14it/s]  7%|▋         | 28900/400000 [00:03<00:45, 8128.60it/s]  7%|▋         | 29839/400000 [00:03<00:43, 8469.36it/s]  8%|▊         | 30786/400000 [00:03<00:42, 8743.70it/s]  8%|▊         | 31743/400000 [00:03<00:41, 8974.19it/s]  8%|▊         | 32679/400000 [00:03<00:40, 9083.97it/s]  8%|▊         | 33613/400000 [00:03<00:40, 9157.19it/s]  9%|▊         | 34563/400000 [00:03<00:39, 9257.11it/s]  9%|▉         | 35499/400000 [00:04<00:39, 9287.53it/s]  9%|▉         | 36433/400000 [00:04<00:39, 9198.91it/s]  9%|▉         | 37357/400000 [00:04<00:40, 9038.94it/s] 10%|▉         | 38286/400000 [00:04<00:39, 9110.23it/s] 10%|▉         | 39200/400000 [00:04<00:39, 9091.72it/s] 10%|█         | 40147/400000 [00:04<00:39, 9201.30it/s] 10%|█         | 41099/400000 [00:04<00:38, 9293.50it/s] 11%|█         | 42051/400000 [00:04<00:38, 9359.76it/s] 11%|█         | 42988/400000 [00:04<00:38, 9314.05it/s] 11%|█         | 43951/400000 [00:04<00:37, 9404.34it/s] 11%|█         | 44897/400000 [00:05<00:37, 9419.48it/s] 11%|█▏        | 45840/400000 [00:05<00:38, 9276.36it/s] 12%|█▏        | 46769/400000 [00:05<00:38, 9202.93it/s] 12%|█▏        | 47692/400000 [00:05<00:38, 9210.26it/s] 12%|█▏        | 48650/400000 [00:05<00:37, 9316.18it/s] 12%|█▏        | 49617/400000 [00:05<00:37, 9418.98it/s] 13%|█▎        | 50593/400000 [00:05<00:36, 9517.98it/s] 13%|█▎        | 51546/400000 [00:05<00:37, 9413.11it/s] 13%|█▎        | 52489/400000 [00:05<00:37, 9258.37it/s] 13%|█▎        | 53420/400000 [00:05<00:37, 9272.57it/s] 14%|█▎        | 54368/400000 [00:06<00:37, 9331.83it/s] 14%|█▍        | 55302/400000 [00:06<00:37, 9282.63it/s] 14%|█▍        | 56236/400000 [00:06<00:36, 9297.63it/s] 14%|█▍        | 57225/400000 [00:06<00:36, 9467.69it/s] 15%|█▍        | 58173/400000 [00:06<00:36, 9447.46it/s] 15%|█▍        | 59119/400000 [00:06<00:36, 9293.95it/s] 15%|█▌        | 60050/400000 [00:06<00:36, 9281.28it/s] 15%|█▌        | 60979/400000 [00:06<00:36, 9238.92it/s] 15%|█▌        | 61939/400000 [00:06<00:36, 9343.54it/s] 16%|█▌        | 62903/400000 [00:06<00:35, 9429.65it/s] 16%|█▌        | 63853/400000 [00:07<00:35, 9448.84it/s] 16%|█▌        | 64817/400000 [00:07<00:35, 9503.34it/s] 16%|█▋        | 65768/400000 [00:07<00:35, 9325.22it/s] 17%|█▋        | 66702/400000 [00:07<00:36, 9195.71it/s] 17%|█▋        | 67623/400000 [00:07<00:36, 9192.15it/s] 17%|█▋        | 68544/400000 [00:07<00:36, 9112.66it/s] 17%|█▋        | 69477/400000 [00:07<00:36, 9174.05it/s] 18%|█▊        | 70396/400000 [00:07<00:36, 9131.78it/s] 18%|█▊        | 71310/400000 [00:07<00:36, 8984.84it/s] 18%|█▊        | 72243/400000 [00:07<00:36, 9082.92it/s] 18%|█▊        | 73191/400000 [00:08<00:35, 9197.03it/s] 19%|█▊        | 74116/400000 [00:08<00:35, 9212.79it/s] 19%|█▉        | 75038/400000 [00:08<00:35, 9056.93it/s] 19%|█▉        | 75945/400000 [00:08<00:36, 8984.99it/s] 19%|█▉        | 76845/400000 [00:08<00:35, 8982.54it/s] 19%|█▉        | 77774/400000 [00:08<00:35, 9071.40it/s] 20%|█▉        | 78748/400000 [00:08<00:34, 9260.22it/s] 20%|█▉        | 79676/400000 [00:08<00:34, 9250.38it/s] 20%|██        | 80603/400000 [00:08<00:34, 9168.71it/s] 20%|██        | 81521/400000 [00:09<00:34, 9137.65it/s] 21%|██        | 82463/400000 [00:09<00:34, 9220.52it/s] 21%|██        | 83386/400000 [00:09<00:34, 9078.35it/s] 21%|██        | 84295/400000 [00:09<00:35, 8918.28it/s] 21%|██▏       | 85232/400000 [00:09<00:34, 9048.59it/s] 22%|██▏       | 86160/400000 [00:09<00:34, 9116.38it/s] 22%|██▏       | 87092/400000 [00:09<00:34, 9174.10it/s] 22%|██▏       | 88037/400000 [00:09<00:33, 9252.41it/s] 22%|██▏       | 88963/400000 [00:09<00:33, 9159.93it/s] 22%|██▏       | 89887/400000 [00:09<00:33, 9182.51it/s] 23%|██▎       | 90806/400000 [00:10<00:33, 9134.27it/s] 23%|██▎       | 91720/400000 [00:10<00:33, 9113.21it/s] 23%|██▎       | 92632/400000 [00:10<00:33, 9077.72it/s] 23%|██▎       | 93541/400000 [00:10<00:34, 8955.89it/s] 24%|██▎       | 94447/400000 [00:10<00:34, 8984.65it/s] 24%|██▍       | 95384/400000 [00:10<00:33, 9096.85it/s] 24%|██▍       | 96295/400000 [00:10<00:33, 9083.26it/s] 24%|██▍       | 97204/400000 [00:10<00:33, 9084.90it/s] 25%|██▍       | 98113/400000 [00:10<00:33, 9060.69it/s] 25%|██▍       | 99020/400000 [00:10<00:33, 8916.15it/s] 25%|██▍       | 99926/400000 [00:11<00:33, 8941.67it/s] 25%|██▌       | 100821/400000 [00:11<00:33, 8841.17it/s] 25%|██▌       | 101709/400000 [00:11<00:33, 8851.77it/s] 26%|██▌       | 102623/400000 [00:11<00:33, 8933.81it/s] 26%|██▌       | 103547/400000 [00:11<00:32, 9020.82it/s] 26%|██▌       | 104469/400000 [00:11<00:32, 9078.36it/s] 26%|██▋       | 105378/400000 [00:11<00:32, 9051.15it/s] 27%|██▋       | 106295/400000 [00:11<00:32, 9084.43it/s] 27%|██▋       | 107204/400000 [00:11<00:32, 9007.84it/s] 27%|██▋       | 108125/400000 [00:11<00:32, 9066.43it/s] 27%|██▋       | 109060/400000 [00:12<00:31, 9147.04it/s] 27%|██▋       | 109976/400000 [00:12<00:31, 9083.76it/s] 28%|██▊       | 110886/400000 [00:12<00:31, 9088.19it/s] 28%|██▊       | 111821/400000 [00:12<00:31, 9165.06it/s] 28%|██▊       | 112741/400000 [00:12<00:31, 9174.50it/s] 28%|██▊       | 113659/400000 [00:12<00:31, 9157.98it/s] 29%|██▊       | 114575/400000 [00:12<00:31, 9078.40it/s] 29%|██▉       | 115484/400000 [00:12<00:31, 9073.78it/s] 29%|██▉       | 116400/400000 [00:12<00:31, 9098.36it/s] 29%|██▉       | 117310/400000 [00:12<00:31, 8961.93it/s] 30%|██▉       | 118237/400000 [00:13<00:31, 9051.77it/s] 30%|██▉       | 119190/400000 [00:13<00:30, 9187.14it/s] 30%|███       | 120110/400000 [00:13<00:30, 9189.47it/s] 30%|███       | 121037/400000 [00:13<00:30, 9211.89it/s] 30%|███       | 121997/400000 [00:13<00:29, 9322.21it/s] 31%|███       | 122937/400000 [00:13<00:29, 9343.26it/s] 31%|███       | 123872/400000 [00:13<00:29, 9320.16it/s] 31%|███       | 124805/400000 [00:13<00:29, 9253.90it/s] 31%|███▏      | 125731/400000 [00:13<00:29, 9232.07it/s] 32%|███▏      | 126681/400000 [00:13<00:29, 9310.22it/s] 32%|███▏      | 127613/400000 [00:14<00:29, 9131.09it/s] 32%|███▏      | 128529/400000 [00:14<00:29, 9138.59it/s] 32%|███▏      | 129444/400000 [00:14<00:29, 9018.80it/s] 33%|███▎      | 130347/400000 [00:14<00:30, 8880.30it/s] 33%|███▎      | 131238/400000 [00:14<00:30, 8887.45it/s] 33%|███▎      | 132172/400000 [00:14<00:29, 9016.62it/s] 33%|███▎      | 133165/400000 [00:14<00:28, 9271.78it/s] 34%|███▎      | 134123/400000 [00:14<00:28, 9361.10it/s] 34%|███▍      | 135062/400000 [00:14<00:28, 9331.34it/s] 34%|███▍      | 136016/400000 [00:14<00:28, 9391.40it/s] 34%|███▍      | 136957/400000 [00:15<00:28, 9319.41it/s] 34%|███▍      | 137890/400000 [00:15<00:28, 9321.39it/s] 35%|███▍      | 138823/400000 [00:15<00:28, 9292.99it/s] 35%|███▍      | 139753/400000 [00:15<00:28, 9224.67it/s] 35%|███▌      | 140716/400000 [00:15<00:27, 9341.55it/s] 35%|███▌      | 141651/400000 [00:15<00:27, 9280.03it/s] 36%|███▌      | 142591/400000 [00:15<00:27, 9315.21it/s] 36%|███▌      | 143562/400000 [00:15<00:27, 9428.68it/s] 36%|███▌      | 144506/400000 [00:15<00:27, 9293.14it/s] 36%|███▋      | 145437/400000 [00:16<00:27, 9230.72it/s] 37%|███▋      | 146361/400000 [00:16<00:27, 9206.07it/s] 37%|███▋      | 147295/400000 [00:16<00:27, 9244.57it/s] 37%|███▋      | 148235/400000 [00:16<00:27, 9289.13it/s] 37%|███▋      | 149172/400000 [00:16<00:26, 9313.06it/s] 38%|███▊      | 150138/400000 [00:16<00:26, 9413.36it/s] 38%|███▊      | 151089/400000 [00:16<00:26, 9440.32it/s] 38%|███▊      | 152034/400000 [00:16<00:26, 9375.21it/s] 38%|███▊      | 152972/400000 [00:16<00:26, 9275.89it/s] 38%|███▊      | 153906/400000 [00:16<00:26, 9293.40it/s] 39%|███▊      | 154836/400000 [00:17<00:26, 9235.19it/s] 39%|███▉      | 155760/400000 [00:17<00:26, 9187.15it/s] 39%|███▉      | 156694/400000 [00:17<00:26, 9231.43it/s] 39%|███▉      | 157638/400000 [00:17<00:26, 9292.18it/s] 40%|███▉      | 158580/400000 [00:17<00:25, 9328.12it/s] 40%|███▉      | 159552/400000 [00:17<00:25, 9440.41it/s] 40%|████      | 160497/400000 [00:17<00:25, 9430.37it/s] 40%|████      | 161441/400000 [00:17<00:25, 9247.47it/s] 41%|████      | 162367/400000 [00:17<00:26, 9136.24it/s] 41%|████      | 163282/400000 [00:17<00:25, 9116.88it/s] 41%|████      | 164226/400000 [00:18<00:25, 9210.28it/s] 41%|████▏     | 165170/400000 [00:18<00:25, 9276.60it/s] 42%|████▏     | 166116/400000 [00:18<00:25, 9329.95it/s] 42%|████▏     | 167071/400000 [00:18<00:24, 9393.96it/s] 42%|████▏     | 168011/400000 [00:18<00:24, 9393.71it/s] 42%|████▏     | 168987/400000 [00:18<00:24, 9498.49it/s] 42%|████▏     | 169938/400000 [00:18<00:24, 9485.79it/s] 43%|████▎     | 170898/400000 [00:18<00:24, 9517.06it/s] 43%|████▎     | 171857/400000 [00:18<00:23, 9537.96it/s] 43%|████▎     | 172811/400000 [00:18<00:24, 9411.65it/s] 43%|████▎     | 173753/400000 [00:19<00:24, 9383.42it/s] 44%|████▎     | 174708/400000 [00:19<00:23, 9428.97it/s] 44%|████▍     | 175656/400000 [00:19<00:23, 9442.20it/s] 44%|████▍     | 176601/400000 [00:19<00:23, 9363.51it/s] 44%|████▍     | 177538/400000 [00:19<00:24, 9206.08it/s] 45%|████▍     | 178472/400000 [00:19<00:23, 9244.64it/s] 45%|████▍     | 179425/400000 [00:19<00:23, 9328.43it/s] 45%|████▌     | 180377/400000 [00:19<00:23, 9383.09it/s] 45%|████▌     | 181316/400000 [00:19<00:23, 9293.59it/s] 46%|████▌     | 182246/400000 [00:19<00:23, 9250.83it/s] 46%|████▌     | 183172/400000 [00:20<00:23, 9205.04it/s] 46%|████▌     | 184093/400000 [00:20<00:23, 9160.53it/s] 46%|████▋     | 185010/400000 [00:20<00:23, 9159.90it/s] 46%|████▋     | 185973/400000 [00:20<00:23, 9294.32it/s] 47%|████▋     | 186914/400000 [00:20<00:22, 9325.96it/s] 47%|████▋     | 187875/400000 [00:20<00:22, 9408.29it/s] 47%|████▋     | 188817/400000 [00:20<00:22, 9312.84it/s] 47%|████▋     | 189767/400000 [00:20<00:22, 9367.78it/s] 48%|████▊     | 190718/400000 [00:20<00:22, 9409.83it/s] 48%|████▊     | 191660/400000 [00:20<00:22, 9293.15it/s] 48%|████▊     | 192590/400000 [00:21<00:22, 9280.22it/s] 48%|████▊     | 193519/400000 [00:21<00:22, 9195.35it/s] 49%|████▊     | 194452/400000 [00:21<00:22, 9232.50it/s] 49%|████▉     | 195420/400000 [00:21<00:21, 9359.93it/s] 49%|████▉     | 196357/400000 [00:21<00:21, 9345.85it/s] 49%|████▉     | 197293/400000 [00:21<00:21, 9314.57it/s] 50%|████▉     | 198229/400000 [00:21<00:21, 9328.08it/s] 50%|████▉     | 199171/400000 [00:21<00:21, 9354.59it/s] 50%|█████     | 200135/400000 [00:21<00:21, 9436.27it/s] 50%|█████     | 201080/400000 [00:21<00:21, 9437.99it/s] 51%|█████     | 202025/400000 [00:22<00:20, 9431.44it/s] 51%|█████     | 202969/400000 [00:22<00:21, 9358.65it/s] 51%|█████     | 203906/400000 [00:22<00:21, 9310.57it/s] 51%|█████     | 204865/400000 [00:22<00:20, 9392.27it/s] 51%|█████▏    | 205805/400000 [00:22<00:20, 9394.28it/s] 52%|█████▏    | 206745/400000 [00:22<00:20, 9345.49it/s] 52%|█████▏    | 207704/400000 [00:22<00:20, 9415.56it/s] 52%|█████▏    | 208646/400000 [00:22<00:20, 9186.92it/s] 52%|█████▏    | 209567/400000 [00:22<00:20, 9069.01it/s] 53%|█████▎    | 210486/400000 [00:22<00:20, 9103.74it/s] 53%|█████▎    | 211458/400000 [00:23<00:20, 9279.97it/s] 53%|█████▎    | 212392/400000 [00:23<00:20, 9296.88it/s] 53%|█████▎    | 213325/400000 [00:23<00:20, 9304.41it/s] 54%|█████▎    | 214261/400000 [00:23<00:19, 9320.29it/s] 54%|█████▍    | 215220/400000 [00:23<00:19, 9396.96it/s] 54%|█████▍    | 216170/400000 [00:23<00:19, 9427.00it/s] 54%|█████▍    | 217114/400000 [00:23<00:19, 9414.36it/s] 55%|█████▍    | 218069/400000 [00:23<00:19, 9452.04it/s] 55%|█████▍    | 219015/400000 [00:23<00:19, 9167.01it/s] 55%|█████▍    | 219934/400000 [00:23<00:19, 9164.31it/s] 55%|█████▌    | 220892/400000 [00:24<00:19, 9284.37it/s] 55%|█████▌    | 221845/400000 [00:24<00:19, 9355.79it/s] 56%|█████▌    | 222782/400000 [00:24<00:19, 9298.63it/s] 56%|█████▌    | 223713/400000 [00:24<00:19, 9244.22it/s] 56%|█████▌    | 224656/400000 [00:24<00:18, 9298.75it/s] 56%|█████▋    | 225587/400000 [00:24<00:18, 9196.36it/s] 57%|█████▋    | 226508/400000 [00:24<00:18, 9179.34it/s] 57%|█████▋    | 227463/400000 [00:24<00:18, 9285.95it/s] 57%|█████▋    | 228393/400000 [00:24<00:18, 9227.30it/s] 57%|█████▋    | 229317/400000 [00:25<00:18, 9225.82it/s] 58%|█████▊    | 230240/400000 [00:25<00:18, 9214.08it/s] 58%|█████▊    | 231168/400000 [00:25<00:18, 9233.25it/s] 58%|█████▊    | 232106/400000 [00:25<00:18, 9275.29it/s] 58%|█████▊    | 233067/400000 [00:25<00:17, 9371.85it/s] 59%|█████▊    | 234048/400000 [00:25<00:17, 9497.20it/s] 59%|█████▊    | 234999/400000 [00:25<00:17, 9457.89it/s] 59%|█████▉    | 235971/400000 [00:25<00:17, 9532.38it/s] 59%|█████▉    | 236948/400000 [00:25<00:16, 9600.15it/s] 59%|█████▉    | 237909/400000 [00:25<00:16, 9601.36it/s] 60%|█████▉    | 238870/400000 [00:26<00:16, 9536.87it/s] 60%|█████▉    | 239825/400000 [00:26<00:16, 9523.62it/s] 60%|██████    | 240799/400000 [00:26<00:16, 9586.67it/s] 60%|██████    | 241758/400000 [00:26<00:16, 9443.00it/s] 61%|██████    | 242703/400000 [00:26<00:16, 9436.45it/s] 61%|██████    | 243648/400000 [00:26<00:16, 9410.58it/s] 61%|██████    | 244590/400000 [00:26<00:16, 9359.57it/s] 61%|██████▏   | 245527/400000 [00:26<00:16, 9259.24it/s] 62%|██████▏   | 246455/400000 [00:26<00:16, 9263.97it/s] 62%|██████▏   | 247415/400000 [00:26<00:16, 9360.51it/s] 62%|██████▏   | 248369/400000 [00:27<00:16, 9412.18it/s] 62%|██████▏   | 249311/400000 [00:27<00:16, 9352.39it/s] 63%|██████▎   | 250262/400000 [00:27<00:15, 9398.97it/s] 63%|██████▎   | 251203/400000 [00:27<00:15, 9304.07it/s] 63%|██████▎   | 252134/400000 [00:27<00:15, 9279.39it/s] 63%|██████▎   | 253072/400000 [00:27<00:15, 9308.73it/s] 64%|██████▎   | 254023/400000 [00:27<00:15, 9367.62it/s] 64%|██████▎   | 254964/400000 [00:27<00:15, 9379.87it/s] 64%|██████▍   | 255940/400000 [00:27<00:15, 9490.25it/s] 64%|██████▍   | 256890/400000 [00:27<00:15, 9381.09it/s] 64%|██████▍   | 257840/400000 [00:28<00:15, 9415.13it/s] 65%|██████▍   | 258803/400000 [00:28<00:14, 9477.14it/s] 65%|██████▍   | 259757/400000 [00:28<00:14, 9494.30it/s] 65%|██████▌   | 260707/400000 [00:28<00:14, 9329.95it/s] 65%|██████▌   | 261648/400000 [00:28<00:14, 9351.68it/s] 66%|██████▌   | 262610/400000 [00:28<00:14, 9428.50it/s] 66%|██████▌   | 263554/400000 [00:28<00:14, 9399.82it/s] 66%|██████▌   | 264525/400000 [00:28<00:14, 9490.35it/s] 66%|██████▋   | 265475/400000 [00:28<00:14, 9466.43it/s] 67%|██████▋   | 266423/400000 [00:28<00:14, 9463.52it/s] 67%|██████▋   | 267370/400000 [00:29<00:14, 9456.31it/s] 67%|██████▋   | 268316/400000 [00:29<00:14, 9300.61it/s] 67%|██████▋   | 269264/400000 [00:29<00:13, 9351.06it/s] 68%|██████▊   | 270207/400000 [00:29<00:13, 9372.49it/s] 68%|██████▊   | 271162/400000 [00:29<00:13, 9422.78it/s] 68%|██████▊   | 272108/400000 [00:29<00:13, 9433.37it/s] 68%|██████▊   | 273052/400000 [00:29<00:13, 9313.58it/s] 68%|██████▊   | 273984/400000 [00:29<00:13, 9276.41it/s] 69%|██████▊   | 274959/400000 [00:29<00:13, 9411.90it/s] 69%|██████▉   | 275905/400000 [00:29<00:13, 9425.25it/s] 69%|██████▉   | 276849/400000 [00:30<00:13, 9394.07it/s] 69%|██████▉   | 277789/400000 [00:30<00:13, 9356.16it/s] 70%|██████▉   | 278727/400000 [00:30<00:12, 9362.80it/s] 70%|██████▉   | 279664/400000 [00:30<00:12, 9308.37it/s] 70%|███████   | 280643/400000 [00:30<00:12, 9446.28it/s] 70%|███████   | 281618/400000 [00:30<00:12, 9534.03it/s] 71%|███████   | 282573/400000 [00:30<00:12, 9482.38it/s] 71%|███████   | 283543/400000 [00:30<00:12, 9544.85it/s] 71%|███████   | 284498/400000 [00:30<00:12, 9510.55it/s] 71%|███████▏  | 285477/400000 [00:30<00:11, 9591.25it/s] 72%|███████▏  | 286447/400000 [00:31<00:11, 9621.88it/s] 72%|███████▏  | 287410/400000 [00:31<00:11, 9562.56it/s] 72%|███████▏  | 288367/400000 [00:31<00:11, 9477.66it/s] 72%|███████▏  | 289316/400000 [00:31<00:11, 9393.56it/s] 73%|███████▎  | 290282/400000 [00:31<00:11, 9471.56it/s] 73%|███████▎  | 291249/400000 [00:31<00:11, 9529.66it/s] 73%|███████▎  | 292203/400000 [00:31<00:11, 9377.62it/s] 73%|███████▎  | 293181/400000 [00:31<00:11, 9492.52it/s] 74%|███████▎  | 294159/400000 [00:31<00:11, 9576.27it/s] 74%|███████▍  | 295118/400000 [00:31<00:10, 9571.01it/s] 74%|███████▍  | 296076/400000 [00:32<00:10, 9533.13it/s] 74%|███████▍  | 297030/400000 [00:32<00:11, 9275.93it/s] 74%|███████▍  | 297960/400000 [00:32<00:11, 9197.03it/s] 75%|███████▍  | 298886/400000 [00:32<00:10, 9213.36it/s] 75%|███████▍  | 299809/400000 [00:32<00:10, 9184.43it/s] 75%|███████▌  | 300729/400000 [00:32<00:10, 9178.47it/s] 75%|███████▌  | 301648/400000 [00:32<00:10, 9178.70it/s] 76%|███████▌  | 302589/400000 [00:32<00:10, 9245.58it/s] 76%|███████▌  | 303549/400000 [00:32<00:10, 9348.89it/s] 76%|███████▌  | 304485/400000 [00:32<00:10, 9300.08it/s] 76%|███████▋  | 305416/400000 [00:33<00:10, 9198.02it/s] 77%|███████▋  | 306351/400000 [00:33<00:10, 9242.00it/s] 77%|███████▋  | 307276/400000 [00:33<00:10, 9206.14it/s] 77%|███████▋  | 308223/400000 [00:33<00:09, 9282.10it/s] 77%|███████▋  | 309175/400000 [00:33<00:09, 9350.33it/s] 78%|███████▊  | 310152/400000 [00:33<00:09, 9470.72it/s] 78%|███████▊  | 311110/400000 [00:33<00:09, 9501.71it/s] 78%|███████▊  | 312062/400000 [00:33<00:09, 9505.14it/s] 78%|███████▊  | 313013/400000 [00:33<00:09, 9488.48it/s] 78%|███████▊  | 313976/400000 [00:34<00:09, 9529.75it/s] 79%|███████▊  | 314950/400000 [00:34<00:08, 9591.28it/s] 79%|███████▉  | 315910/400000 [00:34<00:08, 9495.68it/s] 79%|███████▉  | 316860/400000 [00:34<00:08, 9493.90it/s] 79%|███████▉  | 317810/400000 [00:34<00:08, 9411.69it/s] 80%|███████▉  | 318754/400000 [00:34<00:08, 9419.99it/s] 80%|███████▉  | 319716/400000 [00:34<00:08, 9477.20it/s] 80%|████████  | 320664/400000 [00:34<00:08, 9416.20it/s] 80%|████████  | 321613/400000 [00:34<00:08, 9436.56it/s] 81%|████████  | 322598/400000 [00:34<00:08, 9554.24it/s] 81%|████████  | 323554/400000 [00:35<00:08, 9508.51it/s] 81%|████████  | 324506/400000 [00:35<00:08, 9335.04it/s] 81%|████████▏ | 325441/400000 [00:35<00:08, 8899.31it/s] 82%|████████▏ | 326360/400000 [00:35<00:08, 8983.12it/s] 82%|████████▏ | 327274/400000 [00:35<00:08, 9027.45it/s] 82%|████████▏ | 328211/400000 [00:35<00:07, 9126.83it/s] 82%|████████▏ | 329181/400000 [00:35<00:07, 9289.14it/s] 83%|████████▎ | 330113/400000 [00:35<00:07, 9277.68it/s] 83%|████████▎ | 331081/400000 [00:35<00:07, 9393.27it/s] 83%|████████▎ | 332031/400000 [00:35<00:07, 9422.30it/s] 83%|████████▎ | 332975/400000 [00:36<00:07, 9338.56it/s] 83%|████████▎ | 333910/400000 [00:36<00:07, 9137.42it/s] 84%|████████▎ | 334828/400000 [00:36<00:07, 9150.10it/s] 84%|████████▍ | 335774/400000 [00:36<00:06, 9240.03it/s] 84%|████████▍ | 336699/400000 [00:36<00:06, 9226.84it/s] 84%|████████▍ | 337626/400000 [00:36<00:06, 9236.72it/s] 85%|████████▍ | 338564/400000 [00:36<00:06, 9278.29it/s] 85%|████████▍ | 339493/400000 [00:36<00:06, 9235.33it/s] 85%|████████▌ | 340482/400000 [00:36<00:06, 9422.25it/s] 85%|████████▌ | 341470/400000 [00:36<00:06, 9553.03it/s] 86%|████████▌ | 342438/400000 [00:37<00:06, 9590.12it/s] 86%|████████▌ | 343398/400000 [00:37<00:05, 9519.18it/s] 86%|████████▌ | 344351/400000 [00:37<00:05, 9493.88it/s] 86%|████████▋ | 345306/400000 [00:37<00:05, 9509.42it/s] 87%|████████▋ | 346258/400000 [00:37<00:05, 9449.02it/s] 87%|████████▋ | 347204/400000 [00:37<00:05, 9432.92it/s] 87%|████████▋ | 348148/400000 [00:37<00:05, 9383.68it/s] 87%|████████▋ | 349087/400000 [00:37<00:05, 9249.89it/s] 88%|████████▊ | 350013/400000 [00:37<00:05, 9169.50it/s] 88%|████████▊ | 350956/400000 [00:37<00:05, 9245.33it/s] 88%|████████▊ | 351883/400000 [00:38<00:05, 9249.98it/s] 88%|████████▊ | 352809/400000 [00:38<00:05, 9215.97it/s] 88%|████████▊ | 353731/400000 [00:38<00:05, 9189.56it/s] 89%|████████▊ | 354651/400000 [00:38<00:04, 9099.56it/s] 89%|████████▉ | 355562/400000 [00:38<00:04, 9092.72it/s] 89%|████████▉ | 356507/400000 [00:38<00:04, 9196.90it/s] 89%|████████▉ | 357465/400000 [00:38<00:04, 9306.69it/s] 90%|████████▉ | 358397/400000 [00:38<00:04, 9252.67it/s] 90%|████████▉ | 359344/400000 [00:38<00:04, 9315.06it/s] 90%|█████████ | 360276/400000 [00:38<00:04, 9294.05it/s] 90%|█████████ | 361231/400000 [00:39<00:04, 9368.73it/s] 91%|█████████ | 362198/400000 [00:39<00:03, 9455.05it/s] 91%|█████████ | 363144/400000 [00:39<00:03, 9379.78it/s] 91%|█████████ | 364114/400000 [00:39<00:03, 9472.97it/s] 91%|█████████▏| 365074/400000 [00:39<00:03, 9508.93it/s] 92%|█████████▏| 366033/400000 [00:39<00:03, 9532.13it/s] 92%|█████████▏| 366993/400000 [00:39<00:03, 9551.37it/s] 92%|█████████▏| 367949/400000 [00:39<00:03, 9376.26it/s] 92%|█████████▏| 368888/400000 [00:39<00:03, 9147.94it/s] 92%|█████████▏| 369855/400000 [00:39<00:03, 9296.33it/s] 93%|█████████▎| 370789/400000 [00:40<00:03, 9307.71it/s] 93%|█████████▎| 371722/400000 [00:40<00:03, 9192.24it/s] 93%|█████████▎| 372643/400000 [00:40<00:03, 9083.78it/s] 93%|█████████▎| 373553/400000 [00:40<00:02, 9080.65it/s] 94%|█████████▎| 374490/400000 [00:40<00:02, 9164.23it/s] 94%|█████████▍| 375423/400000 [00:40<00:02, 9212.69it/s] 94%|█████████▍| 376362/400000 [00:40<00:02, 9263.70it/s] 94%|█████████▍| 377289/400000 [00:40<00:02, 9203.31it/s] 95%|█████████▍| 378230/400000 [00:40<00:02, 9262.95it/s] 95%|█████████▍| 379181/400000 [00:41<00:02, 9335.38it/s] 95%|█████████▌| 380115/400000 [00:41<00:02, 9221.65it/s] 95%|█████████▌| 381038/400000 [00:41<00:02, 9162.48it/s] 95%|█████████▌| 381964/400000 [00:41<00:01, 9189.53it/s] 96%|█████████▌| 382904/400000 [00:41<00:01, 9251.31it/s] 96%|█████████▌| 383830/400000 [00:41<00:01, 9194.66it/s] 96%|█████████▌| 384750/400000 [00:41<00:01, 9194.69it/s] 96%|█████████▋| 385685/400000 [00:41<00:01, 9238.91it/s] 97%|█████████▋| 386613/400000 [00:41<00:01, 9248.80it/s] 97%|█████████▋| 387572/400000 [00:41<00:01, 9346.24it/s] 97%|█████████▋| 388507/400000 [00:42<00:01, 9304.41it/s] 97%|█████████▋| 389438/400000 [00:42<00:01, 9295.35it/s] 98%|█████████▊| 390378/400000 [00:42<00:01, 9325.95it/s] 98%|█████████▊| 391311/400000 [00:42<00:00, 9308.78it/s] 98%|█████████▊| 392260/400000 [00:42<00:00, 9362.06it/s] 98%|█████████▊| 393228/400000 [00:42<00:00, 9453.27it/s] 99%|█████████▊| 394174/400000 [00:42<00:00, 9425.46it/s] 99%|█████████▉| 395181/400000 [00:42<00:00, 9609.74it/s] 99%|█████████▉| 396144/400000 [00:42<00:00, 9563.40it/s] 99%|█████████▉| 397102/400000 [00:42<00:00, 9520.39it/s]100%|█████████▉| 398055/400000 [00:43<00:00, 9427.61it/s]100%|█████████▉| 398999/400000 [00:43<00:00, 9425.29it/s]100%|█████████▉| 399949/400000 [00:43<00:00, 9447.52it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9252.18it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f68a6b464e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011547703002825998 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.011734882724723688 	 Accuracy: 47

  model saves at 47% accuracy 

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

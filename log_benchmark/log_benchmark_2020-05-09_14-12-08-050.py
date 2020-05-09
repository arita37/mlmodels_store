
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_benchmark GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/0732dc2301f5134d4ff715e34818ffb83581739d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '0732dc2301f5134d4ff715e34818ffb83581739d', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/0732dc2301f5134d4ff715e34818ffb83581739d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/0732dc2301f5134d4ff715e34818ffb83581739d

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6c10e19f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:12:20.836318
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 14:12:20.839412
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 14:12:20.842185
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 14:12:20.844940
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6c19fe3ac8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 352395.8438
Epoch 2/10

1/1 [==============================] - 0s 84ms/step - loss: 230954.1250
Epoch 3/10

1/1 [==============================] - 0s 83ms/step - loss: 129234.4375
Epoch 4/10

1/1 [==============================] - 0s 88ms/step - loss: 65247.2773
Epoch 5/10

1/1 [==============================] - 0s 82ms/step - loss: 35497.4805
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 21399.5566
Epoch 7/10

1/1 [==============================] - 0s 84ms/step - loss: 14131.1299
Epoch 8/10

1/1 [==============================] - 0s 87ms/step - loss: 10045.4688
Epoch 9/10

1/1 [==============================] - 0s 82ms/step - loss: 7573.2529
Epoch 10/10

1/1 [==============================] - 0s 79ms/step - loss: 5993.1074

  #### Inference Need return ypred, ytrue ######################### 
[[-4.34259772e-02 -1.48485768e+00 -4.70793366e-01 -1.35846448e+00
   6.46657884e-01  7.08631158e-01 -7.05211997e-01 -7.47150660e-01
  -9.54497337e-01  2.66351342e-01 -9.58698213e-01 -1.21279371e+00
   3.26781154e-01  8.12227607e-01  6.20155156e-01 -2.20863640e-01
   1.40446305e+00  4.21198010e-01  1.49237502e+00  4.87728894e-01
  -3.68383825e-02  1.23032749e+00  6.80079877e-01 -7.02279508e-01
   8.30221593e-01  8.09349835e-01 -1.22521818e-01 -2.68748379e+00
  -2.87205219e-01 -5.27517259e-01 -6.52167499e-02  7.93453753e-01
   5.83492219e-01  7.59610534e-01  8.19020271e-01 -3.42257410e-01
  -4.87467080e-01  6.84395671e-01  6.32128477e-01  1.45823872e+00
   4.46083724e-01 -7.12366164e-01  1.10863400e+00  5.97691000e-01
  -1.88129449e+00 -1.08885217e+00  2.35549283e+00  9.25881386e-01
   9.89493430e-01  1.60396731e+00 -9.58251595e-01  1.28542936e+00
   1.13618958e+00  1.87783825e+00  5.58149099e-01  1.07967734e+00
   1.94681275e+00 -3.41256946e-01 -2.19908714e-01 -1.01037621e-02
   9.76877570e-01  1.04324841e+00 -2.30658841e+00 -1.82634234e-01
  -2.10130429e+00  8.52317214e-01 -2.95779228e-01  1.21264315e+00
  -1.63068664e+00  5.98171949e-02 -1.31355596e+00 -9.87673223e-01
   1.31650925e+00  1.08357418e+00  3.87793273e-01  9.50787961e-01
   9.68590140e-01 -1.74256420e+00 -1.06780243e+00 -2.57430768e+00
   2.08575904e-01  1.67359650e+00  1.93201780e+00  2.36273050e+00
   1.11081636e+00 -1.59891391e+00  1.76204133e+00 -5.01733840e-01
  -6.02300942e-01 -1.05688453e+00  1.54598951e+00  9.15106595e-01
   1.81400132e+00 -1.09731281e+00  1.61304259e+00  2.28110313e+00
  -9.50387597e-01  6.63808107e-01  1.27642393e-01 -6.29732966e-01
  -9.27684665e-01 -1.95863891e+00  2.03875482e-01  2.00338602e+00
  -9.67113137e-01 -3.34229082e-01  7.35276937e-01 -1.37498975e-01
   9.31285501e-01  5.65718770e-01 -2.99073458e-01  4.49612439e-01
  -1.50387394e+00 -1.70298409e+00  1.23789132e+00 -5.42071760e-01
   2.70778596e-01  4.22673076e-01  1.91330910e-04 -1.12152398e+00
  -3.71363699e-01  1.10117474e+01  9.03123951e+00  6.84974289e+00
   7.25120258e+00  6.29809046e+00  8.10823917e+00  7.74867725e+00
   7.96205187e+00  8.01394463e+00  7.44139671e+00  7.48180103e+00
   8.07981586e+00  7.12797403e+00  7.09676647e+00  8.31932545e+00
   1.01127434e+01  5.69914627e+00  9.20712280e+00  7.16634798e+00
   9.50141335e+00  8.99613285e+00  8.15564060e+00  9.32816410e+00
   9.37021446e+00  9.29522419e+00  7.78973436e+00  9.75184631e+00
   9.06013584e+00  1.00361862e+01  7.98149204e+00  7.87625122e+00
   9.76893330e+00  1.01147099e+01  1.09867840e+01  8.60036945e+00
   8.16516876e+00  8.39168930e+00  1.04513988e+01  7.37104082e+00
   8.32009029e+00  1.11166363e+01  1.00452967e+01  9.27433872e+00
   7.62404203e+00  8.83082008e+00  7.69718266e+00  8.40095806e+00
   9.97413731e+00  9.14503193e+00  6.85883570e+00  9.01954079e+00
   7.82401419e+00  9.74947357e+00  8.66661453e+00  1.05274315e+01
   9.61431599e+00  8.74030590e+00  9.20822811e+00  8.44887924e+00
   1.89640892e+00  2.09637928e+00  7.62476504e-01  1.25745225e+00
   8.89925122e-01  2.30492830e-01  6.37769938e-01  2.62456274e+00
   1.86928368e+00  2.06847239e+00  6.35388792e-01  5.24812818e-01
   1.38207424e+00  3.03470612e-01  5.51154554e-01  1.68445826e-01
   9.61300731e-02  2.18487334e+00  1.62559581e+00  1.65170658e+00
   9.68934774e-01  1.24007034e+00  2.75327158e+00  3.64006877e-01
   7.01490045e-01  1.56613362e+00  3.21934521e-01  2.08523655e+00
   1.33229637e+00  1.76995516e+00  1.59316516e+00  1.71496892e+00
   2.63589859e-01  1.37195981e+00  2.10186720e-01  2.89425993e+00
   2.69216347e+00  7.27217555e-01  3.88887882e-01  7.13740587e-02
   2.92017698e-01  1.70303166e+00  1.46269238e+00  1.43368149e+00
   2.39133978e+00  2.63047576e-01  3.12069058e-01  2.08955193e+00
   2.04041338e+00  1.84809446e+00  3.66768408e+00  9.91853476e-01
   3.83313274e+00  5.19696951e-01  2.18681335e+00  1.18226671e+00
   1.32899237e+00  1.61159515e-01  1.09718084e-01  1.25049996e+00
   1.33126259e-01  1.22767520e+00  1.64296472e+00  3.75200510e-01
   1.17953515e+00  1.58925414e-01  2.81878090e+00  1.24867189e+00
   3.34205294e+00  1.57876921e+00  1.16604877e+00  5.12628794e-01
   3.96472871e-01  1.05268466e+00  3.72227859e+00  1.31093824e+00
   5.72346926e-01  1.14948392e-01  7.56074727e-01  1.84720039e-01
   2.58130455e+00  4.68787432e-01  2.17795610e+00  1.94824386e+00
   1.00986719e+00  3.10083199e+00  1.21971822e+00  1.62579441e+00
   4.17084217e-01  1.49826407e-01  1.07008648e+00  6.79002821e-01
   2.44608784e+00  1.89483702e+00  2.28654099e+00  2.31394589e-01
   5.06509602e-01  1.83868146e+00  5.13717294e-01  5.34238040e-01
   4.77596045e-01  1.05374694e-01  2.60862589e+00  1.56462574e+00
   1.95431101e+00  2.27281094e+00  4.35031414e-01  2.76296854e+00
   2.60447264e-01  3.19034386e+00  1.61389625e+00  3.36341798e-01
   2.52868271e+00  9.74562168e-02  2.39613581e+00  2.37778068e+00
   4.85537529e-01  3.19272566e+00  1.75308537e+00  1.52874088e+00
   4.31534052e-01  8.96298885e+00  8.91038704e+00  7.63756895e+00
   7.90056849e+00  8.33089542e+00  6.69348764e+00  9.32217216e+00
   8.53550816e+00  8.95827866e+00  8.45217896e+00  8.78146648e+00
   8.10146618e+00  8.31879330e+00  8.69377708e+00  8.20805740e+00
   9.69480228e+00  7.52202892e+00  6.97114706e+00  8.31528568e+00
   9.05227375e+00  9.69571495e+00  1.11135416e+01  6.79318953e+00
   9.64490700e+00  9.89921284e+00  9.51626682e+00  6.83731794e+00
   7.99153042e+00  6.88941765e+00  8.26188660e+00  9.63997078e+00
   9.62829685e+00  8.08180904e+00  6.73276329e+00  9.75104332e+00
   1.02112846e+01  9.00273800e+00  8.54549122e+00  1.09329576e+01
   9.51284027e+00  8.11983967e+00  9.10098934e+00  9.67288685e+00
   9.28058434e+00  1.00376797e+01  8.62882614e+00  8.95760059e+00
   9.35500336e+00  9.99749184e+00  8.21620274e+00  9.25884533e+00
   8.41787148e+00  9.91152954e+00  7.37641335e+00  8.41828632e+00
   8.83149338e+00  9.09235287e+00  9.01389790e+00  1.00236206e+01
  -8.38475227e+00 -1.38664699e+00  8.65438938e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:12:28.141160
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.7383
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 14:12:28.144691
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8806.73
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 14:12:28.147667
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.7414
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 14:12:28.150265
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -787.705
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140101757248512
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140100547277600
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140100547278104
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140100547278608
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140100547279112
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140100547279616

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6c182f1080> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.542916
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.503072
grad_step = 000002, loss = 0.469014
grad_step = 000003, loss = 0.432850
grad_step = 000004, loss = 0.393493
grad_step = 000005, loss = 0.358026
grad_step = 000006, loss = 0.336407
grad_step = 000007, loss = 0.327206
grad_step = 000008, loss = 0.315558
grad_step = 000009, loss = 0.295687
grad_step = 000010, loss = 0.279319
grad_step = 000011, loss = 0.268219
grad_step = 000012, loss = 0.260067
grad_step = 000013, loss = 0.251726
grad_step = 000014, loss = 0.242118
grad_step = 000015, loss = 0.231376
grad_step = 000016, loss = 0.220127
grad_step = 000017, loss = 0.209270
grad_step = 000018, loss = 0.199511
grad_step = 000019, loss = 0.190684
grad_step = 000020, loss = 0.181971
grad_step = 000021, loss = 0.173003
grad_step = 000022, loss = 0.164111
grad_step = 000023, loss = 0.155759
grad_step = 000024, loss = 0.147972
grad_step = 000025, loss = 0.140401
grad_step = 000026, loss = 0.132774
grad_step = 000027, loss = 0.125201
grad_step = 000028, loss = 0.118006
grad_step = 000029, loss = 0.111400
grad_step = 000030, loss = 0.105229
grad_step = 000031, loss = 0.098977
grad_step = 000032, loss = 0.092377
grad_step = 000033, loss = 0.086155
grad_step = 000034, loss = 0.080761
grad_step = 000035, loss = 0.075691
grad_step = 000036, loss = 0.070514
grad_step = 000037, loss = 0.065460
grad_step = 000038, loss = 0.060851
grad_step = 000039, loss = 0.056457
grad_step = 000040, loss = 0.052088
grad_step = 000041, loss = 0.047987
grad_step = 000042, loss = 0.044174
grad_step = 000043, loss = 0.040224
grad_step = 000044, loss = 0.036594
grad_step = 000045, loss = 0.033357
grad_step = 000046, loss = 0.030404
grad_step = 000047, loss = 0.027850
grad_step = 000048, loss = 0.025467
grad_step = 000049, loss = 0.022958
grad_step = 000050, loss = 0.020582
grad_step = 000051, loss = 0.018454
grad_step = 000052, loss = 0.016562
grad_step = 000053, loss = 0.014947
grad_step = 000054, loss = 0.013377
grad_step = 000055, loss = 0.011917
grad_step = 000056, loss = 0.010659
grad_step = 000057, loss = 0.009457
grad_step = 000058, loss = 0.008429
grad_step = 000059, loss = 0.007529
grad_step = 000060, loss = 0.006748
grad_step = 000061, loss = 0.006057
grad_step = 000062, loss = 0.005406
grad_step = 000063, loss = 0.004907
grad_step = 000064, loss = 0.004458
grad_step = 000065, loss = 0.004078
grad_step = 000066, loss = 0.003769
grad_step = 000067, loss = 0.003529
grad_step = 000068, loss = 0.003339
grad_step = 000069, loss = 0.003158
grad_step = 000070, loss = 0.003043
grad_step = 000071, loss = 0.002940
grad_step = 000072, loss = 0.002862
grad_step = 000073, loss = 0.002797
grad_step = 000074, loss = 0.002772
grad_step = 000075, loss = 0.002744
grad_step = 000076, loss = 0.002718
grad_step = 000077, loss = 0.002696
grad_step = 000078, loss = 0.002681
grad_step = 000079, loss = 0.002665
grad_step = 000080, loss = 0.002652
grad_step = 000081, loss = 0.002646
grad_step = 000082, loss = 0.002632
grad_step = 000083, loss = 0.002616
grad_step = 000084, loss = 0.002600
grad_step = 000085, loss = 0.002586
grad_step = 000086, loss = 0.002571
grad_step = 000087, loss = 0.002561
grad_step = 000088, loss = 0.002548
grad_step = 000089, loss = 0.002534
grad_step = 000090, loss = 0.002519
grad_step = 000091, loss = 0.002507
grad_step = 000092, loss = 0.002493
grad_step = 000093, loss = 0.002480
grad_step = 000094, loss = 0.002467
grad_step = 000095, loss = 0.002453
grad_step = 000096, loss = 0.002439
grad_step = 000097, loss = 0.002427
grad_step = 000098, loss = 0.002415
grad_step = 000099, loss = 0.002403
grad_step = 000100, loss = 0.002392
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002381
grad_step = 000102, loss = 0.002372
grad_step = 000103, loss = 0.002364
grad_step = 000104, loss = 0.002357
grad_step = 000105, loss = 0.002349
grad_step = 000106, loss = 0.002343
grad_step = 000107, loss = 0.002337
grad_step = 000108, loss = 0.002332
grad_step = 000109, loss = 0.002328
grad_step = 000110, loss = 0.002324
grad_step = 000111, loss = 0.002320
grad_step = 000112, loss = 0.002316
grad_step = 000113, loss = 0.002313
grad_step = 000114, loss = 0.002311
grad_step = 000115, loss = 0.002308
grad_step = 000116, loss = 0.002306
grad_step = 000117, loss = 0.002303
grad_step = 000118, loss = 0.002301
grad_step = 000119, loss = 0.002299
grad_step = 000120, loss = 0.002298
grad_step = 000121, loss = 0.002296
grad_step = 000122, loss = 0.002294
grad_step = 000123, loss = 0.002292
grad_step = 000124, loss = 0.002291
grad_step = 000125, loss = 0.002289
grad_step = 000126, loss = 0.002287
grad_step = 000127, loss = 0.002286
grad_step = 000128, loss = 0.002284
grad_step = 000129, loss = 0.002282
grad_step = 000130, loss = 0.002281
grad_step = 000131, loss = 0.002279
grad_step = 000132, loss = 0.002277
grad_step = 000133, loss = 0.002276
grad_step = 000134, loss = 0.002274
grad_step = 000135, loss = 0.002272
grad_step = 000136, loss = 0.002271
grad_step = 000137, loss = 0.002269
grad_step = 000138, loss = 0.002268
grad_step = 000139, loss = 0.002266
grad_step = 000140, loss = 0.002264
grad_step = 000141, loss = 0.002263
grad_step = 000142, loss = 0.002261
grad_step = 000143, loss = 0.002259
grad_step = 000144, loss = 0.002258
grad_step = 000145, loss = 0.002256
grad_step = 000146, loss = 0.002254
grad_step = 000147, loss = 0.002253
grad_step = 000148, loss = 0.002251
grad_step = 000149, loss = 0.002249
grad_step = 000150, loss = 0.002248
grad_step = 000151, loss = 0.002246
grad_step = 000152, loss = 0.002244
grad_step = 000153, loss = 0.002242
grad_step = 000154, loss = 0.002241
grad_step = 000155, loss = 0.002239
grad_step = 000156, loss = 0.002237
grad_step = 000157, loss = 0.002236
grad_step = 000158, loss = 0.002234
grad_step = 000159, loss = 0.002232
grad_step = 000160, loss = 0.002231
grad_step = 000161, loss = 0.002229
grad_step = 000162, loss = 0.002227
grad_step = 000163, loss = 0.002226
grad_step = 000164, loss = 0.002224
grad_step = 000165, loss = 0.002222
grad_step = 000166, loss = 0.002221
grad_step = 000167, loss = 0.002219
grad_step = 000168, loss = 0.002217
grad_step = 000169, loss = 0.002215
grad_step = 000170, loss = 0.002214
grad_step = 000171, loss = 0.002212
grad_step = 000172, loss = 0.002210
grad_step = 000173, loss = 0.002208
grad_step = 000174, loss = 0.002207
grad_step = 000175, loss = 0.002205
grad_step = 000176, loss = 0.002203
grad_step = 000177, loss = 0.002201
grad_step = 000178, loss = 0.002199
grad_step = 000179, loss = 0.002198
grad_step = 000180, loss = 0.002196
grad_step = 000181, loss = 0.002194
grad_step = 000182, loss = 0.002192
grad_step = 000183, loss = 0.002191
grad_step = 000184, loss = 0.002189
grad_step = 000185, loss = 0.002187
grad_step = 000186, loss = 0.002185
grad_step = 000187, loss = 0.002183
grad_step = 000188, loss = 0.002182
grad_step = 000189, loss = 0.002180
grad_step = 000190, loss = 0.002178
grad_step = 000191, loss = 0.002176
grad_step = 000192, loss = 0.002174
grad_step = 000193, loss = 0.002172
grad_step = 000194, loss = 0.002171
grad_step = 000195, loss = 0.002172
grad_step = 000196, loss = 0.002187
grad_step = 000197, loss = 0.002243
grad_step = 000198, loss = 0.002348
grad_step = 000199, loss = 0.002470
grad_step = 000200, loss = 0.002294
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002160
grad_step = 000202, loss = 0.002258
grad_step = 000203, loss = 0.002290
grad_step = 000204, loss = 0.002184
grad_step = 000205, loss = 0.002178
grad_step = 000206, loss = 0.002248
grad_step = 000207, loss = 0.002197
grad_step = 000208, loss = 0.002154
grad_step = 000209, loss = 0.002216
grad_step = 000210, loss = 0.002193
grad_step = 000211, loss = 0.002145
grad_step = 000212, loss = 0.002192
grad_step = 000213, loss = 0.002188
grad_step = 000214, loss = 0.002139
grad_step = 000215, loss = 0.002172
grad_step = 000216, loss = 0.002177
grad_step = 000217, loss = 0.002135
grad_step = 000218, loss = 0.002155
grad_step = 000219, loss = 0.002165
grad_step = 000220, loss = 0.002130
grad_step = 000221, loss = 0.002141
grad_step = 000222, loss = 0.002153
grad_step = 000223, loss = 0.002126
grad_step = 000224, loss = 0.002129
grad_step = 000225, loss = 0.002141
grad_step = 000226, loss = 0.002121
grad_step = 000227, loss = 0.002118
grad_step = 000228, loss = 0.002129
grad_step = 000229, loss = 0.002116
grad_step = 000230, loss = 0.002108
grad_step = 000231, loss = 0.002116
grad_step = 000232, loss = 0.002110
grad_step = 000233, loss = 0.002100
grad_step = 000234, loss = 0.002103
grad_step = 000235, loss = 0.002103
grad_step = 000236, loss = 0.002094
grad_step = 000237, loss = 0.002092
grad_step = 000238, loss = 0.002093
grad_step = 000239, loss = 0.002089
grad_step = 000240, loss = 0.002084
grad_step = 000241, loss = 0.002082
grad_step = 000242, loss = 0.002082
grad_step = 000243, loss = 0.002076
grad_step = 000244, loss = 0.002073
grad_step = 000245, loss = 0.002072
grad_step = 000246, loss = 0.002069
grad_step = 000247, loss = 0.002064
grad_step = 000248, loss = 0.002061
grad_step = 000249, loss = 0.002060
grad_step = 000250, loss = 0.002057
grad_step = 000251, loss = 0.002052
grad_step = 000252, loss = 0.002050
grad_step = 000253, loss = 0.002048
grad_step = 000254, loss = 0.002046
grad_step = 000255, loss = 0.002044
grad_step = 000256, loss = 0.002046
grad_step = 000257, loss = 0.002043
grad_step = 000258, loss = 0.002037
grad_step = 000259, loss = 0.002031
grad_step = 000260, loss = 0.002027
grad_step = 000261, loss = 0.002026
grad_step = 000262, loss = 0.002025
grad_step = 000263, loss = 0.002025
grad_step = 000264, loss = 0.002022
grad_step = 000265, loss = 0.002018
grad_step = 000266, loss = 0.002012
grad_step = 000267, loss = 0.002007
grad_step = 000268, loss = 0.002003
grad_step = 000269, loss = 0.002000
grad_step = 000270, loss = 0.001998
grad_step = 000271, loss = 0.001996
grad_step = 000272, loss = 0.001995
grad_step = 000273, loss = 0.001999
grad_step = 000274, loss = 0.002012
grad_step = 000275, loss = 0.002048
grad_step = 000276, loss = 0.002086
grad_step = 000277, loss = 0.002118
grad_step = 000278, loss = 0.002076
grad_step = 000279, loss = 0.002024
grad_step = 000280, loss = 0.001980
grad_step = 000281, loss = 0.001985
grad_step = 000282, loss = 0.002023
grad_step = 000283, loss = 0.002021
grad_step = 000284, loss = 0.001981
grad_step = 000285, loss = 0.001956
grad_step = 000286, loss = 0.001975
grad_step = 000287, loss = 0.001994
grad_step = 000288, loss = 0.001974
grad_step = 000289, loss = 0.001951
grad_step = 000290, loss = 0.001949
grad_step = 000291, loss = 0.001951
grad_step = 000292, loss = 0.001947
grad_step = 000293, loss = 0.001946
grad_step = 000294, loss = 0.001953
grad_step = 000295, loss = 0.001944
grad_step = 000296, loss = 0.001928
grad_step = 000297, loss = 0.001917
grad_step = 000298, loss = 0.001919
grad_step = 000299, loss = 0.001920
grad_step = 000300, loss = 0.001915
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001914
grad_step = 000302, loss = 0.001918
grad_step = 000303, loss = 0.001922
grad_step = 000304, loss = 0.001919
grad_step = 000305, loss = 0.001922
grad_step = 000306, loss = 0.001925
grad_step = 000307, loss = 0.001930
grad_step = 000308, loss = 0.001918
grad_step = 000309, loss = 0.001911
grad_step = 000310, loss = 0.001899
grad_step = 000311, loss = 0.001887
grad_step = 000312, loss = 0.001872
grad_step = 000313, loss = 0.001864
grad_step = 000314, loss = 0.001862
grad_step = 000315, loss = 0.001861
grad_step = 000316, loss = 0.001861
grad_step = 000317, loss = 0.001865
grad_step = 000318, loss = 0.001877
grad_step = 000319, loss = 0.001891
grad_step = 000320, loss = 0.001918
grad_step = 000321, loss = 0.001940
grad_step = 000322, loss = 0.001974
grad_step = 000323, loss = 0.001952
grad_step = 000324, loss = 0.001916
grad_step = 000325, loss = 0.001854
grad_step = 000326, loss = 0.001826
grad_step = 000327, loss = 0.001839
grad_step = 000328, loss = 0.001866
grad_step = 000329, loss = 0.001877
grad_step = 000330, loss = 0.001848
grad_step = 000331, loss = 0.001820
grad_step = 000332, loss = 0.001815
grad_step = 000333, loss = 0.001831
grad_step = 000334, loss = 0.001843
grad_step = 000335, loss = 0.001833
grad_step = 000336, loss = 0.001815
grad_step = 000337, loss = 0.001804
grad_step = 000338, loss = 0.001804
grad_step = 000339, loss = 0.001811
grad_step = 000340, loss = 0.001815
grad_step = 000341, loss = 0.001816
grad_step = 000342, loss = 0.001809
grad_step = 000343, loss = 0.001799
grad_step = 000344, loss = 0.001792
grad_step = 000345, loss = 0.001789
grad_step = 000346, loss = 0.001790
grad_step = 000347, loss = 0.001793
grad_step = 000348, loss = 0.001795
grad_step = 000349, loss = 0.001796
grad_step = 000350, loss = 0.001794
grad_step = 000351, loss = 0.001789
grad_step = 000352, loss = 0.001785
grad_step = 000353, loss = 0.001780
grad_step = 000354, loss = 0.001776
grad_step = 000355, loss = 0.001773
grad_step = 000356, loss = 0.001771
grad_step = 000357, loss = 0.001769
grad_step = 000358, loss = 0.001768
grad_step = 000359, loss = 0.001768
grad_step = 000360, loss = 0.001768
grad_step = 000361, loss = 0.001770
grad_step = 000362, loss = 0.001774
grad_step = 000363, loss = 0.001783
grad_step = 000364, loss = 0.001797
grad_step = 000365, loss = 0.001822
grad_step = 000366, loss = 0.001845
grad_step = 000367, loss = 0.001867
grad_step = 000368, loss = 0.001851
grad_step = 000369, loss = 0.001810
grad_step = 000370, loss = 0.001763
grad_step = 000371, loss = 0.001754
grad_step = 000372, loss = 0.001776
grad_step = 000373, loss = 0.001789
grad_step = 000374, loss = 0.001774
grad_step = 000375, loss = 0.001751
grad_step = 000376, loss = 0.001747
grad_step = 000377, loss = 0.001760
grad_step = 000378, loss = 0.001767
grad_step = 000379, loss = 0.001756
grad_step = 000380, loss = 0.001740
grad_step = 000381, loss = 0.001736
grad_step = 000382, loss = 0.001742
grad_step = 000383, loss = 0.001747
grad_step = 000384, loss = 0.001746
grad_step = 000385, loss = 0.001740
grad_step = 000386, loss = 0.001734
grad_step = 000387, loss = 0.001728
grad_step = 000388, loss = 0.001725
grad_step = 000389, loss = 0.001724
grad_step = 000390, loss = 0.001726
grad_step = 000391, loss = 0.001729
grad_step = 000392, loss = 0.001729
grad_step = 000393, loss = 0.001726
grad_step = 000394, loss = 0.001723
grad_step = 000395, loss = 0.001721
grad_step = 000396, loss = 0.001718
grad_step = 000397, loss = 0.001714
grad_step = 000398, loss = 0.001711
grad_step = 000399, loss = 0.001709
grad_step = 000400, loss = 0.001707
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001705
grad_step = 000402, loss = 0.001703
grad_step = 000403, loss = 0.001701
grad_step = 000404, loss = 0.001700
grad_step = 000405, loss = 0.001700
grad_step = 000406, loss = 0.001700
grad_step = 000407, loss = 0.001702
grad_step = 000408, loss = 0.001707
grad_step = 000409, loss = 0.001721
grad_step = 000410, loss = 0.001746
grad_step = 000411, loss = 0.001798
grad_step = 000412, loss = 0.001861
grad_step = 000413, loss = 0.001951
grad_step = 000414, loss = 0.001938
grad_step = 000415, loss = 0.001867
grad_step = 000416, loss = 0.001738
grad_step = 000417, loss = 0.001687
grad_step = 000418, loss = 0.001740
grad_step = 000419, loss = 0.001791
grad_step = 000420, loss = 0.001760
grad_step = 000421, loss = 0.001691
grad_step = 000422, loss = 0.001686
grad_step = 000423, loss = 0.001732
grad_step = 000424, loss = 0.001734
grad_step = 000425, loss = 0.001686
grad_step = 000426, loss = 0.001670
grad_step = 000427, loss = 0.001698
grad_step = 000428, loss = 0.001707
grad_step = 000429, loss = 0.001684
grad_step = 000430, loss = 0.001670
grad_step = 000431, loss = 0.001669
grad_step = 000432, loss = 0.001673
grad_step = 000433, loss = 0.001678
grad_step = 000434, loss = 0.001673
grad_step = 000435, loss = 0.001659
grad_step = 000436, loss = 0.001649
grad_step = 000437, loss = 0.001655
grad_step = 000438, loss = 0.001663
grad_step = 000439, loss = 0.001657
grad_step = 000440, loss = 0.001648
grad_step = 000441, loss = 0.001644
grad_step = 000442, loss = 0.001642
grad_step = 000443, loss = 0.001639
grad_step = 000444, loss = 0.001639
grad_step = 000445, loss = 0.001641
grad_step = 000446, loss = 0.001639
grad_step = 000447, loss = 0.001633
grad_step = 000448, loss = 0.001627
grad_step = 000449, loss = 0.001625
grad_step = 000450, loss = 0.001623
grad_step = 000451, loss = 0.001620
grad_step = 000452, loss = 0.001619
grad_step = 000453, loss = 0.001620
grad_step = 000454, loss = 0.001619
grad_step = 000455, loss = 0.001618
grad_step = 000456, loss = 0.001618
grad_step = 000457, loss = 0.001619
grad_step = 000458, loss = 0.001621
grad_step = 000459, loss = 0.001624
grad_step = 000460, loss = 0.001632
grad_step = 000461, loss = 0.001643
grad_step = 000462, loss = 0.001666
grad_step = 000463, loss = 0.001686
grad_step = 000464, loss = 0.001721
grad_step = 000465, loss = 0.001732
grad_step = 000466, loss = 0.001736
grad_step = 000467, loss = 0.001691
grad_step = 000468, loss = 0.001635
grad_step = 000469, loss = 0.001590
grad_step = 000470, loss = 0.001585
grad_step = 000471, loss = 0.001611
grad_step = 000472, loss = 0.001632
grad_step = 000473, loss = 0.001629
grad_step = 000474, loss = 0.001601
grad_step = 000475, loss = 0.001575
grad_step = 000476, loss = 0.001568
grad_step = 000477, loss = 0.001577
grad_step = 000478, loss = 0.001592
grad_step = 000479, loss = 0.001597
grad_step = 000480, loss = 0.001591
grad_step = 000481, loss = 0.001575
grad_step = 000482, loss = 0.001558
grad_step = 000483, loss = 0.001549
grad_step = 000484, loss = 0.001548
grad_step = 000485, loss = 0.001552
grad_step = 000486, loss = 0.001556
grad_step = 000487, loss = 0.001558
grad_step = 000488, loss = 0.001555
grad_step = 000489, loss = 0.001549
grad_step = 000490, loss = 0.001542
grad_step = 000491, loss = 0.001535
grad_step = 000492, loss = 0.001528
grad_step = 000493, loss = 0.001523
grad_step = 000494, loss = 0.001518
grad_step = 000495, loss = 0.001514
grad_step = 000496, loss = 0.001511
grad_step = 000497, loss = 0.001509
grad_step = 000498, loss = 0.001507
grad_step = 000499, loss = 0.001508
grad_step = 000500, loss = 0.001512
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001526
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

  date_run                              2020-05-09 14:12:43.759701
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.214263
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 14:12:43.764788
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.098422
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 14:12:43.771760
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.144313
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 14:12:43.776175
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.495558
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
0   2020-05-09 14:12:20.836318  ...    mean_absolute_error
1   2020-05-09 14:12:20.839412  ...     mean_squared_error
2   2020-05-09 14:12:20.842185  ...  median_absolute_error
3   2020-05-09 14:12:20.844940  ...               r2_score
4   2020-05-09 14:12:28.141160  ...    mean_absolute_error
5   2020-05-09 14:12:28.144691  ...     mean_squared_error
6   2020-05-09 14:12:28.147667  ...  median_absolute_error
7   2020-05-09 14:12:28.150265  ...               r2_score
8   2020-05-09 14:12:43.759701  ...    mean_absolute_error
9   2020-05-09 14:12:43.764788  ...     mean_squared_error
10  2020-05-09 14:12:43.771760  ...  median_absolute_error
11  2020-05-09 14:12:43.776175  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:18, 125766.33it/s] 83%|████████▎ | 8224768/9912422 [00:00<00:09, 179548.06it/s]9920512it [00:00, 40634337.79it/s]                           
0it [00:00, ?it/s]32768it [00:00, 615564.59it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457481.24it/s]1654784it [00:00, 11712644.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 187606.41it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0385164780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03228a79b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f038511be48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03228a7d68> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f038511be48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f032bfca438> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0385164f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0337b18cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0385164f98> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0337b18cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f038511be48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f14bda78208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=50df0c9099c9440714b1cb9ba927f263ff0e7d040ea7ef471ab4a419ded21904
  Stored in directory: /tmp/pip-ephem-wheel-cache-z34xkxto/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f14b3be7048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  753664/17464789 [>.............................] - ETA: 1s
 1572864/17464789 [=>............................] - ETA: 1s
 2408448/17464789 [===>..........................] - ETA: 1s
 3260416/17464789 [====>.........................] - ETA: 0s
 4177920/17464789 [======>.......................] - ETA: 0s
 5087232/17464789 [=======>......................] - ETA: 0s
 6127616/17464789 [=========>....................] - ETA: 0s
 7086080/17464789 [===========>..................] - ETA: 0s
 8151040/17464789 [=============>................] - ETA: 0s
 9232384/17464789 [==============>...............] - ETA: 0s
10362880/17464789 [================>.............] - ETA: 0s
11542528/17464789 [==================>...........] - ETA: 0s
12746752/17464789 [====================>.........] - ETA: 0s
14016512/17464789 [=======================>......] - ETA: 0s
15269888/17464789 [=========================>....] - ETA: 0s
16556032/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-09 14:14:07.806010: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 14:14:07.809622: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-09 14:14:07.809748: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560497a50c00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 14:14:07.809760: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 7s - loss: 8.0193 - accuracy: 0.4770 
 3000/25000 [==>...........................] - ETA: 6s - loss: 8.0040 - accuracy: 0.4780
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.9043 - accuracy: 0.4845
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.9150 - accuracy: 0.4838
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8838 - accuracy: 0.4858
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8024 - accuracy: 0.4911
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6496 - accuracy: 0.5011
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
11000/25000 [============>.................] - ETA: 3s - loss: 7.6694 - accuracy: 0.4998
12000/25000 [=============>................] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6478 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6414 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6298 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6417 - accuracy: 0.5016
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6639 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6659 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6542 - accuracy: 0.5008
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6597 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 6s 248us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:14:19.589211
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-09 14:14:19.589211  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-09 14:14:24.542588: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-09 14:14:24.547184: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-09 14:14:24.547299: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5643d5ffce10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-09 14:14:24.547310: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1119b0fd30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 892ms/step - loss: 1.2423 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.2039 - val_crf_viterbi_accuracy: 0.2800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f110eeb7f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 9s - loss: 7.2986 - accuracy: 0.5240
 2000/25000 [=>............................] - ETA: 6s - loss: 7.3753 - accuracy: 0.5190
 3000/25000 [==>...........................] - ETA: 5s - loss: 7.4826 - accuracy: 0.5120
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5670 - accuracy: 0.5065
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6491 - accuracy: 0.5011
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6609 - accuracy: 0.5004
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
11000/25000 [============>.................] - ETA: 2s - loss: 7.6485 - accuracy: 0.5012
12000/25000 [=============>................] - ETA: 2s - loss: 7.6232 - accuracy: 0.5028
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6230 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
15000/25000 [=================>............] - ETA: 2s - loss: 7.6165 - accuracy: 0.5033
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6331 - accuracy: 0.5022
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6477 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6303 - accuracy: 0.5024
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6275 - accuracy: 0.5026
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6308 - accuracy: 0.5023
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6387 - accuracy: 0.5018
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6480 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
25000/25000 [==============================] - 6s 240us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f10fc384278> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:43:30, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:44:25, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:22:22, 23.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:16:09, 32.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:04:33, 47.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.51M/862M [00:01<3:31:49, 67.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.4M/862M [00:01<2:27:42, 95.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:01<1:42:55, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.4M/862M [00:01<1:11:46, 195kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<50:06, 278kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.9M/862M [00:01<34:59, 396kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:02<24:27, 563kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<17:08, 800kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:02<12:01, 1.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<08:28, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<06:27, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:25, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<08:23, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:52, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<05:02, 2.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<08:27, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<07:38, 1.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<05:43, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<06:37, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:44, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<06:06, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<04:26, 2.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<08:43, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<07:35, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<05:37, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:49, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<07:27, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<05:50, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:13<04:14, 3.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<2:33:30, 85.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<1:48:47, 120kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<1:16:17, 171kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<56:16, 231kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<40:42, 319kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<28:46, 451kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<23:08, 559kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<17:31, 738kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<12:34, 1.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<11:49, 1.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<09:35, 1.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:20<06:58, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<07:56, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<06:50, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<05:06, 2.50MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<06:34, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<04:24, 2.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:06, 2.08MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:34, 2.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:10, 3.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:54, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:26, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:07, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:50, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:23, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:04, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:48, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:35, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:14, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:40, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:16, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:00, 3.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:40, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:30, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:10, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:36, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:12, 2.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:57, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:36, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:11, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:56, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:10, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<03:52, 3.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:34, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:08, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:53, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:07, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:53, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:31, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:18, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:55, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<03:35, 3.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:31, 1.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:54, 1.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:47, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:57, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:25, 2.66MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:53, 1.99MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:17, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<03:57, 2.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:32, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:15, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:58, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:20, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:56, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:42, 3.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:17, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:04, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:49, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:13, 2.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:49, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:40, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:12, 2.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:58, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:46, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:27, 3.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<43:11, 262kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<31:24, 359kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<22:14, 507kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<18:08, 619kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<13:50, 811kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:57, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<09:35, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:00, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:51, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:34, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:44, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<04:17, 2.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:35, 1.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:02, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<03:48, 2.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:47, 2.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:37, 3.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:06, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:41, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:33, 3.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:38, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:30, 3.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:59, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:36, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:29, 3.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:33, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:27, 3.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:55, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:31, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:25, 3.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:53, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:24, 3.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:51, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:17, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:16, 3.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<02:24, 4.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<41:00, 254kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<30:49, 337kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<22:05, 470kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<17:03, 606kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<12:59, 795kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<09:20, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:55, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:17, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<05:21, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:28, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:58, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:37, 2.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:46, 1.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:48, 1.75MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:18, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:22, 1.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<05:54, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:34, 2.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:20, 3.00MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:38, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:41, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:14, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:17, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:42, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:31, 2.81MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:46, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:14, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:33, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:12, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:09, 3.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:27, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:02, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:56, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<02:52, 3.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:35, 1.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:19, 1.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:39, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:49, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:37, 2.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:46, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:18, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:12, 2.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:29, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:04, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<03:57, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<02:53, 3.25MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:38, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:37, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:10, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:05, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:30, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:19, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:05, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:03, 3.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:18, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:47, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:53, 3.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:09, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:46, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:43, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:46, 3.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:44, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:14, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:11, 2.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:19, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:57, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:57, 3.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:09, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:39, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:38, 2.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:42, 3.27MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:47, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:16, 2.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:10, 2.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:16, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:53, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:56, 2.97MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:06, 2.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:45, 2.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:50, 3.04MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:01, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:34, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:38, 2.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:55, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:38, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:44, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:29, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:32, 3.31MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:48, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:37, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:08, 2.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:50, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:13, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:09, 2.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:09, 1.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:36, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:35, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:37, 3.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:16, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:32, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:20, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:15, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:39, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:36, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:37, 3.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:28, 1.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:39, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:25, 2.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:16, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:38, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:34, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:37, 3.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:43, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:08, 1.92MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:03, 2.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:58, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:23, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:28, 2.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:40, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:23, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:34, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:35, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:12, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:19, 3.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<09:25, 814kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<07:23, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:21, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:30, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:38, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:26, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:26, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:26, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:29, 3.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:47, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:48, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:32, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:12, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:27, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:30, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:28, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:54, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:54, 2.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:43, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:05, 1.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:10, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:18, 3.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:18, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:26, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:17, 2.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:57, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:29, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:35, 2.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:28, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:52, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:03, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:12, 3.17MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<6:44:28, 17.3kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<4:43:36, 24.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<3:17:57, 35.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<2:19:28, 49.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<1:38:14, 70.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<1:08:41, 100kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<49:26, 138kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<35:16, 194kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<24:46, 275kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<18:51, 359kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<14:34, 464kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<10:28, 645kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<07:23, 910kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<08:43, 768kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:48, 984kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:54, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:58, 1.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:09, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:04, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:40, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:56, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:05, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:11, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:54, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:11, 2.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:01, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:25, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:41, 2.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<01:56, 3.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<07:03, 902kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<05:34, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:01, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:17, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:38, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:41, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:21, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:58, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:13, 2.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:01, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:22, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:40, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<01:55, 3.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<5:53:05, 17.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<4:07:32, 24.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<2:52:41, 35.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<2:01:30, 49.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<1:26:11, 69.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:00:30, 99.2kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<42:57, 138kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<30:39, 194kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<21:31, 275kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<16:20, 360kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<12:41, 463kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:09, 640kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<06:26, 903kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<45:53, 127kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<32:40, 178kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<22:55, 252kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<17:15, 333kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<13:15, 433kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<09:32, 599kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<07:32, 752kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<05:51, 966kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:14, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:14, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:06, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:09, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:05, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:44, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:01, 2.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:41, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:26, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:50, 2.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:32, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:50, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:14, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:16, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:43, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:24, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:45, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:08, 2.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<01:33, 3.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:09, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:27, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:31, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:57, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:06, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:23, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:45, 2.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:57, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:35, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:56, 2.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:30, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:46, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:12, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:34, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<20:03, 245kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<14:32, 338kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<10:15, 476kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<08:15, 587kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<06:46, 716kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<04:56, 977kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<03:29, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:59, 958kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:59, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:53, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:06, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:39, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:58, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:27, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:11, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:37, 2.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:13, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:29, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:56, 2.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:24, 3.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:27, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:31, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:58, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:06, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:57, 2.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:29, 2.96MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:59, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:19, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:48, 2.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:19, 3.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:35, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:15, 1.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:39, 2.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:09, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:56, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:27, 2.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:59, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:14, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:46, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:53, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:44, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:18, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:51, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:43, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:17, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:48, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:05, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:37, 2.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:10, 3.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<05:40, 683kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:21, 887kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:08, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<03:04, 1.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:31, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:50, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:09, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:53, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:23, 2.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:02, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:35, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:08, 3.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:07, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:33, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:52, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:07, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:12, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:41, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:13, 2.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:48, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:19, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:41, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:09, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:55, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:21, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:43, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:47, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:25, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:01, 3.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<06:01, 541kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<04:51, 671kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<03:30, 922kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<02:29, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:06, 1.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:30, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:48, 1.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:57, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:00, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:33, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:34, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:25, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:03, 2.84MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:25, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:36, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:16, 2.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:54, 3.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<21:27, 136kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<15:17, 190kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<10:40, 270kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<08:02, 354kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<06:13, 457kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:28, 633kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:31, 788kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<03:01, 915kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:15, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:59, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:40, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:13, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:27, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:17, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<00:57, 2.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:13, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:21, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:04, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:08, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:04, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:48, 3.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:05, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:15, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<00:59, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:42, 3.36MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<03:43, 633kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:47, 839kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:04, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:27, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:35, 882kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:16, 1.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:42, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:11, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<04:52, 454kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<03:38, 607kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:33, 851kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<02:16, 946kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<02:00, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:29, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:03, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:50, 1.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:30, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:05, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:13, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:15, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:58, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<00:59, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:39, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:53, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:00, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:46, 2.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:33, 3.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:06, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:57, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:42, 2.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:47, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:35, 2.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:47, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:53, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:42, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:29, 3.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<06:31, 245kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<04:42, 338kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<03:16, 478kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:36, 589kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:07, 717kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:32, 982kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<01:04, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:22, 1.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:06, 1.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:47, 1.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:52, 1.60MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:44, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:32, 2.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:41, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:45, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:35, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:51<00:24, 3.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<02:39, 473kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<02:06, 593kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:30, 818kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<01:02, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:15, 937kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:00, 1.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:42, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:44, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:38, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<01:42, 611kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:13, 799kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:54, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:36, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:32, 594kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:15, 718kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:54, 981kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:37, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:45, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:36, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:26, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:28, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:28, 1.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:21, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:14, 2.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:44, 959kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:34, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:24, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:24, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:25, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:12, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<04:11, 135kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<02:57, 189kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:59, 269kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<01:24, 353kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:04, 459kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:45, 638kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:29, 899kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:32, 791kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:24, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:16, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:15, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:08, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:05, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:03, 3.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 3.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 3.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.85MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1046/400000 [00:00<00:38, 10449.97it/s]  1%|          | 2056/400000 [00:00<00:38, 10340.20it/s]  1%|          | 3203/400000 [00:00<00:37, 10653.91it/s]  1%|          | 4263/400000 [00:00<00:37, 10635.10it/s]  1%|▏         | 5393/400000 [00:00<00:36, 10822.71it/s]  2%|▏         | 6357/400000 [00:00<00:37, 10436.95it/s]  2%|▏         | 7452/400000 [00:00<00:37, 10584.12it/s]  2%|▏         | 8575/400000 [00:00<00:36, 10767.63it/s]  2%|▏         | 9680/400000 [00:00<00:35, 10849.73it/s]  3%|▎         | 10809/400000 [00:01<00:35, 10974.57it/s]  3%|▎         | 11879/400000 [00:01<00:35, 10872.90it/s]  3%|▎         | 12948/400000 [00:01<00:37, 10448.86it/s]  4%|▎         | 14056/400000 [00:01<00:36, 10629.66it/s]  4%|▍         | 15198/400000 [00:01<00:35, 10854.32it/s]  4%|▍         | 16360/400000 [00:01<00:34, 11071.79it/s]  4%|▍         | 17467/400000 [00:01<00:34, 10982.78it/s]  5%|▍         | 18643/400000 [00:01<00:34, 11202.54it/s]  5%|▍         | 19764/400000 [00:01<00:35, 10731.07it/s]  5%|▌         | 20899/400000 [00:01<00:34, 10906.93it/s]  6%|▌         | 22058/400000 [00:02<00:34, 11103.17it/s]  6%|▌         | 23172/400000 [00:02<00:34, 10962.25it/s]  6%|▌         | 24272/400000 [00:02<00:34, 10928.19it/s]  6%|▋         | 25367/400000 [00:02<00:36, 10376.95it/s]  7%|▋         | 26412/400000 [00:02<00:36, 10254.36it/s]  7%|▋         | 27443/400000 [00:02<00:36, 10143.78it/s]  7%|▋         | 28462/400000 [00:02<00:36, 10104.05it/s]  7%|▋         | 29490/400000 [00:02<00:36, 10154.68it/s]  8%|▊         | 30508/400000 [00:02<00:37, 9730.57it/s]   8%|▊         | 31487/400000 [00:02<00:39, 9277.60it/s]  8%|▊         | 32432/400000 [00:03<00:39, 9326.82it/s]  8%|▊         | 33371/400000 [00:03<00:39, 9311.82it/s]  9%|▊         | 34403/400000 [00:03<00:38, 9591.87it/s]  9%|▉         | 35558/400000 [00:03<00:36, 10105.45it/s]  9%|▉         | 36742/400000 [00:03<00:34, 10569.82it/s]  9%|▉         | 37813/400000 [00:03<00:35, 10347.18it/s] 10%|▉         | 38875/400000 [00:03<00:34, 10426.32it/s] 10%|▉         | 39925/400000 [00:03<00:36, 9986.16it/s]  10%|█         | 40933/400000 [00:03<00:36, 9957.43it/s] 10%|█         | 41964/400000 [00:04<00:35, 10059.55it/s] 11%|█         | 42984/400000 [00:04<00:35, 10099.45it/s] 11%|█         | 44064/400000 [00:04<00:34, 10298.86it/s] 11%|█▏        | 45202/400000 [00:04<00:33, 10600.15it/s] 12%|█▏        | 46267/400000 [00:04<00:34, 10353.25it/s] 12%|█▏        | 47330/400000 [00:04<00:33, 10432.51it/s] 12%|█▏        | 48377/400000 [00:04<00:33, 10419.60it/s] 12%|█▏        | 49422/400000 [00:04<00:33, 10333.54it/s] 13%|█▎        | 50559/400000 [00:04<00:32, 10621.56it/s] 13%|█▎        | 51642/400000 [00:04<00:32, 10680.67it/s] 13%|█▎        | 52772/400000 [00:05<00:31, 10858.49it/s] 13%|█▎        | 53891/400000 [00:05<00:31, 10953.67it/s] 14%|█▍        | 55014/400000 [00:05<00:31, 11033.10it/s] 14%|█▍        | 56174/400000 [00:05<00:30, 11195.17it/s] 14%|█▍        | 57296/400000 [00:05<00:31, 10912.94it/s] 15%|█▍        | 58390/400000 [00:05<00:31, 10677.29it/s] 15%|█▍        | 59461/400000 [00:05<00:33, 10149.83it/s] 15%|█▌        | 60484/400000 [00:05<00:34, 9944.87it/s]  15%|█▌        | 61485/400000 [00:05<00:34, 9778.05it/s] 16%|█▌        | 62468/400000 [00:05<00:35, 9500.92it/s] 16%|█▌        | 63433/400000 [00:06<00:35, 9544.43it/s] 16%|█▌        | 64392/400000 [00:06<00:35, 9462.72it/s] 16%|█▋        | 65342/400000 [00:06<00:36, 9281.83it/s] 17%|█▋        | 66273/400000 [00:06<00:36, 9246.15it/s] 17%|█▋        | 67243/400000 [00:06<00:35, 9376.38it/s] 17%|█▋        | 68183/400000 [00:06<00:36, 9215.11it/s] 17%|█▋        | 69107/400000 [00:06<00:36, 9007.35it/s] 18%|█▊        | 70100/400000 [00:06<00:35, 9263.89it/s] 18%|█▊        | 71078/400000 [00:06<00:34, 9411.85it/s] 18%|█▊        | 72077/400000 [00:07<00:34, 9576.63it/s] 18%|█▊        | 73139/400000 [00:07<00:33, 9867.43it/s] 19%|█▊        | 74133/400000 [00:07<00:32, 9888.21it/s] 19%|█▉        | 75166/400000 [00:07<00:32, 10016.50it/s] 19%|█▉        | 76192/400000 [00:07<00:32, 10086.81it/s] 19%|█▉        | 77244/400000 [00:07<00:31, 10211.43it/s] 20%|█▉        | 78277/400000 [00:07<00:31, 10244.13it/s] 20%|█▉        | 79303/400000 [00:07<00:31, 10134.00it/s] 20%|██        | 80318/400000 [00:07<00:31, 10088.58it/s] 20%|██        | 81328/400000 [00:07<00:31, 10009.14it/s] 21%|██        | 82347/400000 [00:08<00:31, 10061.01it/s] 21%|██        | 83373/400000 [00:08<00:31, 10118.65it/s] 21%|██        | 84386/400000 [00:08<00:31, 10026.31it/s] 21%|██▏       | 85432/400000 [00:08<00:30, 10151.13it/s] 22%|██▏       | 86561/400000 [00:08<00:29, 10465.70it/s] 22%|██▏       | 87671/400000 [00:08<00:29, 10647.49it/s] 22%|██▏       | 88759/400000 [00:08<00:29, 10715.24it/s] 22%|██▏       | 89833/400000 [00:08<00:29, 10464.13it/s] 23%|██▎       | 90984/400000 [00:08<00:28, 10755.86it/s] 23%|██▎       | 92106/400000 [00:08<00:28, 10889.81it/s] 23%|██▎       | 93246/400000 [00:09<00:27, 11037.92it/s] 24%|██▎       | 94353/400000 [00:09<00:28, 10627.94it/s] 24%|██▍       | 95421/400000 [00:09<00:29, 10339.35it/s] 24%|██▍       | 96533/400000 [00:09<00:28, 10560.50it/s] 24%|██▍       | 97625/400000 [00:09<00:28, 10664.63it/s] 25%|██▍       | 98720/400000 [00:09<00:28, 10746.54it/s] 25%|██▍       | 99798/400000 [00:09<00:28, 10695.29it/s] 25%|██▌       | 100870/400000 [00:09<00:28, 10427.64it/s] 25%|██▌       | 101986/400000 [00:09<00:28, 10636.24it/s] 26%|██▌       | 103126/400000 [00:09<00:27, 10853.23it/s] 26%|██▌       | 104252/400000 [00:10<00:26, 10969.62it/s] 26%|██▋       | 105395/400000 [00:10<00:26, 11102.62it/s] 27%|██▋       | 106508/400000 [00:10<00:26, 11074.81it/s] 27%|██▋       | 107618/400000 [00:10<00:26, 10903.92it/s] 27%|██▋       | 108799/400000 [00:10<00:26, 11159.74it/s] 27%|██▋       | 109918/400000 [00:10<00:26, 11017.40it/s] 28%|██▊       | 111100/400000 [00:10<00:25, 11244.91it/s] 28%|██▊       | 112228/400000 [00:10<00:25, 11117.17it/s] 28%|██▊       | 113342/400000 [00:10<00:26, 10883.13it/s] 29%|██▊       | 114433/400000 [00:11<00:27, 10329.52it/s] 29%|██▉       | 115481/400000 [00:11<00:27, 10373.41it/s] 29%|██▉       | 116524/400000 [00:11<00:27, 10318.91it/s] 29%|██▉       | 117560/400000 [00:11<00:27, 10185.16it/s] 30%|██▉       | 118662/400000 [00:11<00:26, 10421.79it/s] 30%|██▉       | 119708/400000 [00:11<00:27, 10336.55it/s] 30%|███       | 120745/400000 [00:11<00:27, 10090.18it/s] 30%|███       | 121758/400000 [00:11<00:27, 10038.07it/s] 31%|███       | 122765/400000 [00:11<00:28, 9894.27it/s]  31%|███       | 123919/400000 [00:11<00:26, 10334.31it/s] 31%|███▏      | 125006/400000 [00:12<00:26, 10486.95it/s] 32%|███▏      | 126060/400000 [00:12<00:26, 10332.63it/s] 32%|███▏      | 127098/400000 [00:12<00:27, 9966.48it/s]  32%|███▏      | 128101/400000 [00:12<00:27, 9865.72it/s] 32%|███▏      | 129225/400000 [00:12<00:26, 10240.46it/s] 33%|███▎      | 130282/400000 [00:12<00:26, 10334.38it/s] 33%|███▎      | 131368/400000 [00:12<00:25, 10486.60it/s] 33%|███▎      | 132421/400000 [00:12<00:26, 10259.57it/s] 33%|███▎      | 133451/400000 [00:12<00:26, 10050.37it/s] 34%|███▎      | 134489/400000 [00:12<00:26, 10145.45it/s] 34%|███▍      | 135507/400000 [00:13<00:26, 10057.97it/s] 34%|███▍      | 136596/400000 [00:13<00:25, 10293.09it/s] 34%|███▍      | 137688/400000 [00:13<00:25, 10468.52it/s] 35%|███▍      | 138738/400000 [00:13<00:25, 10141.28it/s] 35%|███▍      | 139856/400000 [00:13<00:24, 10429.95it/s] 35%|███▌      | 140955/400000 [00:13<00:24, 10589.62it/s] 36%|███▌      | 142111/400000 [00:13<00:23, 10862.40it/s] 36%|███▌      | 143250/400000 [00:13<00:23, 11014.68it/s] 36%|███▌      | 144356/400000 [00:13<00:23, 10961.93it/s] 36%|███▋      | 145467/400000 [00:13<00:23, 11005.03it/s] 37%|███▋      | 146570/400000 [00:14<00:23, 10902.79it/s] 37%|███▋      | 147662/400000 [00:14<00:23, 10727.26it/s] 37%|███▋      | 148737/400000 [00:14<00:24, 10100.72it/s] 37%|███▋      | 149756/400000 [00:14<00:25, 9722.26it/s]  38%|███▊      | 150783/400000 [00:14<00:25, 9878.60it/s] 38%|███▊      | 151868/400000 [00:14<00:24, 10149.58it/s] 38%|███▊      | 152950/400000 [00:14<00:23, 10341.72it/s] 38%|███▊      | 153990/400000 [00:14<00:23, 10349.81it/s] 39%|███▉      | 155029/400000 [00:14<00:24, 9957.64it/s]  39%|███▉      | 156107/400000 [00:15<00:23, 10190.83it/s] 39%|███▉      | 157248/400000 [00:15<00:23, 10527.14it/s] 40%|███▉      | 158361/400000 [00:15<00:22, 10701.01it/s] 40%|███▉      | 159437/400000 [00:15<00:22, 10654.48it/s] 40%|████      | 160557/400000 [00:15<00:22, 10810.11it/s] 40%|████      | 161742/400000 [00:15<00:21, 11099.67it/s] 41%|████      | 162857/400000 [00:15<00:22, 10753.18it/s] 41%|████      | 163938/400000 [00:15<00:22, 10646.35it/s] 41%|████▏     | 165007/400000 [00:15<00:22, 10557.00it/s] 42%|████▏     | 166115/400000 [00:15<00:21, 10707.55it/s] 42%|████▏     | 167189/400000 [00:16<00:21, 10673.71it/s] 42%|████▏     | 168282/400000 [00:16<00:21, 10747.13it/s] 42%|████▏     | 169359/400000 [00:16<00:21, 10681.87it/s] 43%|████▎     | 170429/400000 [00:16<00:21, 10478.88it/s] 43%|████▎     | 171479/400000 [00:16<00:21, 10466.88it/s] 43%|████▎     | 172558/400000 [00:16<00:21, 10560.98it/s] 43%|████▎     | 173728/400000 [00:16<00:20, 10876.43it/s] 44%|████▎     | 174830/400000 [00:16<00:20, 10918.63it/s] 44%|████▍     | 175925/400000 [00:16<00:20, 10799.36it/s] 44%|████▍     | 177059/400000 [00:16<00:20, 10954.15it/s] 45%|████▍     | 178157/400000 [00:17<00:20, 10912.83it/s] 45%|████▍     | 179293/400000 [00:17<00:19, 11042.26it/s] 45%|████▌     | 180481/400000 [00:17<00:19, 11280.00it/s] 45%|████▌     | 181612/400000 [00:17<00:19, 11236.74it/s] 46%|████▌     | 182776/400000 [00:17<00:19, 11352.59it/s] 46%|████▌     | 183961/400000 [00:17<00:18, 11495.96it/s] 46%|████▋     | 185123/400000 [00:17<00:18, 11532.17it/s] 47%|████▋     | 186305/400000 [00:17<00:18, 11615.36it/s] 47%|████▋     | 187468/400000 [00:17<00:18, 11418.86it/s] 47%|████▋     | 188612/400000 [00:18<00:19, 11026.59it/s] 47%|████▋     | 189719/400000 [00:18<00:19, 10753.59it/s] 48%|████▊     | 190825/400000 [00:18<00:19, 10841.41it/s] 48%|████▊     | 191981/400000 [00:18<00:18, 11045.11it/s] 48%|████▊     | 193089/400000 [00:18<00:19, 10811.41it/s] 49%|████▊     | 194235/400000 [00:18<00:18, 10996.57it/s] 49%|████▉     | 195338/400000 [00:18<00:18, 10962.27it/s] 49%|████▉     | 196466/400000 [00:18<00:18, 11053.72it/s] 49%|████▉     | 197574/400000 [00:18<00:18, 10949.61it/s] 50%|████▉     | 198671/400000 [00:18<00:18, 10928.77it/s] 50%|████▉     | 199797/400000 [00:19<00:18, 11025.78it/s] 50%|█████     | 200901/400000 [00:19<00:18, 10748.70it/s] 51%|█████     | 202007/400000 [00:19<00:18, 10838.81it/s] 51%|█████     | 203162/400000 [00:19<00:17, 11042.51it/s] 51%|█████     | 204269/400000 [00:19<00:17, 10982.04it/s] 51%|█████▏    | 205369/400000 [00:19<00:17, 10904.60it/s] 52%|█████▏    | 206484/400000 [00:19<00:17, 10974.84it/s] 52%|█████▏    | 207583/400000 [00:19<00:17, 10940.18it/s] 52%|█████▏    | 208688/400000 [00:19<00:17, 10970.79it/s] 52%|█████▏    | 209786/400000 [00:19<00:18, 10534.78it/s] 53%|█████▎    | 210844/400000 [00:20<00:18, 10376.52it/s] 53%|█████▎    | 211887/400000 [00:20<00:18, 10391.93it/s] 53%|█████▎    | 212965/400000 [00:20<00:17, 10505.06it/s] 54%|█████▎    | 214029/400000 [00:20<00:17, 10544.51it/s] 54%|█████▍    | 215108/400000 [00:20<00:17, 10615.63it/s] 54%|█████▍    | 216305/400000 [00:20<00:16, 10986.80it/s] 54%|█████▍    | 217408/400000 [00:20<00:16, 10913.17it/s] 55%|█████▍    | 218531/400000 [00:20<00:16, 11003.80it/s] 55%|█████▍    | 219644/400000 [00:20<00:16, 11041.31it/s] 55%|█████▌    | 220750/400000 [00:20<00:17, 10449.77it/s] 55%|█████▌    | 221882/400000 [00:21<00:16, 10695.20it/s] 56%|█████▌    | 222959/400000 [00:21<00:16, 10492.87it/s] 56%|█████▌    | 224056/400000 [00:21<00:16, 10629.31it/s] 56%|█████▋    | 225149/400000 [00:21<00:16, 10714.97it/s] 57%|█████▋    | 226224/400000 [00:21<00:16, 10618.46it/s] 57%|█████▋    | 227398/400000 [00:21<00:15, 10930.27it/s] 57%|█████▋    | 228573/400000 [00:21<00:15, 11163.22it/s] 57%|█████▋    | 229718/400000 [00:21<00:15, 11247.23it/s] 58%|█████▊    | 230846/400000 [00:21<00:15, 11013.03it/s] 58%|█████▊    | 231951/400000 [00:22<00:15, 10762.87it/s] 58%|█████▊    | 233049/400000 [00:22<00:15, 10826.73it/s] 59%|█████▊    | 234135/400000 [00:22<00:15, 10823.46it/s] 59%|█████▉    | 235248/400000 [00:22<00:15, 10912.85it/s] 59%|█████▉    | 236341/400000 [00:22<00:15, 10859.63it/s] 59%|█████▉    | 237429/400000 [00:22<00:14, 10849.58it/s] 60%|█████▉    | 238539/400000 [00:22<00:14, 10920.13it/s] 60%|█████▉    | 239676/400000 [00:22<00:14, 11047.75it/s] 60%|██████    | 240855/400000 [00:22<00:14, 11259.73it/s] 60%|██████    | 241983/400000 [00:22<00:14, 11219.00it/s] 61%|██████    | 243107/400000 [00:23<00:13, 11211.93it/s] 61%|██████    | 244266/400000 [00:23<00:13, 11322.08it/s] 61%|██████▏   | 245400/400000 [00:23<00:13, 11185.86it/s] 62%|██████▏   | 246576/400000 [00:23<00:13, 11350.75it/s] 62%|██████▏   | 247769/400000 [00:23<00:13, 11517.06it/s] 62%|██████▏   | 248923/400000 [00:23<00:13, 11358.04it/s] 63%|██████▎   | 250061/400000 [00:23<00:13, 11079.25it/s] 63%|██████▎   | 251172/400000 [00:23<00:13, 11087.89it/s] 63%|██████▎   | 252344/400000 [00:23<00:13, 11268.52it/s] 63%|██████▎   | 253489/400000 [00:23<00:12, 11322.15it/s] 64%|██████▎   | 254623/400000 [00:24<00:12, 11216.05it/s] 64%|██████▍   | 255746/400000 [00:24<00:12, 11139.31it/s] 64%|██████▍   | 256861/400000 [00:24<00:13, 10691.03it/s] 64%|██████▍   | 257985/400000 [00:24<00:13, 10847.94it/s] 65%|██████▍   | 259074/400000 [00:24<00:13, 10808.22it/s] 65%|██████▌   | 260158/400000 [00:24<00:13, 10647.47it/s] 65%|██████▌   | 261250/400000 [00:24<00:12, 10726.70it/s] 66%|██████▌   | 262386/400000 [00:24<00:12, 10905.55it/s] 66%|██████▌   | 263479/400000 [00:24<00:12, 10749.74it/s] 66%|██████▌   | 264617/400000 [00:24<00:12, 10929.88it/s] 66%|██████▋   | 265730/400000 [00:25<00:12, 10986.61it/s] 67%|██████▋   | 266852/400000 [00:25<00:12, 11053.68it/s] 67%|██████▋   | 267959/400000 [00:25<00:12, 10858.80it/s] 67%|██████▋   | 269047/400000 [00:25<00:12, 10327.31it/s] 68%|██████▊   | 270137/400000 [00:25<00:12, 10489.16it/s] 68%|██████▊   | 271234/400000 [00:25<00:12, 10626.01it/s] 68%|██████▊   | 272335/400000 [00:25<00:11, 10737.91it/s] 68%|██████▊   | 273441/400000 [00:25<00:11, 10830.98it/s] 69%|██████▊   | 274595/400000 [00:25<00:11, 11033.20it/s] 69%|██████▉   | 275701/400000 [00:26<00:11, 10627.57it/s] 69%|██████▉   | 276814/400000 [00:26<00:11, 10772.87it/s] 69%|██████▉   | 277932/400000 [00:26<00:11, 10889.78it/s] 70%|██████▉   | 279025/400000 [00:26<00:11, 10763.72it/s] 70%|███████   | 280115/400000 [00:26<00:11, 10802.72it/s] 70%|███████   | 281198/400000 [00:26<00:11, 10792.31it/s] 71%|███████   | 282313/400000 [00:26<00:10, 10894.73it/s] 71%|███████   | 283448/400000 [00:26<00:10, 11025.64it/s] 71%|███████   | 284602/400000 [00:26<00:10, 11172.39it/s] 71%|███████▏  | 285751/400000 [00:26<00:10, 11264.13it/s] 72%|███████▏  | 286879/400000 [00:27<00:10, 10877.18it/s] 72%|███████▏  | 287971/400000 [00:27<00:10, 10766.87it/s] 72%|███████▏  | 289051/400000 [00:27<00:10, 10641.10it/s] 73%|███████▎  | 290118/400000 [00:27<00:10, 10387.69it/s] 73%|███████▎  | 291277/400000 [00:27<00:10, 10720.23it/s] 73%|███████▎  | 292399/400000 [00:27<00:09, 10861.72it/s] 73%|███████▎  | 293515/400000 [00:27<00:09, 10947.47it/s] 74%|███████▎  | 294621/400000 [00:27<00:09, 10980.52it/s] 74%|███████▍  | 295722/400000 [00:27<00:09, 10946.51it/s] 74%|███████▍  | 296878/400000 [00:27<00:09, 11120.71it/s] 74%|███████▍  | 297992/400000 [00:28<00:09, 10642.82it/s] 75%|███████▍  | 299062/400000 [00:28<00:09, 10610.76it/s] 75%|███████▌  | 300127/400000 [00:28<00:09, 10540.94it/s] 75%|███████▌  | 301184/400000 [00:28<00:09, 10196.42it/s] 76%|███████▌  | 302209/400000 [00:28<00:09, 10201.74it/s] 76%|███████▌  | 303267/400000 [00:28<00:09, 10311.27it/s] 76%|███████▌  | 304301/400000 [00:28<00:09, 10067.76it/s] 76%|███████▋  | 305321/400000 [00:28<00:09, 10105.25it/s] 77%|███████▋  | 306346/400000 [00:28<00:09, 10147.49it/s] 77%|███████▋  | 307363/400000 [00:28<00:09, 9999.72it/s]  77%|███████▋  | 308377/400000 [00:29<00:09, 10038.41it/s] 77%|███████▋  | 309382/400000 [00:29<00:09, 9988.84it/s]  78%|███████▊  | 310382/400000 [00:29<00:08, 9964.67it/s] 78%|███████▊  | 311409/400000 [00:29<00:08, 10053.31it/s] 78%|███████▊  | 312415/400000 [00:29<00:08, 10027.12it/s] 78%|███████▊  | 313460/400000 [00:29<00:08, 10148.04it/s] 79%|███████▊  | 314527/400000 [00:29<00:08, 10298.91it/s] 79%|███████▉  | 315604/400000 [00:29<00:08, 10434.73it/s] 79%|███████▉  | 316649/400000 [00:29<00:08, 10062.68it/s] 79%|███████▉  | 317659/400000 [00:30<00:08, 10015.97it/s] 80%|███████▉  | 318664/400000 [00:30<00:08, 9906.12it/s]  80%|███████▉  | 319731/400000 [00:30<00:07, 10123.12it/s] 80%|████████  | 320778/400000 [00:30<00:07, 10221.93it/s] 80%|████████  | 321803/400000 [00:30<00:07, 10031.26it/s] 81%|████████  | 322809/400000 [00:30<00:07, 9802.92it/s]  81%|████████  | 323793/400000 [00:30<00:07, 9769.89it/s] 81%|████████  | 324777/400000 [00:30<00:07, 9788.39it/s] 81%|████████▏ | 325758/400000 [00:30<00:07, 9627.46it/s] 82%|████████▏ | 326723/400000 [00:30<00:07, 9624.16it/s] 82%|████████▏ | 327714/400000 [00:31<00:07, 9705.64it/s] 82%|████████▏ | 328686/400000 [00:31<00:07, 9654.27it/s] 82%|████████▏ | 329656/400000 [00:31<00:07, 9667.78it/s] 83%|████████▎ | 330624/400000 [00:31<00:07, 9487.74it/s] 83%|████████▎ | 331627/400000 [00:31<00:07, 9642.95it/s] 83%|████████▎ | 332678/400000 [00:31<00:06, 9886.45it/s] 83%|████████▎ | 333701/400000 [00:31<00:06, 9985.54it/s] 84%|████████▎ | 334711/400000 [00:31<00:06, 10018.75it/s] 84%|████████▍ | 335715/400000 [00:31<00:06, 9975.98it/s]  84%|████████▍ | 336714/400000 [00:31<00:06, 9954.18it/s] 84%|████████▍ | 337742/400000 [00:32<00:06, 10048.78it/s] 85%|████████▍ | 338748/400000 [00:32<00:06, 10006.97it/s] 85%|████████▍ | 339801/400000 [00:32<00:05, 10157.88it/s] 85%|████████▌ | 340818/400000 [00:32<00:05, 9976.04it/s]  85%|████████▌ | 341817/400000 [00:32<00:05, 9900.35it/s] 86%|████████▌ | 342809/400000 [00:32<00:05, 9906.10it/s] 86%|████████▌ | 343801/400000 [00:32<00:05, 9739.48it/s] 86%|████████▌ | 344783/400000 [00:32<00:05, 9763.42it/s] 86%|████████▋ | 345852/400000 [00:32<00:05, 10022.56it/s] 87%|████████▋ | 346892/400000 [00:32<00:05, 10132.14it/s] 87%|████████▋ | 347976/400000 [00:33<00:05, 10332.90it/s] 87%|████████▋ | 349079/400000 [00:33<00:04, 10529.83it/s] 88%|████████▊ | 350138/400000 [00:33<00:04, 10547.25it/s] 88%|████████▊ | 351260/400000 [00:33<00:04, 10738.13it/s] 88%|████████▊ | 352401/400000 [00:33<00:04, 10930.26it/s] 88%|████████▊ | 353497/400000 [00:33<00:04, 10719.05it/s] 89%|████████▊ | 354572/400000 [00:33<00:04, 10552.04it/s] 89%|████████▉ | 355630/400000 [00:33<00:04, 10350.89it/s] 89%|████████▉ | 356668/400000 [00:33<00:04, 9992.56it/s]  89%|████████▉ | 357807/400000 [00:33<00:04, 10374.29it/s] 90%|████████▉ | 358888/400000 [00:34<00:03, 10498.76it/s] 90%|████████▉ | 359944/400000 [00:34<00:03, 10349.73it/s] 90%|█████████ | 361064/400000 [00:34<00:03, 10588.06it/s] 91%|█████████ | 362128/400000 [00:34<00:03, 10495.16it/s] 91%|█████████ | 363181/400000 [00:34<00:03, 10388.39it/s] 91%|█████████ | 364273/400000 [00:34<00:03, 10541.45it/s] 91%|█████████▏| 365364/400000 [00:34<00:03, 10648.65it/s] 92%|█████████▏| 366500/400000 [00:34<00:03, 10850.95it/s] 92%|█████████▏| 367635/400000 [00:34<00:02, 10995.69it/s] 92%|█████████▏| 368737/400000 [00:35<00:03, 10405.22it/s] 92%|█████████▏| 369803/400000 [00:35<00:02, 10477.54it/s] 93%|█████████▎| 370869/400000 [00:35<00:02, 10528.88it/s] 93%|█████████▎| 372056/400000 [00:35<00:02, 10897.74it/s] 93%|█████████▎| 373152/400000 [00:35<00:02, 10790.51it/s] 94%|█████████▎| 374236/400000 [00:35<00:02, 10739.61it/s] 94%|█████████▍| 375314/400000 [00:35<00:02, 10601.67it/s] 94%|█████████▍| 376377/400000 [00:35<00:02, 10379.95it/s] 94%|█████████▍| 377418/400000 [00:35<00:02, 10386.14it/s] 95%|█████████▍| 378548/400000 [00:35<00:02, 10642.31it/s] 95%|█████████▍| 379616/400000 [00:36<00:01, 10486.05it/s] 95%|█████████▌| 380777/400000 [00:36<00:01, 10799.09it/s] 95%|█████████▌| 381865/400000 [00:36<00:01, 10821.43it/s] 96%|█████████▌| 382951/400000 [00:36<00:01, 10791.43it/s] 96%|█████████▌| 384071/400000 [00:36<00:01, 10910.33it/s] 96%|█████████▋| 385213/400000 [00:36<00:01, 11056.27it/s] 97%|█████████▋| 386321/400000 [00:36<00:01, 10961.34it/s] 97%|█████████▋| 387419/400000 [00:36<00:01, 10785.81it/s] 97%|█████████▋| 388572/400000 [00:36<00:01, 10998.37it/s] 97%|█████████▋| 389674/400000 [00:36<00:00, 10913.69it/s] 98%|█████████▊| 390768/400000 [00:37<00:00, 10915.83it/s] 98%|█████████▊| 391861/400000 [00:37<00:00, 10730.58it/s] 98%|█████████▊| 392936/400000 [00:37<00:00, 10588.72it/s] 99%|█████████▊| 394048/400000 [00:37<00:00, 10741.93it/s] 99%|█████████▉| 395145/400000 [00:37<00:00, 10806.68it/s] 99%|█████████▉| 396239/400000 [00:37<00:00, 10845.39it/s] 99%|█████████▉| 397387/400000 [00:37<00:00, 11025.89it/s]100%|█████████▉| 398491/400000 [00:37<00:00, 10805.63it/s]100%|█████████▉| 399589/400000 [00:37<00:00, 10855.58it/s]100%|█████████▉| 399999/400000 [00:37<00:00, 10548.16it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f10d3c464e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011260155103407875 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011074553963332671 	 Accuracy: 54

  model saves at 54% accuracy 

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


  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7feb06fc6fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 14:13:31.715503
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 14:13:31.720784
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 14:13:31.724703
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 14:13:31.729245
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7feb12d904a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356052.8438
Epoch 2/10

1/1 [==============================] - 0s 126ms/step - loss: 305262.0938
Epoch 3/10

1/1 [==============================] - 0s 106ms/step - loss: 190883.8125
Epoch 4/10

1/1 [==============================] - 0s 104ms/step - loss: 101918.4141
Epoch 5/10

1/1 [==============================] - 0s 119ms/step - loss: 49305.8086
Epoch 6/10

1/1 [==============================] - 0s 111ms/step - loss: 25830.7383
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 15244.8643
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 9958.7656
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 7051.5854
Epoch 10/10

1/1 [==============================] - 0s 115ms/step - loss: 5342.2271

  #### Inference Need return ypred, ytrue ######################### 
[[ 7.74216115e-01  7.07402229e-02  8.62609208e-01 -3.99095058e-01
  -8.94527137e-01  4.58162606e-01  8.38057578e-01 -1.64466190e+00
   1.14283419e+00 -1.37595820e+00  4.99994159e-02  2.00699449e-01
   7.86212325e-01 -2.78352141e-01 -1.10150194e+00  1.78026688e+00
  -1.35624397e+00  1.93403125e+00  8.80044699e-03 -3.33953679e-01
  -3.85023654e-01 -8.96893322e-01  3.70757163e-01  4.17391658e-01
  -1.20540130e+00 -2.72300899e-01 -7.91625082e-01  1.26448965e+00
  -5.93412220e-01 -6.68605804e-01  1.77316815e-01  2.02765107e-01
  -1.45155644e+00  3.82974356e-01 -1.43480515e+00 -1.22511363e+00
   3.51754546e-01 -1.96644282e+00 -4.88469124e-01 -7.22658992e-01
  -1.90302944e+00  1.70593619e+00  1.06943822e+00  3.67687821e-01
  -9.37310278e-01 -1.69190609e+00 -6.20087266e-01 -4.74762321e-02
   1.35926044e+00 -1.72464103e-01  4.73049909e-01  3.56524467e-01
  -1.72443748e+00 -1.24030685e+00  8.35309982e-01 -2.00154877e+00
  -1.74989462e+00  1.77081990e+00  7.32063890e-01  1.43207598e+00
   5.15568316e-01 -6.73160553e-02  1.12065971e+00 -5.91445625e-01
   2.75856018e-01  4.18667048e-01 -1.45037103e+00  4.57131267e-02
  -2.28714988e-01 -9.62115467e-01 -1.84413105e-01 -1.26190197e+00
  -8.85934770e-01 -3.02638292e-01 -1.28639889e+00 -1.71878904e-01
  -1.71251178e+00 -8.56365323e-01  8.57362688e-01  1.67283356e-01
  -5.61549187e-01 -1.81668139e+00  7.89875984e-01  5.45029402e-01
  -6.91765785e-01 -6.10387325e-02 -3.51704478e-01 -1.82045054e+00
   6.33888423e-01  9.33183432e-01  9.26978216e-02  1.53042495e-01
  -1.54609883e+00  1.02893686e+00 -1.08397388e+00  3.44542205e-01
  -2.31580067e+00 -1.17189264e+00  4.95901316e-01  8.62486720e-01
   8.55318367e-01  1.59478045e+00  2.37569720e-01  1.00160837e+00
  -4.25936192e-01  1.22673273e-01 -1.78696454e-01  1.02949274e+00
  -3.96505713e-01  1.07198858e+00 -3.52519691e-01 -8.71862173e-02
   3.97602230e-01  1.14780796e+00 -1.46176726e-01 -4.21304584e-01
  -1.02032661e+00 -1.13802886e+00  8.61167848e-01 -2.62546003e-01
  -4.71735984e-01  9.22587013e+00  9.54989052e+00  9.93349838e+00
   8.36637020e+00  9.00555134e+00  1.01262684e+01  1.15971546e+01
   9.92638683e+00  8.08898544e+00  1.01569691e+01  9.45930576e+00
   8.76296234e+00  8.31177616e+00  7.95906925e+00  1.05874109e+01
   9.60267353e+00  8.89276314e+00  9.75719261e+00  8.46651077e+00
   1.12601957e+01  1.00110207e+01  8.06702232e+00  8.34741116e+00
   1.03469381e+01  1.01874628e+01  8.39857483e+00  9.46504784e+00
   9.38295746e+00  9.60260010e+00  8.96060371e+00  9.76390743e+00
   9.95604992e+00  9.43445110e+00  8.38619900e+00  9.54273129e+00
   9.67673969e+00  1.00168037e+01  9.20852089e+00  8.37116528e+00
   9.80671597e+00  7.66866302e+00  8.86572838e+00  8.26423836e+00
   1.08108883e+01  7.81827688e+00  8.87051964e+00  8.98856258e+00
   9.11552334e+00  9.84392166e+00  9.33386421e+00  9.61784935e+00
   8.78266811e+00  7.39294529e+00  8.86611938e+00  9.60711956e+00
   1.03261013e+01  1.01607733e+01  8.72826958e+00  1.03833170e+01
   4.25558090e+00  2.13061857e+00  2.40118551e+00  2.70119047e+00
   5.28237283e-01  6.82062626e-01  9.04652655e-01  5.79941213e-01
   1.97970223e+00  1.37378609e+00  2.94848680e+00  9.56723690e-02
   1.12834108e+00  4.52707708e-01  2.45207548e+00  2.49186456e-01
   3.93167138e-01  9.90086317e-01  1.29206657e+00  1.46128130e+00
   4.72230196e-01  3.95357251e-01  7.57349491e-01  1.96197402e+00
   8.16413581e-01  1.61468983e+00  1.28988397e+00  1.75830007e+00
   1.65714860e+00  1.99157917e+00  5.41110516e-01  1.39566815e+00
   1.92856693e+00  1.81998873e+00  1.40827250e+00  8.18409622e-01
   1.89790332e+00  3.27525854e-01  1.15566218e+00  1.45273924e+00
   4.85197306e-01  2.76255369e-01  1.66855574e-01  2.56013155e+00
   3.26551974e-01  1.48784161e+00  3.12464237e-01  2.55128002e+00
   1.35839319e+00  6.42843843e-01  6.85679436e-01  2.01468873e+00
   1.15221608e+00  1.42726636e+00  1.37793112e+00  2.86162424e+00
   2.23043680e-01  2.43543565e-01  1.52015162e+00  1.31814384e+00
   2.53914952e+00  3.08097661e-01  5.39585769e-01  2.06901574e+00
   4.66963291e-01  6.20552659e-01  1.32117414e+00  5.18283367e-01
   1.35673499e+00  1.34690404e-01  2.83052731e+00  4.23644304e-01
   1.86196613e+00  3.26932573e+00  1.36278105e+00  4.07474875e-01
   1.89808178e+00  1.83514726e+00  3.59481275e-01  2.86784530e-01
   2.08504581e+00  1.60856628e+00  1.35886705e+00  1.18586195e+00
   3.50448847e-01  6.65447116e-02  8.04406643e-01  3.11984062e-01
   8.76989961e-01  2.18509674e-01  6.06653690e-01  3.99633169e-01
   7.01581001e-01  2.25930154e-01  3.35540116e-01  1.53601360e+00
   3.04977417e-01  4.43620205e-01  2.30683184e+00  2.44686031e+00
   2.97805905e+00  9.81363356e-01  1.53807402e+00  2.13361359e+00
   1.87774634e+00  9.12602723e-01  2.01253605e+00  2.00298309e+00
   5.00620127e-01  5.09868860e-01  4.94214892e-01  1.36613166e+00
   2.49104118e+00  6.84852958e-01  4.26400185e-01  1.55942917e-01
   5.72868526e-01  1.25771081e+00  4.30457473e-01  3.64765704e-01
   2.90864110e-01  1.03138971e+01  9.16255283e+00  8.67338467e+00
   1.09893093e+01  9.89649296e+00  8.96662426e+00  9.53456974e+00
   1.00249748e+01  7.52724504e+00  8.61332607e+00  1.01188326e+01
   9.95918274e+00  1.10527029e+01  8.74511242e+00  8.59254456e+00
   9.18099594e+00  8.90672398e+00  8.68277359e+00  9.08241749e+00
   8.80418396e+00  1.07409954e+01  9.11999702e+00  9.17339802e+00
   9.08769608e+00  9.95640469e+00  9.92246056e+00  9.71967793e+00
   8.90147114e+00  9.19066334e+00  8.22949791e+00  1.03888350e+01
   9.91431427e+00  1.02033634e+01  9.80471039e+00  8.75379944e+00
   9.12965775e+00  9.98046589e+00  1.03103695e+01  1.07586908e+01
   8.52441025e+00  1.04195709e+01  9.86502552e+00  9.09263706e+00
   9.03736496e+00  8.11574936e+00  9.92485332e+00  9.29641628e+00
   9.34830379e+00  9.14069653e+00  9.65472698e+00  1.01576986e+01
   9.34946537e+00  8.33665943e+00  8.67543411e+00  8.90986538e+00
   8.18761158e+00  9.39719391e+00  9.78472233e+00  1.13318262e+01
  -4.66810465e-02 -1.23608570e+01  1.15600319e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 14:13:41.834692
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2065
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 14:13:41.840563
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8716.74
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 14:13:41.845580
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.5242
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 14:13:41.850247
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -779.646
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140647052072552
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140645976515248
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140645976515752
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140645976516256
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140645976516760
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140645976517264

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7feb006fd588> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.516263
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.479043
grad_step = 000002, loss = 0.452826
grad_step = 000003, loss = 0.422932
grad_step = 000004, loss = 0.385412
grad_step = 000005, loss = 0.345256
grad_step = 000006, loss = 0.315323
grad_step = 000007, loss = 0.316653
grad_step = 000008, loss = 0.309737
grad_step = 000009, loss = 0.286965
grad_step = 000010, loss = 0.266100
grad_step = 000011, loss = 0.254202
grad_step = 000012, loss = 0.248498
grad_step = 000013, loss = 0.243076
grad_step = 000014, loss = 0.234941
grad_step = 000015, loss = 0.224619
grad_step = 000016, loss = 0.213742
grad_step = 000017, loss = 0.203971
grad_step = 000018, loss = 0.195875
grad_step = 000019, loss = 0.189060
grad_step = 000020, loss = 0.182310
grad_step = 000021, loss = 0.174312
grad_step = 000022, loss = 0.165415
grad_step = 000023, loss = 0.157072
grad_step = 000024, loss = 0.150060
grad_step = 000025, loss = 0.144033
grad_step = 000026, loss = 0.138174
grad_step = 000027, loss = 0.131903
grad_step = 000028, loss = 0.125241
grad_step = 000029, loss = 0.118791
grad_step = 000030, loss = 0.113147
grad_step = 000031, loss = 0.108260
grad_step = 000032, loss = 0.103493
grad_step = 000033, loss = 0.098342
grad_step = 000034, loss = 0.092942
grad_step = 000035, loss = 0.087830
grad_step = 000036, loss = 0.083314
grad_step = 000037, loss = 0.079231
grad_step = 000038, loss = 0.075187
grad_step = 000039, loss = 0.070977
grad_step = 000040, loss = 0.066853
grad_step = 000041, loss = 0.063183
grad_step = 000042, loss = 0.059936
grad_step = 000043, loss = 0.056760
grad_step = 000044, loss = 0.053462
grad_step = 000045, loss = 0.050193
grad_step = 000046, loss = 0.047222
grad_step = 000047, loss = 0.044587
grad_step = 000048, loss = 0.042054
grad_step = 000049, loss = 0.039482
grad_step = 000050, loss = 0.037000
grad_step = 000051, loss = 0.034768
grad_step = 000052, loss = 0.032747
grad_step = 000053, loss = 0.030792
grad_step = 000054, loss = 0.028854
grad_step = 000055, loss = 0.027021
grad_step = 000056, loss = 0.025373
grad_step = 000057, loss = 0.023859
grad_step = 000058, loss = 0.022382
grad_step = 000059, loss = 0.020940
grad_step = 000060, loss = 0.019608
grad_step = 000061, loss = 0.018404
grad_step = 000062, loss = 0.017275
grad_step = 000063, loss = 0.016178
grad_step = 000064, loss = 0.015140
grad_step = 000065, loss = 0.014199
grad_step = 000066, loss = 0.013340
grad_step = 000067, loss = 0.012513
grad_step = 000068, loss = 0.011715
grad_step = 000069, loss = 0.010987
grad_step = 000070, loss = 0.010328
grad_step = 000071, loss = 0.009699
grad_step = 000072, loss = 0.009095
grad_step = 000073, loss = 0.008543
grad_step = 000074, loss = 0.008042
grad_step = 000075, loss = 0.007568
grad_step = 000076, loss = 0.007119
grad_step = 000077, loss = 0.006710
grad_step = 000078, loss = 0.006334
grad_step = 000079, loss = 0.005981
grad_step = 000080, loss = 0.005654
grad_step = 000081, loss = 0.005354
grad_step = 000082, loss = 0.005075
grad_step = 000083, loss = 0.004816
grad_step = 000084, loss = 0.004580
grad_step = 000085, loss = 0.004364
grad_step = 000086, loss = 0.004161
grad_step = 000087, loss = 0.003977
grad_step = 000088, loss = 0.003813
grad_step = 000089, loss = 0.003657
grad_step = 000090, loss = 0.003509
grad_step = 000091, loss = 0.003380
grad_step = 000092, loss = 0.003261
grad_step = 000093, loss = 0.003151
grad_step = 000094, loss = 0.003050
grad_step = 000095, loss = 0.002960
grad_step = 000096, loss = 0.002877
grad_step = 000097, loss = 0.002800
grad_step = 000098, loss = 0.002732
grad_step = 000099, loss = 0.002670
grad_step = 000100, loss = 0.002613
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002563
grad_step = 000102, loss = 0.002516
grad_step = 000103, loss = 0.002475
grad_step = 000104, loss = 0.002439
grad_step = 000105, loss = 0.002405
grad_step = 000106, loss = 0.002375
grad_step = 000107, loss = 0.002350
grad_step = 000108, loss = 0.002327
grad_step = 000109, loss = 0.002306
grad_step = 000110, loss = 0.002288
grad_step = 000111, loss = 0.002272
grad_step = 000112, loss = 0.002258
grad_step = 000113, loss = 0.002245
grad_step = 000114, loss = 0.002234
grad_step = 000115, loss = 0.002225
grad_step = 000116, loss = 0.002216
grad_step = 000117, loss = 0.002208
grad_step = 000118, loss = 0.002201
grad_step = 000119, loss = 0.002196
grad_step = 000120, loss = 0.002196
grad_step = 000121, loss = 0.002204
grad_step = 000122, loss = 0.002206
grad_step = 000123, loss = 0.002190
grad_step = 000124, loss = 0.002168
grad_step = 000125, loss = 0.002171
grad_step = 000126, loss = 0.002182
grad_step = 000127, loss = 0.002169
grad_step = 000128, loss = 0.002150
grad_step = 000129, loss = 0.002149
grad_step = 000130, loss = 0.002156
grad_step = 000131, loss = 0.002151
grad_step = 000132, loss = 0.002134
grad_step = 000133, loss = 0.002124
grad_step = 000134, loss = 0.002127
grad_step = 000135, loss = 0.002129
grad_step = 000136, loss = 0.002123
grad_step = 000137, loss = 0.002108
grad_step = 000138, loss = 0.002095
grad_step = 000139, loss = 0.002089
grad_step = 000140, loss = 0.002089
grad_step = 000141, loss = 0.002092
grad_step = 000142, loss = 0.002096
grad_step = 000143, loss = 0.002103
grad_step = 000144, loss = 0.002091
grad_step = 000145, loss = 0.002071
grad_step = 000146, loss = 0.002045
grad_step = 000147, loss = 0.002033
grad_step = 000148, loss = 0.002037
grad_step = 000149, loss = 0.002044
grad_step = 000150, loss = 0.002053
grad_step = 000151, loss = 0.002039
grad_step = 000152, loss = 0.002018
grad_step = 000153, loss = 0.001990
grad_step = 000154, loss = 0.001976
grad_step = 000155, loss = 0.001976
grad_step = 000156, loss = 0.001984
grad_step = 000157, loss = 0.002002
grad_step = 000158, loss = 0.002004
grad_step = 000159, loss = 0.002000
grad_step = 000160, loss = 0.001947
grad_step = 000161, loss = 0.001910
grad_step = 000162, loss = 0.001902
grad_step = 000163, loss = 0.001913
grad_step = 000164, loss = 0.001928
grad_step = 000165, loss = 0.001909
grad_step = 000166, loss = 0.001882
grad_step = 000167, loss = 0.001843
grad_step = 000168, loss = 0.001819
grad_step = 000169, loss = 0.001811
grad_step = 000170, loss = 0.001815
grad_step = 000171, loss = 0.001844
grad_step = 000172, loss = 0.001895
grad_step = 000173, loss = 0.002015
grad_step = 000174, loss = 0.001820
grad_step = 000175, loss = 0.001717
grad_step = 000176, loss = 0.001764
grad_step = 000177, loss = 0.001757
grad_step = 000178, loss = 0.001700
grad_step = 000179, loss = 0.001647
grad_step = 000180, loss = 0.001677
grad_step = 000181, loss = 0.001755
grad_step = 000182, loss = 0.001684
grad_step = 000183, loss = 0.001616
grad_step = 000184, loss = 0.001571
grad_step = 000185, loss = 0.001561
grad_step = 000186, loss = 0.001567
grad_step = 000187, loss = 0.001587
grad_step = 000188, loss = 0.001676
grad_step = 000189, loss = 0.001544
grad_step = 000190, loss = 0.001494
grad_step = 000191, loss = 0.001429
grad_step = 000192, loss = 0.001404
grad_step = 000193, loss = 0.001436
grad_step = 000194, loss = 0.001490
grad_step = 000195, loss = 0.001750
grad_step = 000196, loss = 0.001381
grad_step = 000197, loss = 0.001377
grad_step = 000198, loss = 0.001613
grad_step = 000199, loss = 0.001317
grad_step = 000200, loss = 0.001324
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001590
grad_step = 000202, loss = 0.001298
grad_step = 000203, loss = 0.001324
grad_step = 000204, loss = 0.001527
grad_step = 000205, loss = 0.001280
grad_step = 000206, loss = 0.001447
grad_step = 000207, loss = 0.001928
grad_step = 000208, loss = 0.001340
grad_step = 000209, loss = 0.002254
grad_step = 000210, loss = 0.002423
grad_step = 000211, loss = 0.002185
grad_step = 000212, loss = 0.001638
grad_step = 000213, loss = 0.002239
grad_step = 000214, loss = 0.001999
grad_step = 000215, loss = 0.002391
grad_step = 000216, loss = 0.001431
grad_step = 000217, loss = 0.002541
grad_step = 000218, loss = 0.001389
grad_step = 000219, loss = 0.001922
grad_step = 000220, loss = 0.001684
grad_step = 000221, loss = 0.001423
grad_step = 000222, loss = 0.001721
grad_step = 000223, loss = 0.001283
grad_step = 000224, loss = 0.001644
grad_step = 000225, loss = 0.001211
grad_step = 000226, loss = 0.001510
grad_step = 000227, loss = 0.001249
grad_step = 000228, loss = 0.001308
grad_step = 000229, loss = 0.001346
grad_step = 000230, loss = 0.001129
grad_step = 000231, loss = 0.001332
grad_step = 000232, loss = 0.001092
grad_step = 000233, loss = 0.001229
grad_step = 000234, loss = 0.001070
grad_step = 000235, loss = 0.001180
grad_step = 000236, loss = 0.001025
grad_step = 000237, loss = 0.001100
grad_step = 000238, loss = 0.001041
grad_step = 000239, loss = 0.001013
grad_step = 000240, loss = 0.001030
grad_step = 000241, loss = 0.000981
grad_step = 000242, loss = 0.001019
grad_step = 000243, loss = 0.000939
grad_step = 000244, loss = 0.000988
grad_step = 000245, loss = 0.000929
grad_step = 000246, loss = 0.000955
grad_step = 000247, loss = 0.000910
grad_step = 000248, loss = 0.000928
grad_step = 000249, loss = 0.000899
grad_step = 000250, loss = 0.000901
grad_step = 000251, loss = 0.000882
grad_step = 000252, loss = 0.000879
grad_step = 000253, loss = 0.000866
grad_step = 000254, loss = 0.000860
grad_step = 000255, loss = 0.000846
grad_step = 000256, loss = 0.000844
grad_step = 000257, loss = 0.000829
grad_step = 000258, loss = 0.000826
grad_step = 000259, loss = 0.000813
grad_step = 000260, loss = 0.000809
grad_step = 000261, loss = 0.000798
grad_step = 000262, loss = 0.000793
grad_step = 000263, loss = 0.000783
grad_step = 000264, loss = 0.000778
grad_step = 000265, loss = 0.000768
grad_step = 000266, loss = 0.000765
grad_step = 000267, loss = 0.000753
grad_step = 000268, loss = 0.000751
grad_step = 000269, loss = 0.000740
grad_step = 000270, loss = 0.000737
grad_step = 000271, loss = 0.000728
grad_step = 000272, loss = 0.000724
grad_step = 000273, loss = 0.000715
grad_step = 000274, loss = 0.000712
grad_step = 000275, loss = 0.000703
grad_step = 000276, loss = 0.000700
grad_step = 000277, loss = 0.000692
grad_step = 000278, loss = 0.000687
grad_step = 000279, loss = 0.000680
grad_step = 000280, loss = 0.000676
grad_step = 000281, loss = 0.000669
grad_step = 000282, loss = 0.000664
grad_step = 000283, loss = 0.000658
grad_step = 000284, loss = 0.000654
grad_step = 000285, loss = 0.000648
grad_step = 000286, loss = 0.000643
grad_step = 000287, loss = 0.000638
grad_step = 000288, loss = 0.000632
grad_step = 000289, loss = 0.000627
grad_step = 000290, loss = 0.000622
grad_step = 000291, loss = 0.000618
grad_step = 000292, loss = 0.000612
grad_step = 000293, loss = 0.000608
grad_step = 000294, loss = 0.000603
grad_step = 000295, loss = 0.000599
grad_step = 000296, loss = 0.000594
grad_step = 000297, loss = 0.000589
grad_step = 000298, loss = 0.000585
grad_step = 000299, loss = 0.000580
grad_step = 000300, loss = 0.000576
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000572
grad_step = 000302, loss = 0.000567
grad_step = 000303, loss = 0.000563
grad_step = 000304, loss = 0.000559
grad_step = 000305, loss = 0.000555
grad_step = 000306, loss = 0.000551
grad_step = 000307, loss = 0.000547
grad_step = 000308, loss = 0.000543
grad_step = 000309, loss = 0.000539
grad_step = 000310, loss = 0.000535
grad_step = 000311, loss = 0.000532
grad_step = 000312, loss = 0.000528
grad_step = 000313, loss = 0.000524
grad_step = 000314, loss = 0.000521
grad_step = 000315, loss = 0.000517
grad_step = 000316, loss = 0.000514
grad_step = 000317, loss = 0.000510
grad_step = 000318, loss = 0.000507
grad_step = 000319, loss = 0.000503
grad_step = 000320, loss = 0.000500
grad_step = 000321, loss = 0.000497
grad_step = 000322, loss = 0.000494
grad_step = 000323, loss = 0.000490
grad_step = 000324, loss = 0.000487
grad_step = 000325, loss = 0.000484
grad_step = 000326, loss = 0.000481
grad_step = 000327, loss = 0.000478
grad_step = 000328, loss = 0.000475
grad_step = 000329, loss = 0.000473
grad_step = 000330, loss = 0.000470
grad_step = 000331, loss = 0.000467
grad_step = 000332, loss = 0.000464
grad_step = 000333, loss = 0.000461
grad_step = 000334, loss = 0.000459
grad_step = 000335, loss = 0.000456
grad_step = 000336, loss = 0.000454
grad_step = 000337, loss = 0.000451
grad_step = 000338, loss = 0.000448
grad_step = 000339, loss = 0.000446
grad_step = 000340, loss = 0.000444
grad_step = 000341, loss = 0.000441
grad_step = 000342, loss = 0.000439
grad_step = 000343, loss = 0.000437
grad_step = 000344, loss = 0.000435
grad_step = 000345, loss = 0.000433
grad_step = 000346, loss = 0.000431
grad_step = 000347, loss = 0.000430
grad_step = 000348, loss = 0.000429
grad_step = 000349, loss = 0.000430
grad_step = 000350, loss = 0.000434
grad_step = 000351, loss = 0.000441
grad_step = 000352, loss = 0.000455
grad_step = 000353, loss = 0.000457
grad_step = 000354, loss = 0.000464
grad_step = 000355, loss = 0.000450
grad_step = 000356, loss = 0.000439
grad_step = 000357, loss = 0.000423
grad_step = 000358, loss = 0.000413
grad_step = 000359, loss = 0.000406
grad_step = 000360, loss = 0.000404
grad_step = 000361, loss = 0.000404
grad_step = 000362, loss = 0.000405
grad_step = 000363, loss = 0.000408
grad_step = 000364, loss = 0.000415
grad_step = 000365, loss = 0.000430
grad_step = 000366, loss = 0.000438
grad_step = 000367, loss = 0.000454
grad_step = 000368, loss = 0.000440
grad_step = 000369, loss = 0.000429
grad_step = 000370, loss = 0.000407
grad_step = 000371, loss = 0.000393
grad_step = 000372, loss = 0.000387
grad_step = 000373, loss = 0.000389
grad_step = 000374, loss = 0.000395
grad_step = 000375, loss = 0.000402
grad_step = 000376, loss = 0.000408
grad_step = 000377, loss = 0.000403
grad_step = 000378, loss = 0.000398
grad_step = 000379, loss = 0.000391
grad_step = 000380, loss = 0.000387
grad_step = 000381, loss = 0.000382
grad_step = 000382, loss = 0.000380
grad_step = 000383, loss = 0.000377
grad_step = 000384, loss = 0.000376
grad_step = 000385, loss = 0.000375
grad_step = 000386, loss = 0.000376
grad_step = 000387, loss = 0.000377
grad_step = 000388, loss = 0.000379
grad_step = 000389, loss = 0.000384
grad_step = 000390, loss = 0.000392
grad_step = 000391, loss = 0.000399
grad_step = 000392, loss = 0.000411
grad_step = 000393, loss = 0.000402
grad_step = 000394, loss = 0.000397
grad_step = 000395, loss = 0.000384
grad_step = 000396, loss = 0.000374
grad_step = 000397, loss = 0.000365
grad_step = 000398, loss = 0.000360
grad_step = 000399, loss = 0.000359
grad_step = 000400, loss = 0.000362
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000367
grad_step = 000402, loss = 0.000374
grad_step = 000403, loss = 0.000385
grad_step = 000404, loss = 0.000392
grad_step = 000405, loss = 0.000404
grad_step = 000406, loss = 0.000394
grad_step = 000407, loss = 0.000388
grad_step = 000408, loss = 0.000373
grad_step = 000409, loss = 0.000361
grad_step = 000410, loss = 0.000354
grad_step = 000411, loss = 0.000351
grad_step = 000412, loss = 0.000352
grad_step = 000413, loss = 0.000356
grad_step = 000414, loss = 0.000361
grad_step = 000415, loss = 0.000364
grad_step = 000416, loss = 0.000367
grad_step = 000417, loss = 0.000367
grad_step = 000418, loss = 0.000367
grad_step = 000419, loss = 0.000364
grad_step = 000420, loss = 0.000364
grad_step = 000421, loss = 0.000361
grad_step = 000422, loss = 0.000361
grad_step = 000423, loss = 0.000359
grad_step = 000424, loss = 0.000358
grad_step = 000425, loss = 0.000354
grad_step = 000426, loss = 0.000351
grad_step = 000427, loss = 0.000347
grad_step = 000428, loss = 0.000345
grad_step = 000429, loss = 0.000343
grad_step = 000430, loss = 0.000341
grad_step = 000431, loss = 0.000340
grad_step = 000432, loss = 0.000340
grad_step = 000433, loss = 0.000339
grad_step = 000434, loss = 0.000339
grad_step = 000435, loss = 0.000339
grad_step = 000436, loss = 0.000339
grad_step = 000437, loss = 0.000339
grad_step = 000438, loss = 0.000339
grad_step = 000439, loss = 0.000342
grad_step = 000440, loss = 0.000346
grad_step = 000441, loss = 0.000357
grad_step = 000442, loss = 0.000374
grad_step = 000443, loss = 0.000413
grad_step = 000444, loss = 0.000402
grad_step = 000445, loss = 0.000405
grad_step = 000446, loss = 0.000378
grad_step = 000447, loss = 0.000357
grad_step = 000448, loss = 0.000343
grad_step = 000449, loss = 0.000336
grad_step = 000450, loss = 0.000335
grad_step = 000451, loss = 0.000340
grad_step = 000452, loss = 0.000346
grad_step = 000453, loss = 0.000355
grad_step = 000454, loss = 0.000367
grad_step = 000455, loss = 0.000370
grad_step = 000456, loss = 0.000380
grad_step = 000457, loss = 0.000364
grad_step = 000458, loss = 0.000356
grad_step = 000459, loss = 0.000346
grad_step = 000460, loss = 0.000339
grad_step = 000461, loss = 0.000334
grad_step = 000462, loss = 0.000331
grad_step = 000463, loss = 0.000331
grad_step = 000464, loss = 0.000333
grad_step = 000465, loss = 0.000335
grad_step = 000466, loss = 0.000338
grad_step = 000467, loss = 0.000342
grad_step = 000468, loss = 0.000346
grad_step = 000469, loss = 0.000352
grad_step = 000470, loss = 0.000356
grad_step = 000471, loss = 0.000367
grad_step = 000472, loss = 0.000363
grad_step = 000473, loss = 0.000365
grad_step = 000474, loss = 0.000356
grad_step = 000475, loss = 0.000348
grad_step = 000476, loss = 0.000339
grad_step = 000477, loss = 0.000332
grad_step = 000478, loss = 0.000328
grad_step = 000479, loss = 0.000328
grad_step = 000480, loss = 0.000331
grad_step = 000481, loss = 0.000335
grad_step = 000482, loss = 0.000338
grad_step = 000483, loss = 0.000339
grad_step = 000484, loss = 0.000341
grad_step = 000485, loss = 0.000341
grad_step = 000486, loss = 0.000342
grad_step = 000487, loss = 0.000342
grad_step = 000488, loss = 0.000344
grad_step = 000489, loss = 0.000343
grad_step = 000490, loss = 0.000343
grad_step = 000491, loss = 0.000341
grad_step = 000492, loss = 0.000338
grad_step = 000493, loss = 0.000334
grad_step = 000494, loss = 0.000330
grad_step = 000495, loss = 0.000327
grad_step = 000496, loss = 0.000325
grad_step = 000497, loss = 0.000325
grad_step = 000498, loss = 0.000325
grad_step = 000499, loss = 0.000327
grad_step = 000500, loss = 0.000329
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000332
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

  date_run                              2020-05-13 14:14:08.155006
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.237791
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 14:14:08.163389
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.145268
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 14:14:08.171876
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.129177
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 14:14:08.179278
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -1.2074
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
0   2020-05-13 14:13:31.715503  ...    mean_absolute_error
1   2020-05-13 14:13:31.720784  ...     mean_squared_error
2   2020-05-13 14:13:31.724703  ...  median_absolute_error
3   2020-05-13 14:13:31.729245  ...               r2_score
4   2020-05-13 14:13:41.834692  ...    mean_absolute_error
5   2020-05-13 14:13:41.840563  ...     mean_squared_error
6   2020-05-13 14:13:41.845580  ...  median_absolute_error
7   2020-05-13 14:13:41.850247  ...               r2_score
8   2020-05-13 14:14:08.155006  ...    mean_absolute_error
9   2020-05-13 14:14:08.163389  ...     mean_squared_error
10  2020-05-13 14:14:08.171876  ...  median_absolute_error
11  2020-05-13 14:14:08.179278  ...               r2_score

[12 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do timeseries 





 ************************************************************************************************************************

  vision_mnist 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_cnn/mnist 

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb6d82b0fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 16%|█▌        | 1589248/9912422 [00:00<00:00, 15384519.54it/s] 73%|███████▎  | 7241728/9912422 [00:00<00:00, 19627801.57it/s]9920512it [00:00, 27231138.90it/s]                             
0it [00:00, ?it/s]32768it [00:00, 489927.79it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 460937.47it/s]1654784it [00:00, 11650477.56it/s]                         
0it [00:00, ?it/s]8192it [00:00, 150661.62it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68acb3eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68a2df0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68acb3eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68a238128> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 

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
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb687a73518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb687a5d780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68acb3eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68a1f5748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb687a73518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
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
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb68a0b0588> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7ffd723208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=089429b35c4d3fc62c899eceed3a173af4c7e794020c67e3114b9bcfd44ccda1
  Stored in directory: /tmp/pip-ephem-wheel-cache-aqqmdzba/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7f9551e748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2646016/17464789 [===>..........................] - ETA: 0s
11157504/17464789 [==================>...........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 14:15:37.216837: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 14:15:37.221020: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 14:15:37.221188: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558f173f8ca0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 14:15:37.221207: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8046 - accuracy: 0.4910
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7740 - accuracy: 0.4930 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7241 - accuracy: 0.4963
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7034 - accuracy: 0.4976
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6590 - accuracy: 0.5005
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6579 - accuracy: 0.5006
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5934 - accuracy: 0.5048
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 4s - loss: 7.6387 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 3s - loss: 7.6431 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6785 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6731 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6462 - accuracy: 0.5013
25000/25000 [==============================] - 10s 409us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 14:15:55.525973
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 14:15:55.525973  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}] 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<127:51:03, 1.87kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<89:43:40, 2.67kB/s] .vector_cache/glove.6B.zip:   0%|          | 106k/862M [00:04<63:06:41, 3.79kB/s] .vector_cache/glove.6B.zip:   0%|          | 442k/862M [00:04<44:11:04, 5.42kB/s].vector_cache/glove.6B.zip:   0%|          | 1.06M/862M [00:04<30:55:06, 7.74kB/s].vector_cache/glove.6B.zip:   0%|          | 2.94M/862M [00:05<21:35:58, 11.1kB/s].vector_cache/glove.6B.zip:   1%|          | 6.54M/862M [00:05<15:03:30, 15.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.8M/862M [00:05<10:29:22, 22.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:05<7:18:24, 32.2kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.5M/862M [00:05<5:05:25, 46.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:05<3:32:49, 65.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.3M/862M [00:05<2:28:16, 93.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:05<1:43:20, 134kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:05<1:12:02, 191kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:05<50:19, 272kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:06<35:06, 388kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:06<24:32, 551kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:06<17:28, 773kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:08<14:05, 954kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:08<13:35, 989kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:08<10:25, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:08<07:30, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:10<10:59, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:10<09:10, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:10<06:47, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:12<07:37, 1.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:12<08:03, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:12<06:19, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:14<06:33, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:14<05:58, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:14<04:30, 2.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:16<06:14, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:16<07:05, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<05:37, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:18<06:02, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:18<05:34, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:18<04:11, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:20<05:59, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:20<05:32, 2.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:20<04:12, 3.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:22<05:57, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:22<06:52, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<05:23, 2.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:22<03:59, 3.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:24<07:12, 1.79MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:24<06:24, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:24<04:48, 2.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:26<06:21, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:26<07:06, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<05:38, 2.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:26<04:06, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:27<1:33:38, 136kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:28<1:06:49, 191kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:28<47:01, 271kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:29<35:47, 354kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<27:41, 458kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<20:01, 633kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:31<15:59, 789kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:32<12:30, 1.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<09:01, 1.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:33<09:14, 1.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<07:45, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<05:44, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:35<06:57, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<06:08, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<04:36, 2.70MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:37<06:26, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<04:47, 2.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<05:20, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<04:03, 3.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<05:43, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<05:16, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<03:57, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<05:13, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<03:57, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<05:38, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<05:10, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<03:55, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:47<05:36, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:47<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<03:53, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:49<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:49<06:21, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<04:58, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<03:45, 3.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:51<05:45, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<05:02, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<03:59, 2.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<02:54, 4.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:53<49:53, 238kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:53<36:06, 328kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<25:31, 464kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:55<20:36, 572kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<15:37, 755kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<11:12, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<10:36, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<08:36, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<06:16, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<07:09, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<07:22, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<05:44, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<05:53, 1.97MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<05:19, 2.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<04:00, 2.88MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<05:30, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<05:01, 2.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<03:48, 3.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:05<05:22, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:05<04:55, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:05<03:41, 3.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:07<05:15, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<03:40, 3.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:09<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<04:49, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<03:39, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:11<05:13, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<04:47, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<03:38, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:10, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<03:36, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<05:09, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<05:52, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<04:40, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<05:02, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<04:39, 2.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<03:32, 3.11MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<05:02, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<04:26, 2.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<03:22, 3.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<02:29, 4.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<42:57, 254kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:21<31:10, 350kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<22:02, 493kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<17:56, 604kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:23<14:47, 732kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<10:53, 993kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<07:42, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<17:43, 608kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<13:29, 797kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<09:39, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:26<09:15, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<08:39, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<06:32, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<04:44, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:28<07:21, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<06:15, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:29<04:38, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<05:43, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<06:13, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:31<04:52, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<03:33, 2.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<07:19, 1.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<06:12, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<04:35, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<05:38, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<06:05, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<04:47, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<05:01, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<04:35, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<03:26, 3.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<04:48, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<04:45, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<04:22, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<03:17, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<05:22, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<04:16, 2.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<03:04, 3.28MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:44<38:51, 259kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<28:14, 357kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<19:58, 503kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<16:15, 616kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<14:44, 679kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<10:45, 930kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<07:38, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<09:29, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<07:26, 1.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<05:31, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<05:42, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<06:05, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<04:43, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<03:24, 2.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<10:45, 912kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<08:32, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<06:10, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<06:35, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<06:35, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<05:07, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<05:07, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<04:35, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<03:26, 2.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<04:38, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<05:12, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<04:07, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:00<04:25, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<04:04, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<03:02, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<04:21, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<04:55, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<03:51, 2.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<02:51, 3.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<05:07, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<04:33, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<03:25, 2.73MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<04:34, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<04:09, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<03:08, 2.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<04:22, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<04:00, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<03:02, 3.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:10<04:17, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:10<04:52, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<03:52, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<02:47, 3.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:12<8:49:14, 17.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<6:11:09, 24.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<4:19:18, 35.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<3:02:55, 49.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<2:09:52, 69.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<1:31:11, 99.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<1:03:40, 141kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:16<49:54, 180kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<35:50, 250kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<25:15, 354kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<19:40, 453kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<14:40, 607kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<10:27, 850kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:19<09:23, 942kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:20<08:17, 1.07MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<06:15, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:22<04:57, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<03:40, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:35, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:59, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<03:54, 2.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<02:51, 3.04MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<05:44, 1.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<04:47, 1.80MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:26<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<02:46, 3.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<06:36, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<05:29, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<04:01, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<02:54, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:29<1:05:19, 130kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<46:33, 182kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<32:42, 259kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<24:47, 340kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<19:04, 442kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<13:40, 616kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<09:47, 858kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<08:47, 952kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<06:51, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<05:10, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<03:42, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<19:59, 415kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<15:41, 528kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<11:19, 731kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<08:04, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<08:01, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<06:29, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:44, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<05:12, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<04:28, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:39<03:19, 2.44MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<04:14, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:41<03:47, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<02:51, 2.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<03:53, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<03:31, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<02:39, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<03:44, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<04:14, 1.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:45<03:18, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<02:27, 3.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:47<04:10, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:47<03:44, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<02:48, 2.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<03:46, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:49<03:26, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<02:33, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:51<03:37, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<04:03, 1.91MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<03:11, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<02:21, 3.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<04:25, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<03:53, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<02:54, 2.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<03:48, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<03:27, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<02:36, 2.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<03:35, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<03:16, 2.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<02:28, 3.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<03:55, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<03:04, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<02:19, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:01<03:33, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:01<03:16, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<02:28, 2.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:03<03:26, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:03<03:57, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<03:05, 2.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<02:15, 3.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:05<05:42, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:05<04:44, 1.53MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<03:27, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:07<04:06, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:07<03:36, 1.99MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<02:41, 2.65MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<03:34, 2.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:09<03:12, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<02:23, 2.96MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<03:21, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<03:03, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<02:18, 3.03MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:16, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:13<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<02:16, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:13, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<02:57, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<02:14, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<03:11, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:17<02:49, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<02:11, 3.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<01:35, 4.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<42:10, 161kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<30:10, 225kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<21:13, 318kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<16:21, 411kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<12:01, 558kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<08:33, 782kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<06:02, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<29:18, 227kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<21:10, 314kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<14:54, 444kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<11:55, 552kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<09:41, 679kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<07:03, 930kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<05:06, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<05:03, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<04:07, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<02:15, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<25:30, 253kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<19:10, 336kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<13:41, 470kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<09:38, 663kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<08:59, 709kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<06:56, 917kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<04:59, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<03:33, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<25:55, 243kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<19:27, 324kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:32<13:54, 452kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<09:44, 641kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<13:52, 449kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<10:21, 602kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<07:22, 842kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<06:34, 939kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<05:14, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<03:48, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:38<04:05, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<03:29, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:34, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:13, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<02:52, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:09, 2.77MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:42<02:54, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<02:39, 2.25MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<01:59, 2.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<02:47, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<02:33, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<01:54, 3.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:46<02:43, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:46<03:03, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<02:23, 2.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<01:45, 3.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<02:50, 2.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:07, 2.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<02:49, 2.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<03:10, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<02:31, 2.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<01:49, 3.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<41:24, 136kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<29:32, 190kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<20:43, 270kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<15:42, 353kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<11:28, 484kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<08:07, 679kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<05:43, 958kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<24:38, 222kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<17:47, 308kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<12:30, 436kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<09:59, 542kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<07:32, 717kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<05:23, 999kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<05:00, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:00<04:02, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:57, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<03:18, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<02:51, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<02:07, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<02:42, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<02:20, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<02:27, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<02:15, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<01:42, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:07<02:21, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<02:10, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<01:37, 3.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<02:19, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<02:07, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<01:36, 3.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<02:17, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<02:37, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:04, 2.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:13<02:13, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:13<02:04, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<01:33, 3.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<02:12, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<02:02, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<01:32, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<02:27, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<01:55, 2.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<01:25, 3.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<02:39, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<02:20, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<01:44, 2.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<02:17, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<02:32, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<02:00, 2.28MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<02:07, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<01:52, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:23<01:25, 3.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<02:01, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<02:19, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<01:51, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:27<01:59, 2.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:27<01:50, 2.38MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<01:22, 3.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<01:00, 4.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<16:45, 257kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<12:36, 342kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<09:01, 477kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<06:16, 676kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<39:46, 107kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:31<28:15, 150kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<19:46, 213kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<14:41, 284kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<10:41, 390kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<07:32, 549kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:35<06:13, 661kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<04:45, 862kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<03:25, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<03:19, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<03:09, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:24, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<02:18, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<02:01, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<01:30, 2.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<01:57, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<01:42, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<01:18, 2.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<00:57, 4.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<16:05, 238kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<11:38, 329kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<08:10, 465kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:45<06:34, 574kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:45<05:21, 702kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<03:56, 953kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<02:45, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:47<3:34:09, 17.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<2:30:02, 24.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<1:44:22, 35.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:49<1:13:12, 49.6kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:49<51:56, 69.9kB/s]  .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<36:23, 99.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<25:15, 142kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:51<19:26, 183kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<13:56, 255kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<09:46, 361kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<07:35, 460kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<06:01, 579kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<04:21, 798kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<03:04, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<03:36, 949kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<02:53, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<02:06, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<02:12, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<02:13, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<01:42, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<01:14, 2.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<01:57, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:59<01:38, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<01:17, 2.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<00:55, 3.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<04:07, 779kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:01<03:32, 906kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<02:38, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<02:19, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<01:57, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<01:26, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<01:42, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<01:30, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<01:07, 2.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<01:29, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<01:20, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:00, 2.94MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:08<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<01:16, 2.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<00:57, 3.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:10<01:20, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:10<01:32, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<01:11, 2.39MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<00:53, 3.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:12<01:25, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:12<01:17, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<00:57, 2.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<01:18, 2.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<01:11, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:14<00:53, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<01:09, 2.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:16<00:52, 3.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:12, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:06, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:18<00:50, 3.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<01:10, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:20<01:05, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:20<00:48, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<01:08, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<00:47, 3.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<01:06, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<01:16, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<01:00, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:26<01:04, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<00:59, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<00:44, 3.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:02, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:11, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<00:56, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<00:40, 3.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<02:10, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:44, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:30<01:15, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<01:22, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<01:10, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<00:52, 2.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<01:04, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<00:57, 2.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<00:43, 2.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<00:31, 3.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<15:06, 131kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<10:42, 185kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<07:32, 261kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<05:10, 371kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<16:17, 118kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<11:34, 165kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<08:02, 234kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<05:58, 309kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<04:33, 405kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<03:15, 562kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<02:30, 711kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<01:55, 919kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<01:22, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<01:20, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<01:06, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<00:48, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<00:56, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<00:59, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:46<00:46, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:32, 2.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<01:39, 949kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<01:17, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<00:56, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:48<00:39, 2.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<04:05, 367kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<03:00, 498kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<02:07, 696kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<01:27, 983kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<11:42, 123kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:52<08:28, 169kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<05:56, 239kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<04:13, 323kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:54<03:05, 440kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<02:09, 620kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:46, 733kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<01:22, 945kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:58, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<00:57, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:58<00:55, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<00:42, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:29, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<01:12, 961kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:57, 1.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<00:41, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:43, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:37, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:26, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:32, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:29, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:21, 2.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:27, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:25, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:05<00:18, 3.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:13, 4.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<06:45, 132kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<04:53, 181kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<03:25, 255kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<02:15, 363kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<03:29, 235kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<02:29, 326kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<01:44, 459kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<01:09, 651kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<02:42, 278kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<02:02, 367kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<01:25, 514kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:11<00:59, 719kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:49, 827kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:38, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:26, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:26, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:22, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:15, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<00:18, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:15, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:11, 2.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:14, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:13, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:12, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:19<00:11, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<00:10, 2.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:09, 2.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:08, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:07, 3.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:07, 3.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:06, 3.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<01:17, 315kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:57, 422kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:40, 581kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:29, 781kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:21, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:15, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:12, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:09, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:22, 904kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:25, 788kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:20, 994kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:14, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<00:10, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:08, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:06, 2.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:05, 3.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:20, 804kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:20, 777kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:15, 997kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:11, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:25<00:08, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:25<00:06, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:04, 2.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:12, 966kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:13, 906kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:09, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:27<00:05, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:03, 2.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:02, 3.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:06, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:07, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:05, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:03, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<00:02, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:01, 2.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:01, 3.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:04, 858kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:04, 874kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:02, 1.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:31<00:01, 1.53MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:31<00:00, 2.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:00, 2.57MB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 740/400000 [00:00<00:53, 7394.89it/s]  0%|          | 1482/400000 [00:00<00:53, 7401.57it/s]  1%|          | 2240/400000 [00:00<00:53, 7453.78it/s]  1%|          | 2964/400000 [00:00<00:53, 7387.51it/s]  1%|          | 3717/400000 [00:00<00:53, 7428.38it/s]  1%|          | 4465/400000 [00:00<00:53, 7440.31it/s]  1%|▏         | 5227/400000 [00:00<00:52, 7490.58it/s]  1%|▏         | 5943/400000 [00:00<00:53, 7388.03it/s]  2%|▏         | 6697/400000 [00:00<00:52, 7431.81it/s]  2%|▏         | 7431/400000 [00:01<00:53, 7402.47it/s]  2%|▏         | 8151/400000 [00:01<00:53, 7338.69it/s]  2%|▏         | 8871/400000 [00:01<00:54, 7205.60it/s]  2%|▏         | 9624/400000 [00:01<00:53, 7298.03it/s]  3%|▎         | 10362/400000 [00:01<00:53, 7320.30it/s]  3%|▎         | 11129/400000 [00:01<00:52, 7420.20it/s]  3%|▎         | 11869/400000 [00:01<00:53, 7303.82it/s]  3%|▎         | 12632/400000 [00:01<00:52, 7397.65it/s]  3%|▎         | 13396/400000 [00:01<00:51, 7467.55it/s]  4%|▎         | 14143/400000 [00:01<00:52, 7318.00it/s]  4%|▎         | 14890/400000 [00:02<00:52, 7360.74it/s]  4%|▍         | 15627/400000 [00:02<00:52, 7358.69it/s]  4%|▍         | 16374/400000 [00:02<00:51, 7388.86it/s]  4%|▍         | 17132/400000 [00:02<00:51, 7444.91it/s]  4%|▍         | 17877/400000 [00:02<00:51, 7441.49it/s]  5%|▍         | 18631/400000 [00:02<00:51, 7469.76it/s]  5%|▍         | 19379/400000 [00:02<00:51, 7426.77it/s]  5%|▌         | 20122/400000 [00:02<00:51, 7426.15it/s]  5%|▌         | 20865/400000 [00:02<00:51, 7360.29it/s]  5%|▌         | 21626/400000 [00:02<00:50, 7431.27it/s]  6%|▌         | 22379/400000 [00:03<00:50, 7458.66it/s]  6%|▌         | 23126/400000 [00:03<00:50, 7445.75it/s]  6%|▌         | 23875/400000 [00:03<00:50, 7458.27it/s]  6%|▌         | 24624/400000 [00:03<00:50, 7466.24it/s]  6%|▋         | 25377/400000 [00:03<00:50, 7483.18it/s]  7%|▋         | 26126/400000 [00:03<00:50, 7474.86it/s]  7%|▋         | 26874/400000 [00:03<00:50, 7446.00it/s]  7%|▋         | 27632/400000 [00:03<00:49, 7483.05it/s]  7%|▋         | 28397/400000 [00:03<00:49, 7531.96it/s]  7%|▋         | 29160/400000 [00:03<00:49, 7560.24it/s]  7%|▋         | 29917/400000 [00:04<00:49, 7520.82it/s]  8%|▊         | 30670/400000 [00:04<00:49, 7453.56it/s]  8%|▊         | 31416/400000 [00:04<00:49, 7446.89it/s]  8%|▊         | 32173/400000 [00:04<00:49, 7480.78it/s]  8%|▊         | 32933/400000 [00:04<00:48, 7515.66it/s]  8%|▊         | 33689/400000 [00:04<00:48, 7528.31it/s]  9%|▊         | 34442/400000 [00:04<00:48, 7463.27it/s]  9%|▉         | 35191/400000 [00:04<00:48, 7470.41it/s]  9%|▉         | 35951/400000 [00:04<00:48, 7506.41it/s]  9%|▉         | 36702/400000 [00:04<00:48, 7495.00it/s]  9%|▉         | 37462/400000 [00:05<00:48, 7526.14it/s] 10%|▉         | 38215/400000 [00:05<00:48, 7466.62it/s] 10%|▉         | 38962/400000 [00:05<00:48, 7406.45it/s] 10%|▉         | 39712/400000 [00:05<00:48, 7432.55it/s] 10%|█         | 40456/400000 [00:05<00:48, 7411.31it/s] 10%|█         | 41227/400000 [00:05<00:47, 7496.24it/s] 10%|█         | 41977/400000 [00:05<00:47, 7490.78it/s] 11%|█         | 42727/400000 [00:05<00:47, 7448.39it/s] 11%|█         | 43481/400000 [00:05<00:47, 7475.06it/s] 11%|█         | 44252/400000 [00:05<00:47, 7542.76it/s] 11%|█▏        | 45020/400000 [00:06<00:46, 7582.55it/s] 11%|█▏        | 45779/400000 [00:06<00:46, 7568.72it/s] 12%|█▏        | 46543/400000 [00:06<00:46, 7588.56it/s] 12%|█▏        | 47308/400000 [00:06<00:46, 7606.86it/s] 12%|█▏        | 48069/400000 [00:06<00:46, 7604.40it/s] 12%|█▏        | 48832/400000 [00:06<00:46, 7609.79it/s] 12%|█▏        | 49594/400000 [00:06<00:46, 7585.48it/s] 13%|█▎        | 50353/400000 [00:06<00:46, 7538.41it/s] 13%|█▎        | 51107/400000 [00:06<00:46, 7535.30it/s] 13%|█▎        | 51862/400000 [00:06<00:46, 7537.62it/s] 13%|█▎        | 52616/400000 [00:07<00:46, 7413.40it/s] 13%|█▎        | 53364/400000 [00:07<00:46, 7431.54it/s] 14%|█▎        | 54108/400000 [00:07<00:46, 7423.42it/s] 14%|█▎        | 54879/400000 [00:07<00:45, 7505.26it/s] 14%|█▍        | 55640/400000 [00:07<00:45, 7534.39it/s] 14%|█▍        | 56394/400000 [00:07<00:45, 7521.67it/s] 14%|█▍        | 57147/400000 [00:07<00:45, 7489.25it/s] 14%|█▍        | 57897/400000 [00:07<00:45, 7469.81it/s] 15%|█▍        | 58645/400000 [00:07<00:45, 7432.15it/s] 15%|█▍        | 59409/400000 [00:07<00:45, 7490.64it/s] 15%|█▌        | 60174/400000 [00:08<00:45, 7535.70it/s] 15%|█▌        | 60948/400000 [00:08<00:44, 7593.06it/s] 15%|█▌        | 61708/400000 [00:08<00:44, 7565.58it/s] 16%|█▌        | 62470/400000 [00:08<00:44, 7581.32it/s] 16%|█▌        | 63240/400000 [00:08<00:44, 7616.17it/s] 16%|█▌        | 64011/400000 [00:08<00:43, 7642.73it/s] 16%|█▌        | 64783/400000 [00:08<00:43, 7662.81it/s] 16%|█▋        | 65552/400000 [00:08<00:43, 7669.90it/s] 17%|█▋        | 66327/400000 [00:08<00:43, 7692.64it/s] 17%|█▋        | 67097/400000 [00:08<00:43, 7661.06it/s] 17%|█▋        | 67864/400000 [00:09<00:43, 7656.02it/s] 17%|█▋        | 68631/400000 [00:09<00:43, 7658.23it/s] 17%|█▋        | 69397/400000 [00:09<00:43, 7563.57it/s] 18%|█▊        | 70172/400000 [00:09<00:43, 7617.11it/s] 18%|█▊        | 70935/400000 [00:09<00:44, 7455.41it/s] 18%|█▊        | 71696/400000 [00:09<00:43, 7468.58it/s] 18%|█▊        | 72455/400000 [00:09<00:43, 7504.35it/s] 18%|█▊        | 73206/400000 [00:09<00:43, 7501.72it/s] 18%|█▊        | 73980/400000 [00:09<00:43, 7571.07it/s] 19%|█▊        | 74749/400000 [00:09<00:42, 7606.14it/s] 19%|█▉        | 75510/400000 [00:10<00:42, 7564.22it/s] 19%|█▉        | 76277/400000 [00:10<00:42, 7595.16it/s] 19%|█▉        | 77037/400000 [00:10<00:42, 7520.89it/s] 19%|█▉        | 77806/400000 [00:10<00:42, 7569.65it/s] 20%|█▉        | 78567/400000 [00:10<00:42, 7581.39it/s] 20%|█▉        | 79326/400000 [00:10<00:42, 7581.20it/s] 20%|██        | 80085/400000 [00:10<00:42, 7533.93it/s] 20%|██        | 80839/400000 [00:10<00:42, 7440.31it/s] 20%|██        | 81608/400000 [00:10<00:42, 7513.19it/s] 21%|██        | 82366/400000 [00:10<00:42, 7531.15it/s] 21%|██        | 83121/400000 [00:11<00:42, 7534.16it/s] 21%|██        | 83875/400000 [00:11<00:42, 7518.91it/s] 21%|██        | 84628/400000 [00:11<00:43, 7272.18it/s] 21%|██▏       | 85389/400000 [00:11<00:42, 7369.51it/s] 22%|██▏       | 86149/400000 [00:11<00:42, 7436.31it/s] 22%|██▏       | 86899/400000 [00:11<00:41, 7455.01it/s] 22%|██▏       | 87646/400000 [00:11<00:41, 7447.62it/s] 22%|██▏       | 88393/400000 [00:11<00:41, 7453.00it/s] 22%|██▏       | 89139/400000 [00:11<00:41, 7446.93it/s] 22%|██▏       | 89896/400000 [00:12<00:41, 7481.47it/s] 23%|██▎       | 90648/400000 [00:12<00:41, 7490.54it/s] 23%|██▎       | 91398/400000 [00:12<00:41, 7489.33it/s] 23%|██▎       | 92148/400000 [00:12<00:41, 7436.59it/s] 23%|██▎       | 92903/400000 [00:12<00:41, 7469.09it/s] 23%|██▎       | 93651/400000 [00:12<00:41, 7464.69it/s] 24%|██▎       | 94398/400000 [00:12<00:40, 7456.28it/s] 24%|██▍       | 95153/400000 [00:12<00:40, 7482.43it/s] 24%|██▍       | 95902/400000 [00:12<00:41, 7408.35it/s] 24%|██▍       | 96659/400000 [00:12<00:40, 7454.55it/s] 24%|██▍       | 97405/400000 [00:13<00:40, 7427.68it/s] 25%|██▍       | 98158/400000 [00:13<00:40, 7457.83it/s] 25%|██▍       | 98910/400000 [00:13<00:40, 7474.70it/s] 25%|██▍       | 99658/400000 [00:13<00:40, 7462.59it/s] 25%|██▌       | 100417/400000 [00:13<00:39, 7497.44it/s] 25%|██▌       | 101175/400000 [00:13<00:39, 7522.01it/s] 25%|██▌       | 101928/400000 [00:13<00:39, 7488.61it/s] 26%|██▌       | 102697/400000 [00:13<00:39, 7547.63it/s] 26%|██▌       | 103452/400000 [00:13<00:39, 7479.33it/s] 26%|██▌       | 104218/400000 [00:13<00:39, 7530.90it/s] 26%|██▌       | 104980/400000 [00:14<00:39, 7556.54it/s] 26%|██▋       | 105736/400000 [00:14<00:38, 7557.20it/s] 27%|██▋       | 106496/400000 [00:14<00:38, 7568.88it/s] 27%|██▋       | 107253/400000 [00:14<00:40, 7219.48it/s] 27%|██▋       | 108005/400000 [00:14<00:39, 7305.14it/s] 27%|██▋       | 108760/400000 [00:14<00:39, 7376.82it/s] 27%|██▋       | 109515/400000 [00:14<00:39, 7425.29it/s] 28%|██▊       | 110261/400000 [00:14<00:38, 7435.64it/s] 28%|██▊       | 111006/400000 [00:14<00:39, 7383.68it/s] 28%|██▊       | 111767/400000 [00:14<00:38, 7448.43it/s] 28%|██▊       | 112521/400000 [00:15<00:38, 7474.26it/s] 28%|██▊       | 113269/400000 [00:15<00:38, 7472.44it/s] 29%|██▊       | 114017/400000 [00:15<00:39, 7272.13it/s] 29%|██▊       | 114748/400000 [00:15<00:39, 7281.85it/s] 29%|██▉       | 115478/400000 [00:15<00:39, 7250.70it/s] 29%|██▉       | 116230/400000 [00:15<00:38, 7326.85it/s] 29%|██▉       | 116986/400000 [00:15<00:38, 7393.04it/s] 29%|██▉       | 117728/400000 [00:15<00:38, 7399.21it/s] 30%|██▉       | 118473/400000 [00:15<00:37, 7413.11it/s] 30%|██▉       | 119223/400000 [00:15<00:37, 7438.02it/s] 30%|██▉       | 119968/400000 [00:16<00:37, 7439.97it/s] 30%|███       | 120727/400000 [00:16<00:37, 7482.68it/s] 30%|███       | 121476/400000 [00:16<00:37, 7483.56it/s] 31%|███       | 122225/400000 [00:16<00:37, 7455.62it/s] 31%|███       | 122980/400000 [00:16<00:37, 7481.82it/s] 31%|███       | 123729/400000 [00:16<00:37, 7426.88it/s] 31%|███       | 124493/400000 [00:16<00:36, 7488.28it/s] 31%|███▏      | 125256/400000 [00:16<00:36, 7527.57it/s] 32%|███▏      | 126009/400000 [00:16<00:36, 7487.30it/s] 32%|███▏      | 126770/400000 [00:16<00:36, 7523.61it/s] 32%|███▏      | 127523/400000 [00:17<00:36, 7495.80it/s] 32%|███▏      | 128279/400000 [00:17<00:36, 7514.22it/s] 32%|███▏      | 129031/400000 [00:17<00:36, 7482.82it/s] 32%|███▏      | 129782/400000 [00:17<00:36, 7490.40it/s] 33%|███▎      | 130542/400000 [00:17<00:35, 7522.93it/s] 33%|███▎      | 131306/400000 [00:17<00:35, 7556.30it/s] 33%|███▎      | 132070/400000 [00:17<00:35, 7579.20it/s] 33%|███▎      | 132829/400000 [00:17<00:35, 7558.82it/s] 33%|███▎      | 133585/400000 [00:17<00:35, 7496.63it/s] 34%|███▎      | 134335/400000 [00:17<00:35, 7404.60it/s] 34%|███▍      | 135080/400000 [00:18<00:35, 7415.78it/s] 34%|███▍      | 135831/400000 [00:18<00:35, 7441.53it/s] 34%|███▍      | 136579/400000 [00:18<00:35, 7451.27it/s] 34%|███▍      | 137325/400000 [00:18<00:35, 7452.94it/s] 35%|███▍      | 138098/400000 [00:18<00:34, 7533.62it/s] 35%|███▍      | 138852/400000 [00:18<00:34, 7535.52it/s] 35%|███▍      | 139610/400000 [00:18<00:34, 7546.59it/s] 35%|███▌      | 140374/400000 [00:18<00:34, 7573.90it/s] 35%|███▌      | 141140/400000 [00:18<00:34, 7593.90it/s] 35%|███▌      | 141900/400000 [00:18<00:33, 7593.78it/s] 36%|███▌      | 142660/400000 [00:19<00:33, 7584.73it/s] 36%|███▌      | 143424/400000 [00:19<00:33, 7592.88it/s] 36%|███▌      | 144187/400000 [00:19<00:33, 7602.16it/s] 36%|███▌      | 144948/400000 [00:19<00:33, 7524.80it/s] 36%|███▋      | 145708/400000 [00:19<00:33, 7544.88it/s] 37%|███▋      | 146463/400000 [00:19<00:33, 7526.99it/s] 37%|███▋      | 147223/400000 [00:19<00:33, 7547.82it/s] 37%|███▋      | 147978/400000 [00:19<00:33, 7511.28it/s] 37%|███▋      | 148730/400000 [00:19<00:33, 7471.72it/s] 37%|███▋      | 149478/400000 [00:19<00:34, 7359.51it/s] 38%|███▊      | 150215/400000 [00:20<00:34, 7309.75it/s] 38%|███▊      | 150972/400000 [00:20<00:33, 7383.92it/s] 38%|███▊      | 151734/400000 [00:20<00:33, 7451.79it/s] 38%|███▊      | 152480/400000 [00:20<00:33, 7437.88it/s] 38%|███▊      | 153225/400000 [00:20<00:33, 7390.47it/s] 38%|███▊      | 153985/400000 [00:20<00:33, 7451.64it/s] 39%|███▊      | 154762/400000 [00:20<00:32, 7544.25it/s] 39%|███▉      | 155524/400000 [00:20<00:32, 7564.24it/s] 39%|███▉      | 156281/400000 [00:20<00:32, 7550.08it/s] 39%|███▉      | 157050/400000 [00:20<00:32, 7589.66it/s] 39%|███▉      | 157811/400000 [00:21<00:31, 7595.40it/s] 40%|███▉      | 158586/400000 [00:21<00:31, 7638.99it/s] 40%|███▉      | 159351/400000 [00:21<00:31, 7608.64it/s] 40%|████      | 160113/400000 [00:21<00:31, 7555.74it/s] 40%|████      | 160869/400000 [00:21<00:31, 7553.76it/s] 40%|████      | 161626/400000 [00:21<00:31, 7557.72it/s] 41%|████      | 162382/400000 [00:21<00:31, 7547.59it/s] 41%|████      | 163144/400000 [00:21<00:31, 7567.99it/s] 41%|████      | 163901/400000 [00:21<00:31, 7500.51it/s] 41%|████      | 164652/400000 [00:21<00:31, 7483.33it/s] 41%|████▏     | 165401/400000 [00:22<00:31, 7449.81it/s] 42%|████▏     | 166153/400000 [00:22<00:31, 7468.38it/s] 42%|████▏     | 166900/400000 [00:22<00:31, 7418.11it/s] 42%|████▏     | 167642/400000 [00:22<00:31, 7300.68it/s] 42%|████▏     | 168392/400000 [00:22<00:31, 7357.70it/s] 42%|████▏     | 169148/400000 [00:22<00:31, 7415.02it/s] 42%|████▏     | 169899/400000 [00:22<00:30, 7442.32it/s] 43%|████▎     | 170647/400000 [00:22<00:30, 7450.72it/s] 43%|████▎     | 171393/400000 [00:22<00:30, 7446.51it/s] 43%|████▎     | 172151/400000 [00:23<00:30, 7484.67it/s] 43%|████▎     | 172900/400000 [00:23<00:30, 7472.15it/s] 43%|████▎     | 173648/400000 [00:23<00:30, 7465.33it/s] 44%|████▎     | 174406/400000 [00:23<00:30, 7497.18it/s] 44%|████▍     | 175170/400000 [00:23<00:29, 7538.88it/s] 44%|████▍     | 175939/400000 [00:23<00:29, 7577.28it/s] 44%|████▍     | 176697/400000 [00:23<00:29, 7564.91it/s] 44%|████▍     | 177467/400000 [00:23<00:29, 7604.66it/s] 45%|████▍     | 178228/400000 [00:23<00:29, 7603.50it/s] 45%|████▍     | 178989/400000 [00:23<00:29, 7568.38it/s] 45%|████▍     | 179746/400000 [00:24<00:29, 7532.65it/s] 45%|████▌     | 180505/400000 [00:24<00:29, 7548.65it/s] 45%|████▌     | 181273/400000 [00:24<00:28, 7587.38it/s] 46%|████▌     | 182039/400000 [00:24<00:28, 7607.35it/s] 46%|████▌     | 182800/400000 [00:24<00:28, 7543.64it/s] 46%|████▌     | 183555/400000 [00:24<00:28, 7506.76it/s] 46%|████▌     | 184323/400000 [00:24<00:28, 7555.81it/s] 46%|████▋     | 185086/400000 [00:24<00:28, 7577.27it/s] 46%|████▋     | 185844/400000 [00:24<00:28, 7436.86it/s] 47%|████▋     | 186589/400000 [00:24<00:29, 7300.20it/s] 47%|████▋     | 187321/400000 [00:25<00:30, 7068.27it/s] 47%|████▋     | 188049/400000 [00:25<00:29, 7130.33it/s] 47%|████▋     | 188764/400000 [00:25<00:29, 7132.06it/s] 47%|████▋     | 189479/400000 [00:25<00:30, 6962.96it/s] 48%|████▊     | 190240/400000 [00:25<00:29, 7144.81it/s] 48%|████▊     | 190957/400000 [00:25<00:29, 7100.93it/s] 48%|████▊     | 191720/400000 [00:25<00:28, 7249.71it/s] 48%|████▊     | 192471/400000 [00:25<00:28, 7323.97it/s] 48%|████▊     | 193219/400000 [00:25<00:28, 7368.16it/s] 48%|████▊     | 193960/400000 [00:25<00:27, 7377.79it/s] 49%|████▊     | 194699/400000 [00:26<00:27, 7363.72it/s] 49%|████▉     | 195446/400000 [00:26<00:27, 7392.92it/s] 49%|████▉     | 196187/400000 [00:26<00:27, 7397.16it/s] 49%|████▉     | 196930/400000 [00:26<00:27, 7405.03it/s] 49%|████▉     | 197682/400000 [00:26<00:27, 7438.02it/s] 50%|████▉     | 198426/400000 [00:26<00:27, 7398.31it/s] 50%|████▉     | 199176/400000 [00:26<00:27, 7426.96it/s] 50%|████▉     | 199919/400000 [00:26<00:27, 7295.47it/s] 50%|█████     | 200682/400000 [00:26<00:26, 7390.10it/s] 50%|█████     | 201452/400000 [00:26<00:26, 7478.71it/s] 51%|█████     | 202204/400000 [00:27<00:26, 7490.79it/s] 51%|█████     | 202970/400000 [00:27<00:26, 7539.35it/s] 51%|█████     | 203742/400000 [00:27<00:25, 7591.22it/s] 51%|█████     | 204502/400000 [00:27<00:25, 7589.73it/s] 51%|█████▏    | 205267/400000 [00:27<00:25, 7604.95it/s] 52%|█████▏    | 206028/400000 [00:27<00:25, 7591.29it/s] 52%|█████▏    | 206789/400000 [00:27<00:25, 7596.04it/s] 52%|█████▏    | 207549/400000 [00:27<00:25, 7576.70it/s] 52%|█████▏    | 208307/400000 [00:27<00:25, 7534.48it/s] 52%|█████▏    | 209061/400000 [00:27<00:25, 7509.85it/s] 52%|█████▏    | 209813/400000 [00:28<00:25, 7511.90it/s] 53%|█████▎    | 210565/400000 [00:28<00:25, 7511.25it/s] 53%|█████▎    | 211317/400000 [00:28<00:25, 7483.45it/s] 53%|█████▎    | 212066/400000 [00:28<00:25, 7422.74it/s] 53%|█████▎    | 212809/400000 [00:28<00:25, 7413.10it/s] 53%|█████▎    | 213560/400000 [00:28<00:25, 7439.87it/s] 54%|█████▎    | 214305/400000 [00:28<00:25, 7397.98it/s] 54%|█████▍    | 215054/400000 [00:28<00:24, 7424.00it/s] 54%|█████▍    | 215824/400000 [00:28<00:24, 7501.99it/s] 54%|█████▍    | 216575/400000 [00:28<00:24, 7502.50it/s] 54%|█████▍    | 217326/400000 [00:29<00:24, 7469.66it/s] 55%|█████▍    | 218099/400000 [00:29<00:24, 7544.81it/s] 55%|█████▍    | 218856/400000 [00:29<00:23, 7550.27it/s] 55%|█████▍    | 219616/400000 [00:29<00:23, 7564.50it/s] 55%|█████▌    | 220393/400000 [00:29<00:23, 7623.50it/s] 55%|█████▌    | 221156/400000 [00:29<00:23, 7623.98it/s] 55%|█████▌    | 221927/400000 [00:29<00:23, 7647.45it/s] 56%|█████▌    | 222698/400000 [00:29<00:23, 7665.31it/s] 56%|█████▌    | 223465/400000 [00:29<00:23, 7663.08it/s] 56%|█████▌    | 224232/400000 [00:29<00:22, 7648.92it/s] 56%|█████▌    | 224997/400000 [00:30<00:23, 7544.59it/s] 56%|█████▋    | 225752/400000 [00:30<00:23, 7516.13it/s] 57%|█████▋    | 226504/400000 [00:30<00:23, 7498.26it/s] 57%|█████▋    | 227255/400000 [00:30<00:23, 7474.95it/s] 57%|█████▋    | 228003/400000 [00:30<00:23, 7439.78it/s] 57%|█████▋    | 228748/400000 [00:30<00:23, 7346.38it/s] 57%|█████▋    | 229499/400000 [00:30<00:23, 7393.69it/s] 58%|█████▊    | 230239/400000 [00:30<00:23, 7289.74it/s] 58%|█████▊    | 231005/400000 [00:30<00:22, 7394.52it/s] 58%|█████▊    | 231771/400000 [00:30<00:22, 7469.60it/s] 58%|█████▊    | 232532/400000 [00:31<00:22, 7510.17it/s] 58%|█████▊    | 233284/400000 [00:31<00:22, 7513.06it/s] 59%|█████▊    | 234055/400000 [00:31<00:21, 7570.24it/s] 59%|█████▊    | 234817/400000 [00:31<00:21, 7584.78it/s] 59%|█████▉    | 235588/400000 [00:31<00:21, 7621.52it/s] 59%|█████▉    | 236353/400000 [00:31<00:21, 7627.91it/s] 59%|█████▉    | 237133/400000 [00:31<00:21, 7675.62it/s] 59%|█████▉    | 237901/400000 [00:31<00:21, 7646.52it/s] 60%|█████▉    | 238666/400000 [00:31<00:21, 7641.68it/s] 60%|█████▉    | 239431/400000 [00:31<00:21, 7625.25it/s] 60%|██████    | 240194/400000 [00:32<00:21, 7575.58it/s] 60%|██████    | 240960/400000 [00:32<00:20, 7599.73it/s] 60%|██████    | 241723/400000 [00:32<00:20, 7608.77it/s] 61%|██████    | 242484/400000 [00:32<00:20, 7595.16it/s] 61%|██████    | 243251/400000 [00:32<00:20, 7617.03it/s] 61%|██████    | 244013/400000 [00:32<00:20, 7530.71it/s] 61%|██████    | 244770/400000 [00:32<00:20, 7539.28it/s] 61%|██████▏   | 245525/400000 [00:32<00:20, 7511.81it/s] 62%|██████▏   | 246280/400000 [00:32<00:20, 7522.60it/s] 62%|██████▏   | 247040/400000 [00:33<00:20, 7543.61it/s] 62%|██████▏   | 247795/400000 [00:33<00:20, 7485.61it/s] 62%|██████▏   | 248544/400000 [00:33<00:20, 7365.66it/s] 62%|██████▏   | 249306/400000 [00:33<00:20, 7439.13it/s] 63%|██████▎   | 250051/400000 [00:33<00:20, 7402.32it/s] 63%|██████▎   | 250792/400000 [00:33<00:20, 7362.40it/s] 63%|██████▎   | 251536/400000 [00:33<00:20, 7383.66it/s] 63%|██████▎   | 252302/400000 [00:33<00:19, 7463.78it/s] 63%|██████▎   | 253072/400000 [00:33<00:19, 7530.50it/s] 63%|██████▎   | 253826/400000 [00:33<00:19, 7513.70it/s] 64%|██████▎   | 254578/400000 [00:34<00:19, 7459.88it/s] 64%|██████▍   | 255325/400000 [00:34<00:19, 7254.19it/s] 64%|██████▍   | 256052/400000 [00:34<00:19, 7214.82it/s] 64%|██████▍   | 256819/400000 [00:34<00:19, 7345.02it/s] 64%|██████▍   | 257582/400000 [00:34<00:19, 7426.54it/s] 65%|██████▍   | 258353/400000 [00:34<00:18, 7506.61it/s] 65%|██████▍   | 259105/400000 [00:34<00:18, 7474.32it/s] 65%|██████▍   | 259871/400000 [00:34<00:18, 7526.85it/s] 65%|██████▌   | 260625/400000 [00:34<00:18, 7516.61it/s] 65%|██████▌   | 261395/400000 [00:34<00:18, 7570.32it/s] 66%|██████▌   | 262163/400000 [00:35<00:18, 7602.70it/s] 66%|██████▌   | 262924/400000 [00:35<00:18, 7514.35it/s] 66%|██████▌   | 263682/400000 [00:35<00:18, 7531.30it/s] 66%|██████▌   | 264436/400000 [00:35<00:18, 7524.24it/s] 66%|██████▋   | 265189/400000 [00:35<00:18, 7442.59it/s] 66%|██████▋   | 265956/400000 [00:35<00:17, 7507.79it/s] 67%|██████▋   | 266710/400000 [00:35<00:17, 7515.71it/s] 67%|██████▋   | 267462/400000 [00:35<00:17, 7405.31it/s] 67%|██████▋   | 268229/400000 [00:35<00:17, 7481.20it/s] 67%|██████▋   | 268978/400000 [00:35<00:17, 7331.89it/s] 67%|██████▋   | 269713/400000 [00:36<00:17, 7291.13it/s] 68%|██████▊   | 270475/400000 [00:36<00:17, 7385.91it/s] 68%|██████▊   | 271232/400000 [00:36<00:17, 7438.68it/s] 68%|██████▊   | 272003/400000 [00:36<00:17, 7516.15it/s] 68%|██████▊   | 272761/400000 [00:36<00:16, 7533.75it/s] 68%|██████▊   | 273521/400000 [00:36<00:16, 7552.31it/s] 69%|██████▊   | 274277/400000 [00:36<00:16, 7491.18it/s] 69%|██████▉   | 275030/400000 [00:36<00:16, 7500.28it/s] 69%|██████▉   | 275781/400000 [00:36<00:16, 7493.85it/s] 69%|██████▉   | 276549/400000 [00:36<00:16, 7547.37it/s] 69%|██████▉   | 277318/400000 [00:37<00:16, 7587.88it/s] 70%|██████▉   | 278077/400000 [00:37<00:16, 7423.32it/s] 70%|██████▉   | 278831/400000 [00:37<00:16, 7456.25it/s] 70%|██████▉   | 279583/400000 [00:37<00:16, 7474.67it/s] 70%|███████   | 280340/400000 [00:37<00:15, 7502.76it/s] 70%|███████   | 281098/400000 [00:37<00:15, 7524.45it/s] 70%|███████   | 281854/400000 [00:37<00:15, 7534.89it/s] 71%|███████   | 282608/400000 [00:37<00:15, 7502.34it/s] 71%|███████   | 283368/400000 [00:37<00:15, 7529.13it/s] 71%|███████   | 284129/400000 [00:37<00:15, 7551.73it/s] 71%|███████   | 284891/400000 [00:38<00:15, 7570.31it/s] 71%|███████▏  | 285649/400000 [00:38<00:15, 7562.67it/s] 72%|███████▏  | 286406/400000 [00:38<00:15, 7491.72it/s] 72%|███████▏  | 287156/400000 [00:38<00:15, 7373.40it/s] 72%|███████▏  | 287905/400000 [00:38<00:15, 7406.84it/s] 72%|███████▏  | 288652/400000 [00:38<00:14, 7425.20it/s] 72%|███████▏  | 289415/400000 [00:38<00:14, 7483.45it/s] 73%|███████▎  | 290164/400000 [00:38<00:14, 7383.84it/s] 73%|███████▎  | 290930/400000 [00:38<00:14, 7461.96it/s] 73%|███████▎  | 291693/400000 [00:38<00:14, 7509.70it/s] 73%|███████▎  | 292447/400000 [00:39<00:14, 7516.44it/s] 73%|███████▎  | 293199/400000 [00:39<00:14, 7510.02it/s] 73%|███████▎  | 293951/400000 [00:39<00:14, 7468.14it/s] 74%|███████▎  | 294725/400000 [00:39<00:13, 7546.11it/s] 74%|███████▍  | 295480/400000 [00:39<00:13, 7544.51it/s] 74%|███████▍  | 296235/400000 [00:39<00:13, 7491.60it/s] 74%|███████▍  | 296993/400000 [00:39<00:13, 7515.35it/s] 74%|███████▍  | 297745/400000 [00:39<00:13, 7514.69it/s] 75%|███████▍  | 298497/400000 [00:39<00:13, 7496.01it/s] 75%|███████▍  | 299251/400000 [00:39<00:13, 7508.64it/s] 75%|███████▌  | 300002/400000 [00:40<00:13, 7409.98it/s] 75%|███████▌  | 300744/400000 [00:40<00:13, 7127.63it/s] 75%|███████▌  | 301481/400000 [00:40<00:13, 7197.24it/s] 76%|███████▌  | 302223/400000 [00:40<00:13, 7262.52it/s] 76%|███████▌  | 302979/400000 [00:40<00:13, 7346.62it/s] 76%|███████▌  | 303740/400000 [00:40<00:12, 7422.36it/s] 76%|███████▌  | 304492/400000 [00:40<00:12, 7449.82it/s] 76%|███████▋  | 305238/400000 [00:40<00:12, 7399.62it/s] 76%|███████▋  | 305996/400000 [00:40<00:12, 7451.84it/s] 77%|███████▋  | 306742/400000 [00:41<00:12, 7396.44it/s] 77%|███████▋  | 307499/400000 [00:41<00:12, 7445.22it/s] 77%|███████▋  | 308244/400000 [00:41<00:12, 7422.03it/s] 77%|███████▋  | 308987/400000 [00:41<00:12, 7348.52it/s] 77%|███████▋  | 309730/400000 [00:41<00:12, 7370.50it/s] 78%|███████▊  | 310474/400000 [00:41<00:12, 7390.98it/s] 78%|███████▊  | 311225/400000 [00:41<00:11, 7425.18it/s] 78%|███████▊  | 311981/400000 [00:41<00:11, 7463.60it/s] 78%|███████▊  | 312728/400000 [00:41<00:11, 7423.29it/s] 78%|███████▊  | 313471/400000 [00:41<00:11, 7409.96it/s] 79%|███████▊  | 314226/400000 [00:42<00:11, 7450.71it/s] 79%|███████▊  | 314979/400000 [00:42<00:11, 7471.72it/s] 79%|███████▉  | 315727/400000 [00:42<00:11, 7465.14it/s] 79%|███████▉  | 316474/400000 [00:42<00:11, 7439.07it/s] 79%|███████▉  | 317235/400000 [00:42<00:11, 7481.86it/s] 79%|███████▉  | 317999/400000 [00:42<00:10, 7527.05it/s] 80%|███████▉  | 318765/400000 [00:42<00:10, 7563.79it/s] 80%|███████▉  | 319524/400000 [00:42<00:10, 7570.10it/s] 80%|████████  | 320282/400000 [00:42<00:10, 7513.33it/s] 80%|████████  | 321034/400000 [00:42<00:10, 7460.04it/s] 80%|████████  | 321797/400000 [00:43<00:10, 7509.62it/s] 81%|████████  | 322552/400000 [00:43<00:10, 7519.71it/s] 81%|████████  | 323305/400000 [00:43<00:10, 7508.30it/s] 81%|████████  | 324056/400000 [00:43<00:10, 7390.86it/s] 81%|████████  | 324796/400000 [00:43<00:10, 7295.27it/s] 81%|████████▏ | 325568/400000 [00:43<00:10, 7416.42it/s] 82%|████████▏ | 326341/400000 [00:43<00:09, 7506.52it/s] 82%|████████▏ | 327096/400000 [00:43<00:09, 7517.09it/s] 82%|████████▏ | 327849/400000 [00:43<00:09, 7396.67it/s] 82%|████████▏ | 328590/400000 [00:43<00:09, 7305.35it/s] 82%|████████▏ | 329348/400000 [00:44<00:09, 7384.05it/s] 83%|████████▎ | 330105/400000 [00:44<00:09, 7437.83it/s] 83%|████████▎ | 330871/400000 [00:44<00:09, 7500.61it/s] 83%|████████▎ | 331622/400000 [00:44<00:09, 7495.47it/s] 83%|████████▎ | 332381/400000 [00:44<00:08, 7521.75it/s] 83%|████████▎ | 333135/400000 [00:44<00:08, 7525.80it/s] 83%|████████▎ | 333888/400000 [00:44<00:08, 7472.56it/s] 84%|████████▎ | 334648/400000 [00:44<00:08, 7507.71it/s] 84%|████████▍ | 335399/400000 [00:44<00:08, 7481.25it/s] 84%|████████▍ | 336150/400000 [00:44<00:08, 7488.25it/s] 84%|████████▍ | 336904/400000 [00:45<00:08, 7503.60it/s] 84%|████████▍ | 337655/400000 [00:45<00:08, 7435.25it/s] 85%|████████▍ | 338399/400000 [00:45<00:08, 7375.22it/s] 85%|████████▍ | 339137/400000 [00:45<00:08, 7215.06it/s] 85%|████████▍ | 339860/400000 [00:45<00:08, 7141.42it/s] 85%|████████▌ | 340622/400000 [00:45<00:08, 7278.22it/s] 85%|████████▌ | 341389/400000 [00:45<00:07, 7390.92it/s] 86%|████████▌ | 342154/400000 [00:45<00:07, 7465.41it/s] 86%|████████▌ | 342923/400000 [00:45<00:07, 7528.76it/s] 86%|████████▌ | 343691/400000 [00:45<00:07, 7572.28it/s] 86%|████████▌ | 344449/400000 [00:46<00:07, 7515.60it/s] 86%|████████▋ | 345202/400000 [00:46<00:07, 7358.70it/s] 86%|████████▋ | 345939/400000 [00:46<00:07, 7167.58it/s] 87%|████████▋ | 346658/400000 [00:46<00:07, 7059.86it/s] 87%|████████▋ | 347423/400000 [00:46<00:07, 7225.88it/s] 87%|████████▋ | 348189/400000 [00:46<00:07, 7350.77it/s] 87%|████████▋ | 348949/400000 [00:46<00:06, 7421.23it/s] 87%|████████▋ | 349704/400000 [00:46<00:06, 7458.92it/s] 88%|████████▊ | 350463/400000 [00:46<00:06, 7494.97it/s] 88%|████████▊ | 351215/400000 [00:46<00:06, 7500.96it/s] 88%|████████▊ | 351976/400000 [00:47<00:06, 7531.82it/s] 88%|████████▊ | 352737/400000 [00:47<00:06, 7553.80it/s] 88%|████████▊ | 353500/400000 [00:47<00:06, 7574.38it/s] 89%|████████▊ | 354258/400000 [00:47<00:06, 7527.24it/s] 89%|████████▉ | 355011/400000 [00:47<00:05, 7510.42it/s] 89%|████████▉ | 355776/400000 [00:47<00:05, 7550.96it/s] 89%|████████▉ | 356533/400000 [00:47<00:05, 7555.75it/s] 89%|████████▉ | 357294/400000 [00:47<00:05, 7570.73it/s] 90%|████████▉ | 358052/400000 [00:47<00:05, 7494.84it/s] 90%|████████▉ | 358802/400000 [00:48<00:05, 7454.31it/s] 90%|████████▉ | 359548/400000 [00:48<00:05, 7454.91it/s] 90%|█████████ | 360294/400000 [00:48<00:05, 7416.92it/s] 90%|█████████ | 361041/400000 [00:48<00:05, 7431.42it/s] 90%|█████████ | 361785/400000 [00:48<00:05, 7161.65it/s] 91%|█████████ | 362504/400000 [00:48<00:05, 7072.01it/s] 91%|█████████ | 363213/400000 [00:48<00:05, 6968.34it/s] 91%|█████████ | 363953/400000 [00:48<00:05, 7091.83it/s] 91%|█████████ | 364710/400000 [00:48<00:04, 7226.46it/s] 91%|█████████▏| 365435/400000 [00:48<00:04, 7131.48it/s] 92%|█████████▏| 366204/400000 [00:49<00:04, 7288.28it/s] 92%|█████████▏| 366935/400000 [00:49<00:04, 7005.08it/s] 92%|█████████▏| 367685/400000 [00:49<00:04, 7145.04it/s] 92%|█████████▏| 368430/400000 [00:49<00:04, 7232.51it/s] 92%|█████████▏| 369156/400000 [00:49<00:04, 7238.60it/s] 92%|█████████▏| 369897/400000 [00:49<00:04, 7288.23it/s] 93%|█████████▎| 370653/400000 [00:49<00:03, 7366.07it/s] 93%|█████████▎| 371398/400000 [00:49<00:03, 7389.71it/s] 93%|█████████▎| 372143/400000 [00:49<00:03, 7402.97it/s] 93%|█████████▎| 372886/400000 [00:49<00:03, 7409.56it/s] 93%|█████████▎| 373628/400000 [00:50<00:03, 7388.34it/s] 94%|█████████▎| 374368/400000 [00:50<00:03, 7376.53it/s] 94%|█████████▍| 375119/400000 [00:50<00:03, 7413.34it/s] 94%|█████████▍| 375871/400000 [00:50<00:03, 7442.69it/s] 94%|█████████▍| 376616/400000 [00:50<00:03, 7440.56it/s] 94%|█████████▍| 377361/400000 [00:50<00:03, 7367.56it/s] 95%|█████████▍| 378098/400000 [00:50<00:02, 7303.97it/s] 95%|█████████▍| 378829/400000 [00:50<00:02, 7246.98it/s] 95%|█████████▍| 379555/400000 [00:50<00:02, 7196.23it/s] 95%|█████████▌| 380300/400000 [00:50<00:02, 7269.93it/s] 95%|█████████▌| 381028/400000 [00:51<00:02, 7205.73it/s] 95%|█████████▌| 381764/400000 [00:51<00:02, 7250.53it/s] 96%|█████████▌| 382513/400000 [00:51<00:02, 7320.44it/s] 96%|█████████▌| 383267/400000 [00:51<00:02, 7383.23it/s] 96%|█████████▌| 384027/400000 [00:51<00:02, 7444.12it/s] 96%|█████████▌| 384785/400000 [00:51<00:02, 7481.54it/s] 96%|█████████▋| 385550/400000 [00:51<00:01, 7530.24it/s] 97%|█████████▋| 386304/400000 [00:51<00:01, 7512.29it/s] 97%|█████████▋| 387057/400000 [00:51<00:01, 7515.80it/s] 97%|█████████▋| 387815/400000 [00:51<00:01, 7532.95it/s] 97%|█████████▋| 388576/400000 [00:52<00:01, 7552.01it/s] 97%|█████████▋| 389332/400000 [00:52<00:01, 7512.68it/s] 98%|█████████▊| 390084/400000 [00:52<00:01, 7495.16it/s] 98%|█████████▊| 390856/400000 [00:52<00:01, 7558.45it/s] 98%|█████████▊| 391613/400000 [00:52<00:01, 7516.84it/s] 98%|█████████▊| 392365/400000 [00:52<00:01, 7413.16it/s] 98%|█████████▊| 393118/400000 [00:52<00:00, 7446.95it/s] 98%|█████████▊| 393864/400000 [00:52<00:00, 7440.36it/s] 99%|█████████▊| 394609/400000 [00:52<00:00, 7413.63it/s] 99%|█████████▉| 395351/400000 [00:52<00:00, 7412.96it/s] 99%|█████████▉| 396093/400000 [00:53<00:00, 7408.73it/s] 99%|█████████▉| 396848/400000 [00:53<00:00, 7448.48it/s] 99%|█████████▉| 397599/400000 [00:53<00:00, 7464.49it/s]100%|█████████▉| 398348/400000 [00:53<00:00, 7471.94it/s]100%|█████████▉| 399107/400000 [00:53<00:00, 7506.75it/s]100%|█████████▉| 399858/400000 [00:53<00:00, 7497.82it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7462.52it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2949ecdd30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01109018491232003 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010846146572393717 	 Accuracy: 61

  model saves at 61% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15982 out of table with 15654 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


### Running {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} 'model_pars' 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 

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
RuntimeError: index out of range: Tried to access index 15982 out of table with 15654 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
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
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 14:25:06.674623: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 14:25:06.678718: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 14:25:06.679602: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563eacdc21c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 14:25:06.679627: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2958141160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 8.1113 - accuracy: 0.4710
 2000/25000 [=>............................] - ETA: 11s - loss: 7.9273 - accuracy: 0.4830
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8455 - accuracy: 0.4883 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7548 - accuracy: 0.4942
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7617 - accuracy: 0.4938
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7345 - accuracy: 0.4956
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6935 - accuracy: 0.4983
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6559 - accuracy: 0.5007
11000/25000 [============>.................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
12000/25000 [=============>................] - ETA: 4s - loss: 7.6615 - accuracy: 0.5003
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 3s - loss: 7.6850 - accuracy: 0.4988
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6810 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6888 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6803 - accuracy: 0.4991
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6849 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 10s 412us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f28ba86cda0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'} 

  #### Setup Model   ############################################## 
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 75)                0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 75, 40)            1720      
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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f2949ed9a20> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.0936 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.0924 - val_crf_viterbi_accuracy: 0.6533

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
    output_path      = out_pars["model_path"]
KeyError: 'model_path'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
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
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

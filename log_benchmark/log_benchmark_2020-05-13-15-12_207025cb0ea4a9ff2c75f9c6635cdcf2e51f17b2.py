
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7efdc1dbbfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 15:12:16.829481
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 15:12:16.833086
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 15:12:16.836083
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 15:12:16.839089
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7efdcdb85470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353505.7812
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 281094.2812
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 202842.3750
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 134283.9531
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 87359.5703
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 58826.4883
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 41299.8633
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 30117.3711
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 22746.6230
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 17735.8711

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.0708521  -0.20249234 -1.3761983  -1.3263497   0.60096467 -0.20641637
  -1.4042972   1.0792503   1.321656    0.1473416   0.03491394  1.2010121
   1.1217695   0.97615933 -0.22166821 -0.7248971  -0.4709776   0.91283125
   0.83341306 -1.4353989   0.8483594  -0.30621892  1.013534   -0.9868621
   0.37990764 -1.1635414  -1.0341812   0.78166467  0.9796601  -0.30888498
   1.2193847   0.80050904  0.43105847 -1.4850794   0.37021384  0.7659383
  -0.19076048  0.14681286  0.10498545 -1.2762586  -0.904192    1.4429094
  -0.87326515 -0.24372974 -1.2939141  -1.5259999   0.6743201  -1.5049248
  -0.71909547  1.3557413   0.36748677  0.7096943   0.19406523  0.1276271
   0.44999418  1.0304458  -0.39021137 -0.4025403   1.1610144   1.1035945
   0.14136691  3.745122    4.14733     4.709248    3.8668752   4.121769
   4.8698726   4.003592    3.5857491   3.6541717   3.1716037   4.5951433
   5.1561413   4.376311    3.8570604   3.2111676   3.6661286   5.6584215
   3.4752033   3.2937882   3.0426524   5.40683     4.374139    5.041834
   2.9778666   3.8618646   5.0173864   3.4088588   5.774586    5.3689184
   5.2564583   5.1693645   4.146128    3.553234    4.30698     5.8101234
   3.4397786   4.3879576   2.902873    4.0344157   4.2212577   3.8354778
   5.5662174   3.662868    4.2738833   5.142226    5.838429    5.008242
   3.4831226   5.229835    2.9887724   3.5350986   2.9507213   4.3887653
   5.1613007   5.3644333   4.30594     4.1544085   5.587746    5.7446895
   0.17477635 -0.9337391  -1.449742    0.7583707   0.18078437  0.6851993
  -0.5990761   0.82115763 -0.36736962 -1.4815739  -0.48317537  0.23406328
   1.3091038  -1.4694222   0.14228418  0.09724773 -0.41062453 -1.4351039
   0.56303275  1.4561079  -0.11442113  1.2699697   0.40276635 -0.36087742
   1.3301952  -0.44621104  0.23180807 -1.0376952   1.3442739   0.94047076
   1.1242989  -0.31005138  0.68768007 -0.49900174 -0.93920046 -0.83880293
  -1.2198536  -0.9169042  -1.1923553   0.9312447   0.57263565 -1.2978827
   1.454637    0.94880193  1.2097763  -1.038509    0.92208356 -0.52822506
   0.03010402 -1.2175597   0.8895227   0.29325402 -1.4697767   0.8859935
   0.28572923 -1.4038641  -0.3455046  -0.01729862  0.56303674  0.520939
   1.8061272   0.90461725  0.34422898  0.41158855  1.4926598   2.2729836
   0.4982506   0.65290916  1.3474262   1.159055    0.78000265  0.7957906
   2.3829587   0.2619617   2.0396132   1.629667    1.9665127   0.48913604
   1.0284452   1.5592983   0.48620284  1.9770855   0.50366336  2.4335585
   2.112403    0.29401052  0.32739055  1.3296287   2.1680427   0.95115334
   2.2100348   1.3774705   2.0744147   0.2674843   0.77700746  0.32658744
   1.8039765   0.36826295  1.0785005   0.22107029  0.47585744  0.4192469
   0.5046842   1.4307591   1.6911726   0.69838834  1.2208902   0.3102073
   0.36407602  2.3981903   2.1486053   2.1130304   0.614113    1.4381444
   0.39920402  0.2717651   0.9385242   0.8209348   1.7422779   1.8434931
   0.04541105  5.643446    3.605113    3.7552772   6.2914624   3.8280563
   5.835803    5.076235    4.2855945   4.887433    5.2563405   6.3907547
   3.7608867   5.5353208   4.071825    6.195472    3.8951373   5.7649446
   4.5583057   6.096048    4.5826063   5.307252    5.883609    6.020583
   5.2793875   4.8579774   4.010824    5.7984653   4.7942295   3.996838
   6.4194536   4.0287414   6.067314    4.711206    4.3026953   5.9857564
   5.5538025   6.168813    5.133345    4.7132416   4.3881907   4.146275
   5.999923    6.2103314   4.903516    6.250815    3.8223853   5.731911
   5.817262    4.8634844   5.5009274   4.1450567   4.922377    5.4386168
   4.9145017   6.182385    5.3313327   6.468968    4.2989783   4.9827805
   0.22146678  2.337       1.8831503   0.8716628   1.2555931   1.6786454
   0.91121113  1.5464272   0.6146345   0.31487733  1.1824142   1.1574037
   0.5675992   2.389371    0.5915594   0.24882627  1.4583647   0.4258902
   0.36788446  0.7713924   0.31437767  0.37233007  0.3639121   0.41665363
   1.7876129   0.24719042  1.1530583   0.72115004  0.57172203  2.411852
   0.33092397  1.8746033   0.51750755  0.6640937   1.8955936   0.32975674
   2.4828773   2.4741187   1.4465597   0.27392018  0.24308509  0.5522357
   0.4158702   1.0175533   0.3498726   0.34874094  0.44595194  1.2879571
   0.6990199   1.3470209   0.46353197  0.5226738   2.4538813   1.8763943
   0.35933542  0.36471283  0.28933305  1.2523721   0.8494579   2.2769175
  -2.1108034   1.4041831  -8.802378  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 15:12:26.397470
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.0811
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 15:12:26.400996
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9635.66
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 15:12:26.404137
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.1196
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 15:12:26.407089
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -861.942
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139627985096832
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139627043517104
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139627043517608
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139627043518112
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139627043518616
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139627043519120

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7efdbb54f668> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.479455
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.452843
grad_step = 000002, loss = 0.432339
grad_step = 000003, loss = 0.411281
grad_step = 000004, loss = 0.389368
grad_step = 000005, loss = 0.370483
grad_step = 000006, loss = 0.357301
grad_step = 000007, loss = 0.345325
grad_step = 000008, loss = 0.327016
grad_step = 000009, loss = 0.308003
grad_step = 000010, loss = 0.292059
grad_step = 000011, loss = 0.278867
grad_step = 000012, loss = 0.266988
grad_step = 000013, loss = 0.256359
grad_step = 000014, loss = 0.246468
grad_step = 000015, loss = 0.236053
grad_step = 000016, loss = 0.225696
grad_step = 000017, loss = 0.215836
grad_step = 000018, loss = 0.206026
grad_step = 000019, loss = 0.196163
grad_step = 000020, loss = 0.186569
grad_step = 000021, loss = 0.177664
grad_step = 000022, loss = 0.169390
grad_step = 000023, loss = 0.161051
grad_step = 000024, loss = 0.152906
grad_step = 000025, loss = 0.145240
grad_step = 000026, loss = 0.137863
grad_step = 000027, loss = 0.130584
grad_step = 000028, loss = 0.123509
grad_step = 000029, loss = 0.116987
grad_step = 000030, loss = 0.111028
grad_step = 000031, loss = 0.105207
grad_step = 000032, loss = 0.099405
grad_step = 000033, loss = 0.093808
grad_step = 000034, loss = 0.088495
grad_step = 000035, loss = 0.083394
grad_step = 000036, loss = 0.078531
grad_step = 000037, loss = 0.073984
grad_step = 000038, loss = 0.069658
grad_step = 000039, loss = 0.065426
grad_step = 000040, loss = 0.061337
grad_step = 000041, loss = 0.057484
grad_step = 000042, loss = 0.053817
grad_step = 000043, loss = 0.050324
grad_step = 000044, loss = 0.047048
grad_step = 000045, loss = 0.043932
grad_step = 000046, loss = 0.040888
grad_step = 000047, loss = 0.038025
grad_step = 000048, loss = 0.035379
grad_step = 000049, loss = 0.032853
grad_step = 000050, loss = 0.030491
grad_step = 000051, loss = 0.028293
grad_step = 000052, loss = 0.026179
grad_step = 000053, loss = 0.024215
grad_step = 000054, loss = 0.022402
grad_step = 000055, loss = 0.020690
grad_step = 000056, loss = 0.019128
grad_step = 000057, loss = 0.017662
grad_step = 000058, loss = 0.016279
grad_step = 000059, loss = 0.015017
grad_step = 000060, loss = 0.013838
grad_step = 000061, loss = 0.012766
grad_step = 000062, loss = 0.011775
grad_step = 000063, loss = 0.010845
grad_step = 000064, loss = 0.010005
grad_step = 000065, loss = 0.009219
grad_step = 000066, loss = 0.008513
grad_step = 000067, loss = 0.007863
grad_step = 000068, loss = 0.007276
grad_step = 000069, loss = 0.006748
grad_step = 000070, loss = 0.006263
grad_step = 000071, loss = 0.005833
grad_step = 000072, loss = 0.005436
grad_step = 000073, loss = 0.005084
grad_step = 000074, loss = 0.004761
grad_step = 000075, loss = 0.004477
grad_step = 000076, loss = 0.004221
grad_step = 000077, loss = 0.003991
grad_step = 000078, loss = 0.003785
grad_step = 000079, loss = 0.003597
grad_step = 000080, loss = 0.003431
grad_step = 000081, loss = 0.003279
grad_step = 000082, loss = 0.003144
grad_step = 000083, loss = 0.003019
grad_step = 000084, loss = 0.002909
grad_step = 000085, loss = 0.002807
grad_step = 000086, loss = 0.002719
grad_step = 000087, loss = 0.002638
grad_step = 000088, loss = 0.002569
grad_step = 000089, loss = 0.002510
grad_step = 000090, loss = 0.002466
grad_step = 000091, loss = 0.002429
grad_step = 000092, loss = 0.002394
grad_step = 000093, loss = 0.002343
grad_step = 000094, loss = 0.002281
grad_step = 000095, loss = 0.002216
grad_step = 000096, loss = 0.002168
grad_step = 000097, loss = 0.002141
grad_step = 000098, loss = 0.002131
grad_step = 000099, loss = 0.002131
grad_step = 000100, loss = 0.002131
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002122
grad_step = 000102, loss = 0.002095
grad_step = 000103, loss = 0.002057
grad_step = 000104, loss = 0.002019
grad_step = 000105, loss = 0.001993
grad_step = 000106, loss = 0.001982
grad_step = 000107, loss = 0.001983
grad_step = 000108, loss = 0.001989
grad_step = 000109, loss = 0.001997
grad_step = 000110, loss = 0.002003
grad_step = 000111, loss = 0.002001
grad_step = 000112, loss = 0.001989
grad_step = 000113, loss = 0.001962
grad_step = 000114, loss = 0.001930
grad_step = 000115, loss = 0.001900
grad_step = 000116, loss = 0.001879
grad_step = 000117, loss = 0.001869
grad_step = 000118, loss = 0.001868
grad_step = 000119, loss = 0.001873
grad_step = 000120, loss = 0.001887
grad_step = 000121, loss = 0.001910
grad_step = 000122, loss = 0.001946
grad_step = 000123, loss = 0.001983
grad_step = 000124, loss = 0.001998
grad_step = 000125, loss = 0.001952
grad_step = 000126, loss = 0.001865
grad_step = 000127, loss = 0.001797
grad_step = 000128, loss = 0.001789
grad_step = 000129, loss = 0.001829
grad_step = 000130, loss = 0.001873
grad_step = 000131, loss = 0.001888
grad_step = 000132, loss = 0.001848
grad_step = 000133, loss = 0.001786
grad_step = 000134, loss = 0.001745
grad_step = 000135, loss = 0.001745
grad_step = 000136, loss = 0.001774
grad_step = 000137, loss = 0.001801
grad_step = 000138, loss = 0.001808
grad_step = 000139, loss = 0.001783
grad_step = 000140, loss = 0.001744
grad_step = 000141, loss = 0.001708
grad_step = 000142, loss = 0.001694
grad_step = 000143, loss = 0.001696
grad_step = 000144, loss = 0.001711
grad_step = 000145, loss = 0.001737
grad_step = 000146, loss = 0.001760
grad_step = 000147, loss = 0.001778
grad_step = 000148, loss = 0.001753
grad_step = 000149, loss = 0.001716
grad_step = 000150, loss = 0.001678
grad_step = 000151, loss = 0.001655
grad_step = 000152, loss = 0.001649
grad_step = 000153, loss = 0.001658
grad_step = 000154, loss = 0.001679
grad_step = 000155, loss = 0.001700
grad_step = 000156, loss = 0.001714
grad_step = 000157, loss = 0.001710
grad_step = 000158, loss = 0.001692
grad_step = 000159, loss = 0.001657
grad_step = 000160, loss = 0.001627
grad_step = 000161, loss = 0.001607
grad_step = 000162, loss = 0.001609
grad_step = 000163, loss = 0.001623
grad_step = 000164, loss = 0.001643
grad_step = 000165, loss = 0.001658
grad_step = 000166, loss = 0.001676
grad_step = 000167, loss = 0.001704
grad_step = 000168, loss = 0.001740
grad_step = 000169, loss = 0.001744
grad_step = 000170, loss = 0.001692
grad_step = 000171, loss = 0.001609
grad_step = 000172, loss = 0.001591
grad_step = 000173, loss = 0.001639
grad_step = 000174, loss = 0.001683
grad_step = 000175, loss = 0.001647
grad_step = 000176, loss = 0.001586
grad_step = 000177, loss = 0.001568
grad_step = 000178, loss = 0.001589
grad_step = 000179, loss = 0.001608
grad_step = 000180, loss = 0.001591
grad_step = 000181, loss = 0.001567
grad_step = 000182, loss = 0.001570
grad_step = 000183, loss = 0.001588
grad_step = 000184, loss = 0.001597
grad_step = 000185, loss = 0.001581
grad_step = 000186, loss = 0.001551
grad_step = 000187, loss = 0.001540
grad_step = 000188, loss = 0.001547
grad_step = 000189, loss = 0.001561
grad_step = 000190, loss = 0.001555
grad_step = 000191, loss = 0.001541
grad_step = 000192, loss = 0.001528
grad_step = 000193, loss = 0.001521
grad_step = 000194, loss = 0.001519
grad_step = 000195, loss = 0.001524
grad_step = 000196, loss = 0.001534
grad_step = 000197, loss = 0.001536
grad_step = 000198, loss = 0.001538
grad_step = 000199, loss = 0.001539
grad_step = 000200, loss = 0.001539
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001546
grad_step = 000202, loss = 0.001573
grad_step = 000203, loss = 0.001628
grad_step = 000204, loss = 0.001731
grad_step = 000205, loss = 0.001909
grad_step = 000206, loss = 0.002078
grad_step = 000207, loss = 0.002081
grad_step = 000208, loss = 0.001796
grad_step = 000209, loss = 0.001535
grad_step = 000210, loss = 0.001575
grad_step = 000211, loss = 0.001763
grad_step = 000212, loss = 0.001760
grad_step = 000213, loss = 0.001552
grad_step = 000214, loss = 0.001525
grad_step = 000215, loss = 0.001673
grad_step = 000216, loss = 0.001652
grad_step = 000217, loss = 0.001506
grad_step = 000218, loss = 0.001532
grad_step = 000219, loss = 0.001642
grad_step = 000220, loss = 0.001566
grad_step = 000221, loss = 0.001486
grad_step = 000222, loss = 0.001537
grad_step = 000223, loss = 0.001557
grad_step = 000224, loss = 0.001503
grad_step = 000225, loss = 0.001484
grad_step = 000226, loss = 0.001517
grad_step = 000227, loss = 0.001517
grad_step = 000228, loss = 0.001478
grad_step = 000229, loss = 0.001476
grad_step = 000230, loss = 0.001503
grad_step = 000231, loss = 0.001490
grad_step = 000232, loss = 0.001457
grad_step = 000233, loss = 0.001470
grad_step = 000234, loss = 0.001486
grad_step = 000235, loss = 0.001465
grad_step = 000236, loss = 0.001450
grad_step = 000237, loss = 0.001463
grad_step = 000238, loss = 0.001469
grad_step = 000239, loss = 0.001456
grad_step = 000240, loss = 0.001455
grad_step = 000241, loss = 0.001469
grad_step = 000242, loss = 0.001491
grad_step = 000243, loss = 0.001525
grad_step = 000244, loss = 0.001560
grad_step = 000245, loss = 0.001592
grad_step = 000246, loss = 0.001602
grad_step = 000247, loss = 0.001551
grad_step = 000248, loss = 0.001471
grad_step = 000249, loss = 0.001431
grad_step = 000250, loss = 0.001455
grad_step = 000251, loss = 0.001493
grad_step = 000252, loss = 0.001501
grad_step = 000253, loss = 0.001463
grad_step = 000254, loss = 0.001425
grad_step = 000255, loss = 0.001435
grad_step = 000256, loss = 0.001460
grad_step = 000257, loss = 0.001468
grad_step = 000258, loss = 0.001449
grad_step = 000259, loss = 0.001419
grad_step = 000260, loss = 0.001415
grad_step = 000261, loss = 0.001432
grad_step = 000262, loss = 0.001441
grad_step = 000263, loss = 0.001437
grad_step = 000264, loss = 0.001431
grad_step = 000265, loss = 0.001415
grad_step = 000266, loss = 0.001406
grad_step = 000267, loss = 0.001405
grad_step = 000268, loss = 0.001404
grad_step = 000269, loss = 0.001405
grad_step = 000270, loss = 0.001410
grad_step = 000271, loss = 0.001415
grad_step = 000272, loss = 0.001418
grad_step = 000273, loss = 0.001419
grad_step = 000274, loss = 0.001410
grad_step = 000275, loss = 0.001399
grad_step = 000276, loss = 0.001391
grad_step = 000277, loss = 0.001385
grad_step = 000278, loss = 0.001383
grad_step = 000279, loss = 0.001384
grad_step = 000280, loss = 0.001385
grad_step = 000281, loss = 0.001385
grad_step = 000282, loss = 0.001387
grad_step = 000283, loss = 0.001388
grad_step = 000284, loss = 0.001392
grad_step = 000285, loss = 0.001399
grad_step = 000286, loss = 0.001411
grad_step = 000287, loss = 0.001425
grad_step = 000288, loss = 0.001437
grad_step = 000289, loss = 0.001442
grad_step = 000290, loss = 0.001430
grad_step = 000291, loss = 0.001406
grad_step = 000292, loss = 0.001376
grad_step = 000293, loss = 0.001358
grad_step = 000294, loss = 0.001361
grad_step = 000295, loss = 0.001375
grad_step = 000296, loss = 0.001386
grad_step = 000297, loss = 0.001383
grad_step = 000298, loss = 0.001372
grad_step = 000299, loss = 0.001357
grad_step = 000300, loss = 0.001347
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001345
grad_step = 000302, loss = 0.001349
grad_step = 000303, loss = 0.001358
grad_step = 000304, loss = 0.001375
grad_step = 000305, loss = 0.001395
grad_step = 000306, loss = 0.001423
grad_step = 000307, loss = 0.001445
grad_step = 000308, loss = 0.001464
grad_step = 000309, loss = 0.001465
grad_step = 000310, loss = 0.001448
grad_step = 000311, loss = 0.001425
grad_step = 000312, loss = 0.001421
grad_step = 000313, loss = 0.001449
grad_step = 000314, loss = 0.001492
grad_step = 000315, loss = 0.001489
grad_step = 000316, loss = 0.001451
grad_step = 000317, loss = 0.001390
grad_step = 000318, loss = 0.001357
grad_step = 000319, loss = 0.001353
grad_step = 000320, loss = 0.001352
grad_step = 000321, loss = 0.001341
grad_step = 000322, loss = 0.001329
grad_step = 000323, loss = 0.001332
grad_step = 000324, loss = 0.001351
grad_step = 000325, loss = 0.001376
grad_step = 000326, loss = 0.001390
grad_step = 000327, loss = 0.001388
grad_step = 000328, loss = 0.001366
grad_step = 000329, loss = 0.001337
grad_step = 000330, loss = 0.001309
grad_step = 000331, loss = 0.001295
grad_step = 000332, loss = 0.001298
grad_step = 000333, loss = 0.001311
grad_step = 000334, loss = 0.001324
grad_step = 000335, loss = 0.001326
grad_step = 000336, loss = 0.001318
grad_step = 000337, loss = 0.001304
grad_step = 000338, loss = 0.001293
grad_step = 000339, loss = 0.001289
grad_step = 000340, loss = 0.001292
grad_step = 000341, loss = 0.001299
grad_step = 000342, loss = 0.001308
grad_step = 000343, loss = 0.001321
grad_step = 000344, loss = 0.001328
grad_step = 000345, loss = 0.001324
grad_step = 000346, loss = 0.001317
grad_step = 000347, loss = 0.001305
grad_step = 000348, loss = 0.001294
grad_step = 000349, loss = 0.001286
grad_step = 000350, loss = 0.001289
grad_step = 000351, loss = 0.001303
grad_step = 000352, loss = 0.001328
grad_step = 000353, loss = 0.001361
grad_step = 000354, loss = 0.001399
grad_step = 000355, loss = 0.001441
grad_step = 000356, loss = 0.001476
grad_step = 000357, loss = 0.001494
grad_step = 000358, loss = 0.001472
grad_step = 000359, loss = 0.001435
grad_step = 000360, loss = 0.001424
grad_step = 000361, loss = 0.001441
grad_step = 000362, loss = 0.001483
grad_step = 000363, loss = 0.001453
grad_step = 000364, loss = 0.001357
grad_step = 000365, loss = 0.001265
grad_step = 000366, loss = 0.001276
grad_step = 000367, loss = 0.001351
grad_step = 000368, loss = 0.001367
grad_step = 000369, loss = 0.001299
grad_step = 000370, loss = 0.001240
grad_step = 000371, loss = 0.001265
grad_step = 000372, loss = 0.001314
grad_step = 000373, loss = 0.001303
grad_step = 000374, loss = 0.001254
grad_step = 000375, loss = 0.001232
grad_step = 000376, loss = 0.001257
grad_step = 000377, loss = 0.001290
grad_step = 000378, loss = 0.001296
grad_step = 000379, loss = 0.001262
grad_step = 000380, loss = 0.001226
grad_step = 000381, loss = 0.001223
grad_step = 000382, loss = 0.001242
grad_step = 000383, loss = 0.001261
grad_step = 000384, loss = 0.001251
grad_step = 000385, loss = 0.001227
grad_step = 000386, loss = 0.001215
grad_step = 000387, loss = 0.001222
grad_step = 000388, loss = 0.001234
grad_step = 000389, loss = 0.001236
grad_step = 000390, loss = 0.001225
grad_step = 000391, loss = 0.001211
grad_step = 000392, loss = 0.001204
grad_step = 000393, loss = 0.001206
grad_step = 000394, loss = 0.001212
grad_step = 000395, loss = 0.001217
grad_step = 000396, loss = 0.001219
grad_step = 000397, loss = 0.001217
grad_step = 000398, loss = 0.001211
grad_step = 000399, loss = 0.001208
grad_step = 000400, loss = 0.001207
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001214
grad_step = 000402, loss = 0.001228
grad_step = 000403, loss = 0.001249
grad_step = 000404, loss = 0.001277
grad_step = 000405, loss = 0.001311
grad_step = 000406, loss = 0.001355
grad_step = 000407, loss = 0.001403
grad_step = 000408, loss = 0.001447
grad_step = 000409, loss = 0.001456
grad_step = 000410, loss = 0.001406
grad_step = 000411, loss = 0.001301
grad_step = 000412, loss = 0.001204
grad_step = 000413, loss = 0.001174
grad_step = 000414, loss = 0.001214
grad_step = 000415, loss = 0.001270
grad_step = 000416, loss = 0.001278
grad_step = 000417, loss = 0.001235
grad_step = 000418, loss = 0.001190
grad_step = 000419, loss = 0.001186
grad_step = 000420, loss = 0.001207
grad_step = 000421, loss = 0.001216
grad_step = 000422, loss = 0.001196
grad_step = 000423, loss = 0.001168
grad_step = 000424, loss = 0.001157
grad_step = 000425, loss = 0.001170
grad_step = 000426, loss = 0.001191
grad_step = 000427, loss = 0.001201
grad_step = 000428, loss = 0.001201
grad_step = 000429, loss = 0.001188
grad_step = 000430, loss = 0.001179
grad_step = 000431, loss = 0.001177
grad_step = 000432, loss = 0.001180
grad_step = 000433, loss = 0.001179
grad_step = 000434, loss = 0.001169
grad_step = 000435, loss = 0.001153
grad_step = 000436, loss = 0.001141
grad_step = 000437, loss = 0.001137
grad_step = 000438, loss = 0.001140
grad_step = 000439, loss = 0.001147
grad_step = 000440, loss = 0.001151
grad_step = 000441, loss = 0.001151
grad_step = 000442, loss = 0.001146
grad_step = 000443, loss = 0.001142
grad_step = 000444, loss = 0.001141
grad_step = 000445, loss = 0.001151
grad_step = 000446, loss = 0.001178
grad_step = 000447, loss = 0.001235
grad_step = 000448, loss = 0.001318
grad_step = 000449, loss = 0.001393
grad_step = 000450, loss = 0.001437
grad_step = 000451, loss = 0.001338
grad_step = 000452, loss = 0.001198
grad_step = 000453, loss = 0.001137
grad_step = 000454, loss = 0.001206
grad_step = 000455, loss = 0.001275
grad_step = 000456, loss = 0.001214
grad_step = 000457, loss = 0.001126
grad_step = 000458, loss = 0.001139
grad_step = 000459, loss = 0.001200
grad_step = 000460, loss = 0.001201
grad_step = 000461, loss = 0.001142
grad_step = 000462, loss = 0.001113
grad_step = 000463, loss = 0.001137
grad_step = 000464, loss = 0.001168
grad_step = 000465, loss = 0.001162
grad_step = 000466, loss = 0.001118
grad_step = 000467, loss = 0.001100
grad_step = 000468, loss = 0.001122
grad_step = 000469, loss = 0.001145
grad_step = 000470, loss = 0.001142
grad_step = 000471, loss = 0.001111
grad_step = 000472, loss = 0.001091
grad_step = 000473, loss = 0.001098
grad_step = 000474, loss = 0.001115
grad_step = 000475, loss = 0.001121
grad_step = 000476, loss = 0.001111
grad_step = 000477, loss = 0.001092
grad_step = 000478, loss = 0.001082
grad_step = 000479, loss = 0.001085
grad_step = 000480, loss = 0.001093
grad_step = 000481, loss = 0.001097
grad_step = 000482, loss = 0.001093
grad_step = 000483, loss = 0.001082
grad_step = 000484, loss = 0.001074
grad_step = 000485, loss = 0.001072
grad_step = 000486, loss = 0.001075
grad_step = 000487, loss = 0.001079
grad_step = 000488, loss = 0.001079
grad_step = 000489, loss = 0.001075
grad_step = 000490, loss = 0.001069
grad_step = 000491, loss = 0.001064
grad_step = 000492, loss = 0.001062
grad_step = 000493, loss = 0.001063
grad_step = 000494, loss = 0.001064
grad_step = 000495, loss = 0.001066
grad_step = 000496, loss = 0.001068
grad_step = 000497, loss = 0.001070
grad_step = 000498, loss = 0.001076
grad_step = 000499, loss = 0.001087
grad_step = 000500, loss = 0.001111
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001156
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

  date_run                              2020-05-13 15:12:44.864272
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.283584
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 15:12:44.870751
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.225551
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 15:12:44.878673
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140558
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 15:12:44.883612
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.42732
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
0   2020-05-13 15:12:16.829481  ...    mean_absolute_error
1   2020-05-13 15:12:16.833086  ...     mean_squared_error
2   2020-05-13 15:12:16.836083  ...  median_absolute_error
3   2020-05-13 15:12:16.839089  ...               r2_score
4   2020-05-13 15:12:26.397470  ...    mean_absolute_error
5   2020-05-13 15:12:26.400996  ...     mean_squared_error
6   2020-05-13 15:12:26.404137  ...  median_absolute_error
7   2020-05-13 15:12:26.407089  ...               r2_score
8   2020-05-13 15:12:44.864272  ...    mean_absolute_error
9   2020-05-13 15:12:44.870751  ...     mean_squared_error
10  2020-05-13 15:12:44.878673  ...  median_absolute_error
11  2020-05-13 15:12:44.883612  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b5f3fccf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3440640/9912422 [00:00<00:00, 34367757.51it/s]9920512it [00:00, 36212527.99it/s]                             
0it [00:00, ?it/s]32768it [00:00, 636356.27it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463569.06it/s]1654784it [00:00, 12461608.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 226867.33it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b11db6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b113e60f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b11db6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b5f403a58> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b0eb77518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b5f403a58> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b11db6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b5f403a58> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b0eb77518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b5f3fccf8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f10d1788208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=df3b3453fb46a1ad334fc14464c15b44249a5f62f3497120214fe35ff0e328e7
  Stored in directory: /tmp/pip-ephem-wheel-cache-eyuiifjw/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f10c7b0e080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2859008/17464789 [===>..........................] - ETA: 0s
10567680/17464789 [=================>............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 15:14:10.726801: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 15:14:10.730736: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-13 15:14:10.730986: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e3bfe763f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 15:14:10.731217: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.9528 - accuracy: 0.4813
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.9465 - accuracy: 0.4818
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.9212 - accuracy: 0.4834
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8634 - accuracy: 0.4872
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.9032 - accuracy: 0.4846
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.8660 - accuracy: 0.4870
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.8148 - accuracy: 0.4903
10000/25000 [===========>..................] - ETA: 3s - loss: 7.8276 - accuracy: 0.4895
11000/25000 [============>.................] - ETA: 3s - loss: 7.8018 - accuracy: 0.4912
12000/25000 [=============>................] - ETA: 3s - loss: 7.7855 - accuracy: 0.4922
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7669 - accuracy: 0.4935
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7488 - accuracy: 0.4946
15000/25000 [=================>............] - ETA: 2s - loss: 7.7280 - accuracy: 0.4960
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7241 - accuracy: 0.4963
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7072 - accuracy: 0.4974
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6811 - accuracy: 0.4991
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6803 - accuracy: 0.4991
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6950 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6757 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6507 - accuracy: 0.5010
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 15:14:24.162603
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 15:14:24.162603  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<42:32:37, 5.63kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<30:01:51, 7.97kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<21:04:27, 11.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 557k/862M [00:01<14:46:45, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 2.15M/862M [00:02<10:19:57, 23.1kB/s].vector_cache/glove.6B.zip:   1%|          | 7.43M/862M [00:02<7:11:24, 33.0kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:02<5:00:06, 47.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:02<3:29:15, 67.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:02<2:25:49, 96.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:02<1:41:37, 137kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:02<1:10:49, 196kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:02<49:27, 279kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<34:31, 397kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:03<35:18, 389kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.2M/862M [00:03<24:41, 553kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:03<17:19, 784kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:03<12:10, 1.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:04<09:26, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:06<3:09:20, 71.0kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:06<2:26:11, 91.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.0M/862M [00:07<1:45:32, 127kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:07<1:14:31, 180kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:07<52:31, 255kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:07<37:00, 362kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:08<31:14, 428kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:08<23:10, 577kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:09<16:26, 811kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:10<14:41, 905kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:10<13:04, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:11<09:51, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:11<07:14, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.3M/862M [00:11<05:23, 2.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:12<10:03, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:12<09:28, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:12<07:18, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:13<05:24, 2.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:14<07:09, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:14<06:30, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:14<04:49, 2.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:15<03:34, 3.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:16<16:49, 778kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:16<13:08, 996kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:16<09:30, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:18<10:06, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:18<09:45, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:19<07:30, 1.73MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:19<05:28, 2.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:20<07:59, 1.62MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:20<06:59, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:21<05:09, 2.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:21<03:48, 3.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:22<22:05, 583kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:22<18:10, 709kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:23<13:15, 971kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:23<09:36, 1.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:24<09:48, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:24<08:33, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:25<06:23, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:26<06:59, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:26<06:28, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:27<04:52, 2.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<03:33, 3.57MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<1:37:16, 130kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<1:09:46, 182kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<49:12, 257kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<34:28, 366kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<1:16:44, 164kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<55:09, 229kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<39:13, 321kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<27:47, 453kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:31<19:51, 633kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:31<14:22, 873kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<20:29, 612kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<17:39, 710kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<13:04, 959kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<09:29, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<09:29, 1.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<09:53, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<07:45, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:35<05:37, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<09:40, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<08:27, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<06:20, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<04:34, 2.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<41:23, 298kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<30:19, 407kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<21:57, 561kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:39<15:29, 793kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<16:22, 749kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<13:07, 934kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<09:35, 1.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<09:02, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<08:00, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<05:57, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<04:21, 2.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<12:10, 996kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<20:35, 589kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<17:16, 702kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<12:41, 954kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<09:11, 1.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<06:38, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<15:19, 787kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<11:39, 1.03MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:47<08:27, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<08:27, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<09:15, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<07:06, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<05:25, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<04:01, 2.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<09:03, 1.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<10:17, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<08:10, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:51<06:00, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<06:14, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<05:11, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<03:56, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<02:54, 4.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<28:51, 408kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<21:51, 539kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<15:36, 753kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<11:07, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<12:38, 926kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<10:32, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<07:48, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<05:38, 2.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<09:41, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<13:15, 878kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<10:37, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<08:09, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<05:54, 1.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<07:27, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<06:48, 1.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:01<03:59, 2.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<03:06, 3.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<10:12, 1.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<09:06, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<07:09, 1.61MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<05:23, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<04:06, 2.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<03:17, 3.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<09:49, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<08:05, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<05:53, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<04:16, 2.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<31:11, 364kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<23:06, 492kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<17:03, 666kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<12:07, 934kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<11:27, 986kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<09:33, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<07:03, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<07:05, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<06:33, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<04:58, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<06:12, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<10:37, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<08:58, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<06:40, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<04:46, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<2:02:07, 90.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<1:26:08, 129kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<1:00:36, 183kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<44:21, 248kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<34:06, 323kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<24:29, 450kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:17<17:26, 630kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<14:24, 760kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<11:33, 947kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<08:27, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<06:02, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<20:12, 539kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<15:40, 694kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<11:25, 951kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<08:18, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<06:07, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:49, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:36, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<06:38, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:47, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<08:07, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<07:01, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<05:14, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<05:48, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<07:00, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<05:34, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<04:12, 2.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<03:06, 3.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<12:32, 845kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<10:13, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<07:30, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<05:21, 1.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<1:00:23, 174kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<43:40, 241kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<30:49, 341kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<21:45, 482kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<18:52, 555kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<14:28, 722kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<10:26, 1.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<09:25, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<08:01, 1.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<05:53, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<04:19, 2.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<07:37, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<06:47, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<05:30, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<04:08, 2.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<03:03, 3.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<11:30, 892kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<11:01, 931kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<08:25, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<06:13, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<04:29, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<13:32, 753kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<10:36, 961kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<08:13, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<05:54, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<04:20, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<1:06:20, 153kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<47:46, 212kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<33:43, 299kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<25:26, 395kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<19:10, 524kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<13:44, 730kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<11:30, 868kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<09:23, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<06:51, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<04:53, 2.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:50<1:26:40, 114kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:50<1:09:01, 144kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<50:44, 195kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<36:01, 275kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<25:14, 391kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:52<22:50, 431kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:52<17:19, 568kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:52<12:25, 791kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<10:32, 927kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<08:42, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:54<06:24, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:56<06:20, 1.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:56<05:47, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:56<04:22, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<03:10, 3.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<15:29, 622kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<11:57, 806kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<08:35, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<06:07, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<18:37, 514kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<18:21, 521kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<14:12, 673kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:00<10:12, 934kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:00<07:17, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<10:20, 919kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<08:23, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:02<06:33, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<04:44, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:04<06:12, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:04<05:36, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:04<04:14, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<04:45, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<04:21, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<03:37, 2.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<02:41, 3.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<04:51, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<04:39, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:08<03:34, 2.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<02:37, 3.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<11:30, 801kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<09:17, 992kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:10<06:45, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<04:52, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<08:02, 1.14MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<06:52, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:12<05:07, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<03:41, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<11:57, 760kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<09:36, 945kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<07:01, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<05:01, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<13:01, 693kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:16<10:21, 870kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:16<07:29, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<05:21, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:18<10:35, 844kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<08:40, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:18<06:21, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<04:33, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<10:25, 852kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<08:29, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:20<06:11, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<04:27, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<08:56, 985kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<07:28, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:22<05:28, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<03:57, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<08:53, 983kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<07:23, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:24<05:27, 1.59MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<05:28, 1.58MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<05:01, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:26<03:46, 2.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<04:17, 2.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<04:10, 2.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:28<03:09, 2.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<02:18, 3.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:30<13:43, 621kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:30<10:45, 793kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:30<07:46, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<05:31, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:32<23:09, 365kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:32<17:22, 487kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:32<12:25, 679kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:34<10:16, 817kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:34<08:19, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:34<06:06, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:36<05:51, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:36<05:13, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:36<03:59, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<02:54, 2.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<02:16, 3.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<3:52:24, 35.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<2:43:44, 50.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:38<1:54:39, 71.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:38<1:23:26, 98.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:40<59:09, 138kB/s]   .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:40<45:58, 177kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:41<33:37, 243kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:41<24:02, 339kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:41<16:56, 479kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<11:57, 677kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:42<3:38:50, 37.0kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:42<2:34:26, 52.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:43<1:48:08, 74.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<1:15:34, 106kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:44<56:27, 142kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:44<40:52, 196kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<28:45, 278kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<20:13, 394kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:46<17:45, 448kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:46<13:16, 599kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<10:01, 792kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:47<07:11, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:48<06:46, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:48<05:55, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<03:15, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:50<04:38, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<04:09, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<03:06, 2.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:51<02:17, 3.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:52<07:52, 983kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<06:24, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<04:40, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:53<03:23, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<07:28, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<06:19, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:40, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<04:40, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<04:20, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<03:17, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<02:22, 3.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:58<25:01, 301kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:58<18:31, 407kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<13:11, 570kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:00<10:37, 703kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:00<08:27, 882kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<06:09, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:02<05:43, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<05:00, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<03:45, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<02:43, 2.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:04<07:48, 940kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:04<06:28, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<04:44, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<03:25, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:06<06:02, 1.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:06<05:14, 1.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<03:54, 1.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:08<04:05, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:08<03:51, 1.87MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<02:56, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:10<03:25, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:10<03:21, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:10<02:35, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<01:53, 3.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:12<13:32, 521kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:12<10:11, 691kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:12<07:18, 961kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<05:11, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:14<09:02, 773kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:14<08:42, 802kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:14<06:50, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:14<05:04, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<03:40, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:16<04:49, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:16<04:19, 1.60MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:16<03:40, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<02:46, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<02:05, 3.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:18<04:03, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:18<03:47, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<02:52, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<03:18, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<03:16, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<02:45, 2.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<02:02, 3.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<03:14, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<03:10, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:22<02:24, 2.78MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<01:48, 3.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:24<04:15, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:24<03:38, 1.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:53, 2.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<02:05, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:26<04:45, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:26<04:07, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:26<03:02, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<02:14, 2.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:28<04:48, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:28<04:05, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:28<03:18, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<02:23, 2.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:30<04:47, 1.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:30<05:08, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:30<04:02, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:32<03:40, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:32<03:27, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:32<02:38, 2.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<01:56, 3.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:34<04:55, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:35<06:55, 909kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:35<05:43, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<04:12, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:36<04:02, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:37<03:39, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:37<03:09, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:37<02:25, 2.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<01:46, 3.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:38<07:48, 788kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:39<06:18, 974kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<04:39, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<03:22, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<02:27, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:40<1:55:38, 52.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:41<1:21:52, 74.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:41<57:26, 106kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:41<40:09, 150kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:42<29:56, 201kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:43<21:49, 275kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:43<15:28, 387kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:43<10:53, 548kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:44<09:48, 606kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:44<07:40, 775kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:45<05:34, 1.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:45<03:58, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:46<05:25, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:46<04:34, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:47<03:23, 1.72MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:47<02:25, 2.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:48<13:54, 418kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:48<10:31, 552kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:49<07:31, 770kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:49<05:20, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:50<06:34, 874kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:50<05:16, 1.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:51<03:52, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:51<02:46, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:52<09:59, 567kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:52<07:46, 729kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:53<05:36, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:54<04:58, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:54<04:14, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:54<03:09, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:55<02:15, 2.46MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:56<28:38, 193kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:56<20:47, 266kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:56<14:39, 376kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:57<10:15, 533kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:58<13:50, 395kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:58<10:24, 524kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:58<07:26, 730kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:00<06:12, 868kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:00<05:03, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:00<03:42, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:02<03:36, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:02<03:16, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:02<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:04<02:43, 1.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:04<02:37, 2.01MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:04<01:59, 2.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:06<02:22, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:06<02:23, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:06<01:50, 2.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:08<02:15, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:08<02:24, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:08<02:00, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:08<01:30, 3.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:10<02:20, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:10<02:19, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:10<01:47, 2.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:12<02:12, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:12<02:28, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:12<01:58, 2.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<01:28, 3.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:14<02:19, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:14<02:52, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:14<02:43, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:14<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:14<01:32, 3.17MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:16<03:10, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:16<02:49, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:16<02:18, 2.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<01:41, 2.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:18<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:18<02:47, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:18<02:20, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:18<01:47, 2.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<01:18, 3.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:20<06:46, 695kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:20<05:57, 788kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:20<04:29, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:20<03:12, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:22<03:33, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:22<03:07, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<02:20, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<01:41, 2.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:24<04:31, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:24<03:47, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<02:46, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<02:00, 2.24MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:26<03:26, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:26<02:56, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:26<02:09, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<01:34, 2.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:28<04:31, 977kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:28<03:46, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<02:46, 1.59MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<01:58, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:31<15:26, 282kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:31<15:10, 287kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:31<11:39, 373kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:31<08:22, 519kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:31<05:58, 725kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:33<04:57, 864kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:33<04:04, 1.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:33<02:57, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:33<02:08, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:35<03:27, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:35<02:59, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:35<02:11, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:35<01:36, 2.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:37<03:03, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:37<02:37, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:37<01:58, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:37<01:25, 2.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:38<04:14, 963kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:39<03:26, 1.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:39<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:39<01:48, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:40<04:42, 852kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:41<03:50, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:41<02:52, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:41<02:03, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:42<03:12, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:43<02:47, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:43<02:02, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:43<01:29, 2.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:44<02:46, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:45<02:28, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:45<01:50, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:45<01:20, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:46<02:31, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:47<02:13, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:47<01:45, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:47<01:17, 2.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:49<02:36, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:49<02:51, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:49<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:49<01:41, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:51<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:51<01:47, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:51<01:23, 2.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:51<01:01, 3.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<02:19, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:53<02:06, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:53<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:53<01:10, 3.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:55<02:41, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:55<02:45, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:55<02:07, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:55<01:30, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:57<03:15, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:57<02:45, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:57<02:02, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<02:03, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<01:54, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:59<01:26, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:59<01:02, 3.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:01<08:30, 390kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:01<06:19, 523kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:01<04:29, 734kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:01<03:09, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:03<04:56, 657kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:03<03:53, 834kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:03<02:48, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:05<02:33, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:05<02:13, 1.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:05<01:38, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:07<01:44, 1.79MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:07<01:38, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:07<01:15, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:09<01:32, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:09<01:55, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:09<01:32, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:09<01:07, 2.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:11<01:38, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:11<01:32, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:11<01:13, 2.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:11<00:54, 3.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:11<00:42, 4.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:13<03:17, 883kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:13<02:54, 997kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:13<02:18, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:13<01:40, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:13<01:12, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:15<02:38, 1.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:15<02:13, 1.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:15<01:45, 1.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:15<01:19, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:15<00:58, 2.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:17<01:47, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:17<01:45, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:17<01:27, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:17<01:07, 2.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:17<00:49, 3.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:19<01:58, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:19<01:54, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:19<01:32, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:19<01:10, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:19<00:51, 3.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:21<02:22, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:21<03:19, 792kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:22<02:37, 998kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:22<02:01, 1.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:22<01:26, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:23<01:47, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:23<01:37, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:23<01:13, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:24<00:54, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:25<01:18, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:25<01:17, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:25<00:59, 2.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:26<00:47, 3.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:26<00:33, 4.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<27:24, 87.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<21:09, 113kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<15:21, 155kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:29<10:50, 219kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:29<07:28, 311kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<07:16, 319kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<05:20, 433kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:30<03:47, 606kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:30<02:38, 855kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<04:39, 482kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<03:33, 631kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:32<02:32, 874kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<02:10, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:34<01:48, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:34<01:19, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<01:18, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<01:30, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<01:11, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:36<00:51, 2.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<01:32, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<01:21, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:38<01:00, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<01:04, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<01:01, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:40<00:46, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<00:53, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<00:52, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:42<00:39, 2.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<00:48, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:44<00:46, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:44<00:35, 3.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<00:46, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:46<00:46, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<00:36, 2.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:44, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:48<00:44, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<00:34, 2.91MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<00:42, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<00:42, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:50<00:32, 2.93MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<00:43, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<00:56, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<00:45, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:53<00:34, 2.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:53<00:24, 3.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<06:33, 227kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:54<04:39, 317kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:54<03:11, 450kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<02:54, 486kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<02:25, 586kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:56<01:46, 791kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:56<01:14, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<01:14, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:58<01:02, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:58<00:45, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:58<00:31, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<03:32, 361kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:00<02:38, 482kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:00<01:51, 672kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:00<01:16, 950kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:02<1:12:23, 16.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:02<50:37, 23.8kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:02<34:53, 33.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:04<23:44, 48.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:04<16:42, 68.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:04<11:31, 96.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<07:57, 135kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<05:41, 188kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:06<03:56, 266kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:06<02:39, 378kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<03:36, 278kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<02:37, 380kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:08<01:49, 535kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:08<01:14, 755kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:10<01:36, 579kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:10<01:13, 754kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:10<00:51, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:10<00:35, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<01:28, 588kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:12<01:08, 755kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:12<00:48, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:12<00:32, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:14<01:03, 752kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:14<00:50, 935kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:14<00:36, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:16<00:32, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:16<00:27, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:16<00:19, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:16<00:13, 2.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:18<00:41, 950kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:18<00:34, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:18<00:24, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:18<00:16, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:20<00:31, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:20<00:26, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:20<00:19, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:20<00:13, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:22, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:19, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:22<00:14, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:22<00:09, 2.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:28, 959kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:24<00:23, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:24<00:16, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:24<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:17, 1.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:26<00:15, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:26<00:10, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:26<00:07, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<00:13, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<00:11, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:28<00:08, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:28<00:05, 2.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:11, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:09, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:30<00:06, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:30<00:04, 2.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:07, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:06, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:32<00:04, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:32<00:02, 2.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:04, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:03, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:34<00:02, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:34<00:00, 3.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:36<00:02, 968kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:36<00:01, 1.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:36<00:00, 1.58MB/s].vector_cache/glove.6B.zip: 862MB [06:36, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 871/400000 [00:00<00:45, 8708.31it/s]  0%|          | 1759/400000 [00:00<00:45, 8757.17it/s]  1%|          | 2650/400000 [00:00<00:45, 8799.02it/s]  1%|          | 3537/400000 [00:00<00:44, 8818.07it/s]  1%|          | 4448/400000 [00:00<00:44, 8903.39it/s]  1%|▏         | 5337/400000 [00:00<00:44, 8899.21it/s]  2%|▏         | 6240/400000 [00:00<00:44, 8935.50it/s]  2%|▏         | 7104/400000 [00:00<00:44, 8844.27it/s]  2%|▏         | 7999/400000 [00:00<00:44, 8875.70it/s]  2%|▏         | 8890/400000 [00:01<00:44, 8885.61it/s]  2%|▏         | 9794/400000 [00:01<00:43, 8930.23it/s]  3%|▎         | 10704/400000 [00:01<00:43, 8980.03it/s]  3%|▎         | 11607/400000 [00:01<00:43, 8993.56it/s]  3%|▎         | 12499/400000 [00:01<00:43, 8936.10it/s]  3%|▎         | 13394/400000 [00:01<00:43, 8939.57it/s]  4%|▎         | 14293/400000 [00:01<00:43, 8954.62it/s]  4%|▍         | 15189/400000 [00:01<00:42, 8954.37it/s]  4%|▍         | 16083/400000 [00:01<00:43, 8886.05it/s]  4%|▍         | 16975/400000 [00:01<00:43, 8893.55it/s]  4%|▍         | 17864/400000 [00:02<00:43, 8845.97it/s]  5%|▍         | 18755/400000 [00:02<00:43, 8862.70it/s]  5%|▍         | 19651/400000 [00:02<00:42, 8889.29it/s]  5%|▌         | 20549/400000 [00:02<00:42, 8914.11it/s]  5%|▌         | 21441/400000 [00:02<00:42, 8865.80it/s]  6%|▌         | 22328/400000 [00:02<00:42, 8806.64it/s]  6%|▌         | 23228/400000 [00:02<00:42, 8863.71it/s]  6%|▌         | 24116/400000 [00:02<00:42, 8866.37it/s]  6%|▋         | 25003/400000 [00:02<00:42, 8835.53it/s]  6%|▋         | 25887/400000 [00:02<00:43, 8629.19it/s]  7%|▋         | 26764/400000 [00:03<00:43, 8669.74it/s]  7%|▋         | 27632/400000 [00:03<00:43, 8611.88it/s]  7%|▋         | 28518/400000 [00:03<00:42, 8682.84it/s]  7%|▋         | 29387/400000 [00:03<00:42, 8684.34it/s]  8%|▊         | 30256/400000 [00:03<00:42, 8641.22it/s]  8%|▊         | 31128/400000 [00:03<00:42, 8662.51it/s]  8%|▊         | 32008/400000 [00:03<00:42, 8701.32it/s]  8%|▊         | 32899/400000 [00:03<00:41, 8762.58it/s]  8%|▊         | 33807/400000 [00:03<00:41, 8855.10it/s]  9%|▊         | 34693/400000 [00:03<00:41, 8734.47it/s]  9%|▉         | 35591/400000 [00:04<00:41, 8804.23it/s]  9%|▉         | 36485/400000 [00:04<00:41, 8842.37it/s]  9%|▉         | 37375/400000 [00:04<00:40, 8857.38it/s] 10%|▉         | 38262/400000 [00:04<00:40, 8857.24it/s] 10%|▉         | 39148/400000 [00:04<00:40, 8838.15it/s] 10%|█         | 40032/400000 [00:04<00:40, 8786.41it/s] 10%|█         | 40938/400000 [00:04<00:40, 8864.67it/s] 10%|█         | 41845/400000 [00:04<00:40, 8922.81it/s] 11%|█         | 42750/400000 [00:04<00:39, 8957.82it/s] 11%|█         | 43647/400000 [00:04<00:39, 8941.62it/s] 11%|█         | 44545/400000 [00:05<00:39, 8950.74it/s] 11%|█▏        | 45441/400000 [00:05<00:39, 8879.44it/s] 12%|█▏        | 46334/400000 [00:05<00:39, 8894.52it/s] 12%|█▏        | 47244/400000 [00:05<00:39, 8953.55it/s] 12%|█▏        | 48140/400000 [00:05<00:39, 8888.69it/s] 12%|█▏        | 49030/400000 [00:05<00:39, 8838.16it/s] 12%|█▏        | 49943/400000 [00:05<00:39, 8922.33it/s] 13%|█▎        | 50849/400000 [00:05<00:38, 8962.29it/s] 13%|█▎        | 51746/400000 [00:05<00:39, 8918.94it/s] 13%|█▎        | 52639/400000 [00:05<00:39, 8902.45it/s] 13%|█▎        | 53530/400000 [00:06<00:38, 8885.36it/s] 14%|█▎        | 54439/400000 [00:06<00:38, 8945.12it/s] 14%|█▍        | 55346/400000 [00:06<00:38, 8980.52it/s] 14%|█▍        | 56248/400000 [00:06<00:38, 8989.72it/s] 14%|█▍        | 57150/400000 [00:06<00:38, 8995.69it/s] 15%|█▍        | 58050/400000 [00:06<00:38, 8986.22it/s] 15%|█▍        | 58949/400000 [00:06<00:38, 8966.92it/s] 15%|█▍        | 59846/400000 [00:06<00:38, 8923.31it/s] 15%|█▌        | 60739/400000 [00:06<00:38, 8765.09it/s] 15%|█▌        | 61617/400000 [00:06<00:38, 8717.70it/s] 16%|█▌        | 62500/400000 [00:07<00:38, 8749.56it/s] 16%|█▌        | 63376/400000 [00:07<00:38, 8750.88it/s] 16%|█▌        | 64270/400000 [00:07<00:38, 8805.16it/s] 16%|█▋        | 65151/400000 [00:07<00:38, 8708.95it/s] 17%|█▋        | 66023/400000 [00:07<00:38, 8703.96it/s] 17%|█▋        | 66894/400000 [00:07<00:38, 8696.88it/s] 17%|█▋        | 67770/400000 [00:07<00:38, 8713.46it/s] 17%|█▋        | 68656/400000 [00:07<00:37, 8756.35it/s] 17%|█▋        | 69565/400000 [00:07<00:37, 8853.00it/s] 18%|█▊        | 70466/400000 [00:07<00:37, 8897.75it/s] 18%|█▊        | 71357/400000 [00:08<00:37, 8855.12it/s] 18%|█▊        | 72243/400000 [00:08<00:37, 8797.88it/s] 18%|█▊        | 73146/400000 [00:08<00:36, 8866.19it/s] 19%|█▊        | 74033/400000 [00:08<00:36, 8815.35it/s] 19%|█▊        | 74932/400000 [00:08<00:36, 8865.02it/s] 19%|█▉        | 75819/400000 [00:08<00:36, 8805.49it/s] 19%|█▉        | 76735/400000 [00:08<00:36, 8907.08it/s] 19%|█▉        | 77648/400000 [00:08<00:35, 8971.50it/s] 20%|█▉        | 78562/400000 [00:08<00:35, 9018.80it/s] 20%|█▉        | 79465/400000 [00:08<00:35, 8930.62it/s] 20%|██        | 80359/400000 [00:09<00:36, 8876.68it/s] 20%|██        | 81248/400000 [00:09<00:36, 8853.62it/s] 21%|██        | 82151/400000 [00:09<00:35, 8905.22it/s] 21%|██        | 83044/400000 [00:09<00:35, 8911.74it/s] 21%|██        | 83958/400000 [00:09<00:35, 8978.71it/s] 21%|██        | 84857/400000 [00:09<00:35, 8915.76it/s] 21%|██▏       | 85755/400000 [00:09<00:35, 8934.76it/s] 22%|██▏       | 86669/400000 [00:09<00:34, 8994.73it/s] 22%|██▏       | 87569/400000 [00:09<00:34, 8975.84it/s] 22%|██▏       | 88467/400000 [00:09<00:35, 8853.70it/s] 22%|██▏       | 89353/400000 [00:10<00:35, 8807.82it/s] 23%|██▎       | 90272/400000 [00:10<00:34, 8916.44it/s] 23%|██▎       | 91189/400000 [00:10<00:34, 8989.32it/s] 23%|██▎       | 92101/400000 [00:10<00:34, 9026.05it/s] 23%|██▎       | 93005/400000 [00:10<00:34, 8963.34it/s] 23%|██▎       | 93902/400000 [00:10<00:34, 8923.11it/s] 24%|██▎       | 94819/400000 [00:10<00:33, 8993.82it/s] 24%|██▍       | 95738/400000 [00:10<00:33, 9051.13it/s] 24%|██▍       | 96644/400000 [00:10<00:33, 9047.66it/s] 24%|██▍       | 97550/400000 [00:10<00:33, 8990.87it/s] 25%|██▍       | 98450/400000 [00:11<00:34, 8820.94it/s] 25%|██▍       | 99342/400000 [00:11<00:33, 8847.74it/s] 25%|██▌       | 100236/400000 [00:11<00:33, 8873.04it/s] 25%|██▌       | 101125/400000 [00:11<00:33, 8875.97it/s] 26%|██▌       | 102036/400000 [00:11<00:33, 8942.55it/s] 26%|██▌       | 102931/400000 [00:11<00:33, 8931.28it/s] 26%|██▌       | 103836/400000 [00:11<00:33, 8964.69it/s] 26%|██▌       | 104753/400000 [00:11<00:32, 9023.82it/s] 26%|██▋       | 105656/400000 [00:11<00:32, 8981.80it/s] 27%|██▋       | 106555/400000 [00:12<00:32, 8926.39it/s] 27%|██▋       | 107448/400000 [00:12<00:33, 8859.16it/s] 27%|██▋       | 108335/400000 [00:12<00:33, 8738.37it/s] 27%|██▋       | 109220/400000 [00:12<00:33, 8771.48it/s] 28%|██▊       | 110098/400000 [00:12<00:33, 8743.20it/s] 28%|██▊       | 110980/400000 [00:12<00:32, 8764.45it/s] 28%|██▊       | 111865/400000 [00:12<00:32, 8787.01it/s] 28%|██▊       | 112762/400000 [00:12<00:32, 8840.54it/s] 28%|██▊       | 113647/400000 [00:12<00:32, 8811.30it/s] 29%|██▊       | 114540/400000 [00:12<00:32, 8845.97it/s] 29%|██▉       | 115428/400000 [00:13<00:32, 8855.28it/s] 29%|██▉       | 116319/400000 [00:13<00:31, 8870.20it/s] 29%|██▉       | 117220/400000 [00:13<00:31, 8909.52it/s] 30%|██▉       | 118112/400000 [00:13<00:31, 8895.52it/s] 30%|██▉       | 119004/400000 [00:13<00:31, 8901.11it/s] 30%|██▉       | 119895/400000 [00:13<00:31, 8805.70it/s] 30%|███       | 120776/400000 [00:13<00:31, 8772.29it/s] 30%|███       | 121654/400000 [00:13<00:31, 8768.71it/s] 31%|███       | 122535/400000 [00:13<00:31, 8778.88it/s] 31%|███       | 123413/400000 [00:13<00:31, 8776.85it/s] 31%|███       | 124293/400000 [00:14<00:31, 8782.22it/s] 31%|███▏      | 125172/400000 [00:14<00:31, 8656.59it/s] 32%|███▏      | 126039/400000 [00:14<00:32, 8406.41it/s] 32%|███▏      | 126912/400000 [00:14<00:32, 8499.81it/s] 32%|███▏      | 127782/400000 [00:14<00:31, 8556.55it/s] 32%|███▏      | 128655/400000 [00:14<00:31, 8605.37it/s] 32%|███▏      | 129517/400000 [00:14<00:31, 8606.72it/s] 33%|███▎      | 130381/400000 [00:14<00:31, 8615.26it/s] 33%|███▎      | 131248/400000 [00:14<00:31, 8631.37it/s] 33%|███▎      | 132132/400000 [00:14<00:30, 8691.63it/s] 33%|███▎      | 133010/400000 [00:15<00:30, 8717.24it/s] 33%|███▎      | 133882/400000 [00:15<00:30, 8665.06it/s] 34%|███▎      | 134759/400000 [00:15<00:30, 8696.11it/s] 34%|███▍      | 135629/400000 [00:15<00:30, 8666.46it/s] 34%|███▍      | 136499/400000 [00:15<00:30, 8673.76it/s] 34%|███▍      | 137392/400000 [00:15<00:30, 8747.66it/s] 35%|███▍      | 138286/400000 [00:15<00:29, 8802.19it/s] 35%|███▍      | 139183/400000 [00:15<00:29, 8849.88it/s] 35%|███▌      | 140069/400000 [00:15<00:29, 8824.92it/s] 35%|███▌      | 140963/400000 [00:15<00:29, 8857.44it/s] 35%|███▌      | 141856/400000 [00:16<00:29, 8877.74it/s] 36%|███▌      | 142744/400000 [00:16<00:29, 8771.74it/s] 36%|███▌      | 143622/400000 [00:16<00:29, 8764.19it/s] 36%|███▌      | 144499/400000 [00:16<00:29, 8765.45it/s] 36%|███▋      | 145376/400000 [00:16<00:29, 8764.84it/s] 37%|███▋      | 146256/400000 [00:16<00:28, 8772.54it/s] 37%|███▋      | 147134/400000 [00:16<00:28, 8772.35it/s] 37%|███▋      | 148027/400000 [00:16<00:28, 8817.62it/s] 37%|███▋      | 148935/400000 [00:16<00:28, 8892.64it/s] 37%|███▋      | 149834/400000 [00:16<00:28, 8921.36it/s] 38%|███▊      | 150727/400000 [00:17<00:27, 8907.11it/s] 38%|███▊      | 151618/400000 [00:17<00:28, 8830.20it/s] 38%|███▊      | 152502/400000 [00:17<00:29, 8503.68it/s] 38%|███▊      | 153397/400000 [00:17<00:28, 8632.78it/s] 39%|███▊      | 154287/400000 [00:17<00:28, 8710.78it/s] 39%|███▉      | 155199/400000 [00:17<00:27, 8829.04it/s] 39%|███▉      | 156087/400000 [00:17<00:27, 8842.46it/s] 39%|███▉      | 156973/400000 [00:17<00:27, 8772.66it/s] 39%|███▉      | 157852/400000 [00:17<00:27, 8727.10it/s] 40%|███▉      | 158737/400000 [00:17<00:27, 8763.34it/s] 40%|███▉      | 159621/400000 [00:18<00:27, 8785.56it/s] 40%|████      | 160500/400000 [00:18<00:27, 8768.19it/s] 40%|████      | 161378/400000 [00:18<00:27, 8761.73it/s] 41%|████      | 162269/400000 [00:18<00:27, 8804.52it/s] 41%|████      | 163156/400000 [00:18<00:26, 8822.15it/s] 41%|████      | 164045/400000 [00:18<00:26, 8842.09it/s] 41%|████      | 164935/400000 [00:18<00:26, 8858.96it/s] 41%|████▏     | 165821/400000 [00:18<00:27, 8547.63it/s] 42%|████▏     | 166705/400000 [00:18<00:27, 8631.56it/s] 42%|████▏     | 167608/400000 [00:18<00:26, 8746.79it/s] 42%|████▏     | 168504/400000 [00:19<00:26, 8809.49it/s] 42%|████▏     | 169387/400000 [00:19<00:26, 8643.39it/s] 43%|████▎     | 170262/400000 [00:19<00:26, 8670.47it/s] 43%|████▎     | 171137/400000 [00:19<00:26, 8692.91it/s] 43%|████▎     | 172008/400000 [00:19<00:26, 8669.15it/s] 43%|████▎     | 172904/400000 [00:19<00:25, 8752.39it/s] 43%|████▎     | 173785/400000 [00:19<00:25, 8768.24it/s] 44%|████▎     | 174663/400000 [00:19<00:25, 8697.72it/s] 44%|████▍     | 175547/400000 [00:19<00:25, 8738.58it/s] 44%|████▍     | 176432/400000 [00:19<00:25, 8770.85it/s] 44%|████▍     | 177317/400000 [00:20<00:25, 8794.30it/s] 45%|████▍     | 178197/400000 [00:20<00:25, 8792.47it/s] 45%|████▍     | 179077/400000 [00:20<00:25, 8725.02it/s] 45%|████▍     | 179960/400000 [00:20<00:25, 8755.54it/s] 45%|████▌     | 180836/400000 [00:20<00:25, 8751.00it/s] 45%|████▌     | 181739/400000 [00:20<00:24, 8831.51it/s] 46%|████▌     | 182646/400000 [00:20<00:24, 8898.86it/s] 46%|████▌     | 183537/400000 [00:20<00:24, 8850.91it/s] 46%|████▌     | 184423/400000 [00:20<00:24, 8821.06it/s] 46%|████▋     | 185309/400000 [00:20<00:24, 8830.54it/s] 47%|████▋     | 186197/400000 [00:21<00:24, 8843.58it/s] 47%|████▋     | 187082/400000 [00:21<00:24, 8818.47it/s] 47%|████▋     | 187964/400000 [00:21<00:24, 8784.79it/s] 47%|████▋     | 188851/400000 [00:21<00:23, 8808.37it/s] 47%|████▋     | 189741/400000 [00:21<00:23, 8832.84it/s] 48%|████▊     | 190635/400000 [00:21<00:23, 8864.01it/s] 48%|████▊     | 191526/400000 [00:21<00:23, 8876.63it/s] 48%|████▊     | 192414/400000 [00:21<00:23, 8822.47it/s] 48%|████▊     | 193310/400000 [00:21<00:23, 8860.59it/s] 49%|████▊     | 194211/400000 [00:22<00:23, 8902.56it/s] 49%|████▉     | 195102/400000 [00:22<00:23, 8897.69it/s] 49%|████▉     | 195992/400000 [00:22<00:22, 8893.73it/s] 49%|████▉     | 196882/400000 [00:22<00:23, 8818.68it/s] 49%|████▉     | 197765/400000 [00:22<00:23, 8750.48it/s] 50%|████▉     | 198641/400000 [00:22<00:23, 8703.79it/s] 50%|████▉     | 199512/400000 [00:22<00:23, 8696.16it/s] 50%|█████     | 200418/400000 [00:22<00:22, 8799.54it/s] 50%|█████     | 201299/400000 [00:22<00:22, 8744.62it/s] 51%|█████     | 202202/400000 [00:22<00:22, 8827.02it/s] 51%|█████     | 203086/400000 [00:23<00:22, 8678.06it/s] 51%|█████     | 203968/400000 [00:23<00:22, 8717.37it/s] 51%|█████     | 204841/400000 [00:23<00:22, 8716.41it/s] 51%|█████▏    | 205714/400000 [00:23<00:22, 8633.68it/s] 52%|█████▏    | 206602/400000 [00:23<00:22, 8705.91it/s] 52%|█████▏    | 207487/400000 [00:23<00:22, 8746.62it/s] 52%|█████▏    | 208387/400000 [00:23<00:21, 8819.32it/s] 52%|█████▏    | 209290/400000 [00:23<00:21, 8878.78it/s] 53%|█████▎    | 210179/400000 [00:23<00:21, 8867.07it/s] 53%|█████▎    | 211066/400000 [00:23<00:21, 8847.00it/s] 53%|█████▎    | 211951/400000 [00:24<00:21, 8785.04it/s] 53%|█████▎    | 212845/400000 [00:24<00:21, 8830.06it/s] 53%|█████▎    | 213739/400000 [00:24<00:21, 8861.09it/s] 54%|█████▎    | 214631/400000 [00:24<00:20, 8877.76it/s] 54%|█████▍    | 215519/400000 [00:24<00:21, 8742.94it/s] 54%|█████▍    | 216394/400000 [00:24<00:21, 8736.90it/s] 54%|█████▍    | 217285/400000 [00:24<00:20, 8787.31it/s] 55%|█████▍    | 218175/400000 [00:24<00:20, 8820.03it/s] 55%|█████▍    | 219059/400000 [00:24<00:20, 8825.75it/s] 55%|█████▍    | 219960/400000 [00:24<00:20, 8877.88it/s] 55%|█████▌    | 220861/400000 [00:25<00:20, 8916.99it/s] 55%|█████▌    | 221753/400000 [00:25<00:20, 8806.62it/s] 56%|█████▌    | 222650/400000 [00:25<00:20, 8853.45it/s] 56%|█████▌    | 223536/400000 [00:25<00:20, 8549.48it/s] 56%|█████▌    | 224437/400000 [00:25<00:20, 8680.44it/s] 56%|█████▋    | 225327/400000 [00:25<00:19, 8743.23it/s] 57%|█████▋    | 226222/400000 [00:25<00:19, 8803.47it/s] 57%|█████▋    | 227109/400000 [00:25<00:19, 8821.50it/s] 57%|█████▋    | 227993/400000 [00:25<00:19, 8778.40it/s] 57%|█████▋    | 228872/400000 [00:25<00:19, 8772.56it/s] 57%|█████▋    | 229754/400000 [00:26<00:19, 8786.53it/s] 58%|█████▊    | 230633/400000 [00:26<00:19, 8746.76it/s] 58%|█████▊    | 231508/400000 [00:26<00:19, 8746.68it/s] 58%|█████▊    | 232383/400000 [00:26<00:19, 8729.68it/s] 58%|█████▊    | 233257/400000 [00:26<00:19, 8687.09it/s] 59%|█████▊    | 234126/400000 [00:26<00:20, 8290.52it/s] 59%|█████▉    | 235015/400000 [00:26<00:19, 8459.39it/s] 59%|█████▉    | 235915/400000 [00:26<00:19, 8613.89it/s] 59%|█████▉    | 236791/400000 [00:26<00:18, 8656.34it/s] 59%|█████▉    | 237674/400000 [00:26<00:18, 8705.30it/s] 60%|█████▉    | 238554/400000 [00:27<00:18, 8731.14it/s] 60%|█████▉    | 239429/400000 [00:27<00:18, 8710.39it/s] 60%|██████    | 240301/400000 [00:27<00:18, 8659.41it/s] 60%|██████    | 241168/400000 [00:27<00:18, 8656.45it/s] 61%|██████    | 242058/400000 [00:27<00:18, 8727.64it/s] 61%|██████    | 242960/400000 [00:27<00:17, 8810.78it/s] 61%|██████    | 243843/400000 [00:27<00:17, 8816.08it/s] 61%|██████    | 244725/400000 [00:27<00:18, 8531.28it/s] 61%|██████▏   | 245586/400000 [00:27<00:18, 8554.19it/s] 62%|██████▏   | 246454/400000 [00:27<00:17, 8591.41it/s] 62%|██████▏   | 247315/400000 [00:28<00:17, 8565.64it/s] 62%|██████▏   | 248195/400000 [00:28<00:17, 8634.29it/s] 62%|██████▏   | 249068/400000 [00:28<00:17, 8660.46it/s] 62%|██████▏   | 249935/400000 [00:28<00:17, 8552.95it/s] 63%|██████▎   | 250803/400000 [00:28<00:17, 8589.38it/s] 63%|██████▎   | 251663/400000 [00:28<00:17, 8470.87it/s] 63%|██████▎   | 252522/400000 [00:28<00:17, 8504.62it/s] 63%|██████▎   | 253404/400000 [00:28<00:17, 8596.16it/s] 64%|██████▎   | 254265/400000 [00:28<00:16, 8576.85it/s] 64%|██████▍   | 255158/400000 [00:28<00:16, 8678.60it/s] 64%|██████▍   | 256038/400000 [00:29<00:16, 8712.83it/s] 64%|██████▍   | 256924/400000 [00:29<00:16, 8755.57it/s] 64%|██████▍   | 257820/400000 [00:29<00:16, 8814.60it/s] 65%|██████▍   | 258702/400000 [00:29<00:16, 8782.39it/s] 65%|██████▍   | 259581/400000 [00:29<00:15, 8777.42it/s] 65%|██████▌   | 260463/400000 [00:29<00:15, 8787.97it/s] 65%|██████▌   | 261342/400000 [00:29<00:15, 8779.51it/s] 66%|██████▌   | 262221/400000 [00:29<00:15, 8775.15it/s] 66%|██████▌   | 263101/400000 [00:29<00:15, 8782.08it/s] 66%|██████▌   | 263992/400000 [00:29<00:15, 8819.08it/s] 66%|██████▌   | 264874/400000 [00:30<00:15, 8743.20it/s] 66%|██████▋   | 265749/400000 [00:30<00:15, 8646.28it/s] 67%|██████▋   | 266635/400000 [00:30<00:15, 8706.95it/s] 67%|██████▋   | 267509/400000 [00:30<00:15, 8714.20it/s] 67%|██████▋   | 268407/400000 [00:30<00:14, 8791.55it/s] 67%|██████▋   | 269310/400000 [00:30<00:14, 8860.58it/s] 68%|██████▊   | 270209/400000 [00:30<00:14, 8898.89it/s] 68%|██████▊   | 271100/400000 [00:30<00:14, 8879.99it/s] 68%|██████▊   | 271989/400000 [00:30<00:14, 8848.05it/s] 68%|██████▊   | 272875/400000 [00:31<00:14, 8849.51it/s] 68%|██████▊   | 273767/400000 [00:31<00:14, 8870.36it/s] 69%|██████▊   | 274656/400000 [00:31<00:14, 8875.99it/s] 69%|██████▉   | 275544/400000 [00:31<00:14, 8876.43it/s] 69%|██████▉   | 276432/400000 [00:31<00:13, 8846.79it/s] 69%|██████▉   | 277317/400000 [00:31<00:13, 8808.99it/s] 70%|██████▉   | 278198/400000 [00:31<00:14, 8685.27it/s] 70%|██████▉   | 279067/400000 [00:31<00:14, 8433.63it/s] 70%|██████▉   | 279955/400000 [00:31<00:14, 8562.67it/s] 70%|███████   | 280835/400000 [00:31<00:13, 8629.85it/s] 70%|███████   | 281714/400000 [00:32<00:13, 8676.55it/s] 71%|███████   | 282605/400000 [00:32<00:13, 8742.56it/s] 71%|███████   | 283488/400000 [00:32<00:13, 8768.18it/s] 71%|███████   | 284373/400000 [00:32<00:13, 8790.93it/s] 71%|███████▏  | 285253/400000 [00:32<00:13, 8792.76it/s] 72%|███████▏  | 286133/400000 [00:32<00:13, 8712.19it/s] 72%|███████▏  | 287028/400000 [00:32<00:12, 8780.34it/s] 72%|███████▏  | 287914/400000 [00:32<00:12, 8802.03it/s] 72%|███████▏  | 288795/400000 [00:32<00:12, 8560.58it/s] 72%|███████▏  | 289657/400000 [00:32<00:12, 8577.47it/s] 73%|███████▎  | 290516/400000 [00:33<00:12, 8567.12it/s] 73%|███████▎  | 291393/400000 [00:33<00:12, 8625.98it/s] 73%|███████▎  | 292290/400000 [00:33<00:12, 8725.64it/s] 73%|███████▎  | 293185/400000 [00:33<00:12, 8790.79it/s] 74%|███████▎  | 294067/400000 [00:33<00:12, 8798.11it/s] 74%|███████▎  | 294965/400000 [00:33<00:11, 8851.70it/s] 74%|███████▍  | 295865/400000 [00:33<00:11, 8894.79it/s] 74%|███████▍  | 296755/400000 [00:33<00:11, 8723.11it/s] 74%|███████▍  | 297642/400000 [00:33<00:11, 8764.16it/s] 75%|███████▍  | 298520/400000 [00:33<00:11, 8736.55it/s] 75%|███████▍  | 299405/400000 [00:34<00:11, 8769.20it/s] 75%|███████▌  | 300283/400000 [00:34<00:11, 8592.08it/s] 75%|███████▌  | 301144/400000 [00:34<00:11, 8568.48it/s] 76%|███████▌  | 302005/400000 [00:34<00:11, 8580.22it/s] 76%|███████▌  | 302864/400000 [00:34<00:11, 8501.52it/s] 76%|███████▌  | 303719/400000 [00:34<00:11, 8515.04it/s] 76%|███████▌  | 304616/400000 [00:34<00:11, 8645.97it/s] 76%|███████▋  | 305500/400000 [00:34<00:10, 8702.40it/s] 77%|███████▋  | 306403/400000 [00:34<00:10, 8796.43it/s] 77%|███████▋  | 307295/400000 [00:34<00:10, 8832.15it/s] 77%|███████▋  | 308194/400000 [00:35<00:10, 8876.66it/s] 77%|███████▋  | 309085/400000 [00:35<00:10, 8884.69it/s] 77%|███████▋  | 309984/400000 [00:35<00:10, 8913.13it/s] 78%|███████▊  | 310878/400000 [00:35<00:09, 8918.80it/s] 78%|███████▊  | 311771/400000 [00:35<00:09, 8852.79it/s] 78%|███████▊  | 312670/400000 [00:35<00:09, 8891.54it/s] 78%|███████▊  | 313572/400000 [00:35<00:09, 8927.16it/s] 79%|███████▊  | 314480/400000 [00:35<00:09, 8970.66it/s] 79%|███████▉  | 315378/400000 [00:35<00:09, 8972.61it/s] 79%|███████▉  | 316276/400000 [00:35<00:09, 8901.96it/s] 79%|███████▉  | 317167/400000 [00:36<00:09, 8888.77it/s] 80%|███████▉  | 318057/400000 [00:36<00:09, 8857.32it/s] 80%|███████▉  | 318950/400000 [00:36<00:09, 8877.04it/s] 80%|███████▉  | 319843/400000 [00:36<00:09, 8890.34it/s] 80%|████████  | 320733/400000 [00:36<00:08, 8882.15it/s] 80%|████████  | 321627/400000 [00:36<00:08, 8899.40it/s] 81%|████████  | 322522/400000 [00:36<00:08, 8914.30it/s] 81%|████████  | 323414/400000 [00:36<00:08, 8904.00it/s] 81%|████████  | 324308/400000 [00:36<00:08, 8912.86it/s] 81%|████████▏ | 325200/400000 [00:36<00:08, 8858.60it/s] 82%|████████▏ | 326086/400000 [00:37<00:08, 8831.51it/s] 82%|████████▏ | 326970/400000 [00:37<00:08, 8801.56it/s] 82%|████████▏ | 327860/400000 [00:37<00:08, 8828.68it/s] 82%|████████▏ | 328743/400000 [00:37<00:08, 8602.88it/s] 82%|████████▏ | 329630/400000 [00:37<00:08, 8680.35it/s] 83%|████████▎ | 330500/400000 [00:37<00:08, 8588.00it/s] 83%|████████▎ | 331374/400000 [00:37<00:07, 8631.87it/s] 83%|████████▎ | 332250/400000 [00:37<00:07, 8667.52it/s] 83%|████████▎ | 333127/400000 [00:37<00:07, 8695.18it/s] 84%|████████▎ | 334022/400000 [00:37<00:07, 8766.90it/s] 84%|████████▎ | 334900/400000 [00:38<00:07, 8762.39it/s] 84%|████████▍ | 335777/400000 [00:38<00:07, 8758.31it/s] 84%|████████▍ | 336677/400000 [00:38<00:07, 8828.82it/s] 84%|████████▍ | 337566/400000 [00:38<00:07, 8844.40it/s] 85%|████████▍ | 338453/400000 [00:38<00:06, 8851.97it/s] 85%|████████▍ | 339349/400000 [00:38<00:06, 8883.16it/s] 85%|████████▌ | 340248/400000 [00:38<00:06, 8914.59it/s] 85%|████████▌ | 341140/400000 [00:38<00:06, 8901.53it/s] 86%|████████▌ | 342035/400000 [00:38<00:06, 8915.21it/s] 86%|████████▌ | 342927/400000 [00:38<00:06, 8889.01it/s] 86%|████████▌ | 343822/400000 [00:39<00:06, 8906.86it/s] 86%|████████▌ | 344724/400000 [00:39<00:06, 8939.88it/s] 86%|████████▋ | 345619/400000 [00:39<00:06, 8876.10it/s] 87%|████████▋ | 346517/400000 [00:39<00:06, 8904.52it/s] 87%|████████▋ | 347427/400000 [00:39<00:05, 8961.47it/s] 87%|████████▋ | 348324/400000 [00:39<00:05, 8888.52it/s] 87%|████████▋ | 349214/400000 [00:39<00:05, 8887.71it/s] 88%|████████▊ | 350116/400000 [00:39<00:05, 8924.62it/s] 88%|████████▊ | 351018/400000 [00:39<00:05, 8950.92it/s] 88%|████████▊ | 351924/400000 [00:39<00:05, 8982.28it/s] 88%|████████▊ | 352823/400000 [00:40<00:05, 8916.74it/s] 88%|████████▊ | 353724/400000 [00:40<00:05, 8942.66it/s] 89%|████████▊ | 354625/400000 [00:40<00:05, 8961.82it/s] 89%|████████▉ | 355523/400000 [00:40<00:04, 8964.32it/s] 89%|████████▉ | 356420/400000 [00:40<00:04, 8954.85it/s] 89%|████████▉ | 357316/400000 [00:40<00:04, 8889.43it/s] 90%|████████▉ | 358215/400000 [00:40<00:04, 8918.09it/s] 90%|████████▉ | 359130/400000 [00:40<00:04, 8984.14it/s] 90%|█████████ | 360029/400000 [00:40<00:04, 8943.35it/s] 90%|█████████ | 360929/400000 [00:41<00:04, 8958.83it/s] 90%|█████████ | 361826/400000 [00:41<00:04, 8957.57it/s] 91%|█████████ | 362728/400000 [00:41<00:04, 8973.53it/s] 91%|█████████ | 363630/400000 [00:41<00:04, 8984.73it/s] 91%|█████████ | 364529/400000 [00:41<00:03, 8980.98it/s] 91%|█████████▏| 365428/400000 [00:41<00:03, 8945.88it/s] 92%|█████████▏| 366323/400000 [00:41<00:03, 8942.76it/s] 92%|█████████▏| 367218/400000 [00:41<00:03, 8895.62it/s] 92%|█████████▏| 368137/400000 [00:41<00:03, 8980.40it/s] 92%|█████████▏| 369036/400000 [00:41<00:03, 8942.99it/s] 92%|█████████▏| 369939/400000 [00:42<00:03, 8968.47it/s] 93%|█████████▎| 370837/400000 [00:42<00:03, 8965.36it/s] 93%|█████████▎| 371736/400000 [00:42<00:03, 8970.69it/s] 93%|█████████▎| 372634/400000 [00:42<00:03, 8960.32it/s] 93%|█████████▎| 373531/400000 [00:42<00:02, 8951.65it/s] 94%|█████████▎| 374427/400000 [00:42<00:02, 8921.78it/s] 94%|█████████▍| 375320/400000 [00:42<00:02, 8902.11it/s] 94%|█████████▍| 376216/400000 [00:42<00:02, 8917.35it/s] 94%|█████████▍| 377122/400000 [00:42<00:02, 8957.96it/s] 95%|█████████▍| 378018/400000 [00:42<00:02, 8876.02it/s] 95%|█████████▍| 378934/400000 [00:43<00:02, 8958.62it/s] 95%|█████████▍| 379831/400000 [00:43<00:02, 8922.31it/s] 95%|█████████▌| 380726/400000 [00:43<00:02, 8927.96it/s] 95%|█████████▌| 381637/400000 [00:43<00:02, 8980.30it/s] 96%|█████████▌| 382536/400000 [00:43<00:01, 8953.41it/s] 96%|█████████▌| 383432/400000 [00:43<00:01, 8848.27it/s] 96%|█████████▌| 384318/400000 [00:43<00:01, 8839.79it/s] 96%|█████████▋| 385229/400000 [00:43<00:01, 8916.93it/s] 97%|█████████▋| 386149/400000 [00:43<00:01, 8999.32it/s] 97%|█████████▋| 387050/400000 [00:43<00:01, 8981.68it/s] 97%|█████████▋| 387953/400000 [00:44<00:01, 8994.89it/s] 97%|█████████▋| 388853/400000 [00:44<00:01, 8941.71it/s] 97%|█████████▋| 389748/400000 [00:44<00:01, 8921.73it/s] 98%|█████████▊| 390642/400000 [00:44<00:01, 8925.85it/s] 98%|█████████▊| 391550/400000 [00:44<00:00, 8969.04it/s] 98%|█████████▊| 392460/400000 [00:44<00:00, 9006.94it/s] 98%|█████████▊| 393361/400000 [00:44<00:00, 9002.04it/s] 99%|█████████▊| 394262/400000 [00:44<00:00, 8989.44it/s] 99%|█████████▉| 395162/400000 [00:44<00:00, 8980.76it/s] 99%|█████████▉| 396061/400000 [00:44<00:00, 8922.06it/s] 99%|█████████▉| 396954/400000 [00:45<00:00, 8890.04it/s] 99%|█████████▉| 397844/400000 [00:45<00:00, 8815.43it/s]100%|█████████▉| 398743/400000 [00:45<00:00, 8865.03it/s]100%|█████████▉| 399642/400000 [00:45<00:00, 8901.88it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8815.73it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f18fdcf3d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011085932773240817 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010895290502337708 	 Accuracy: 67

  model saves at 67% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15891 out of table with 15624 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15891 out of table with 15624 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 15:23:30.064541: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 15:23:30.069002: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-13 15:23:30.069147: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5625d5fe97b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 15:23:30.069161: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f190bf67160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4520 - accuracy: 0.5140 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4673 - accuracy: 0.5130
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4750 - accuracy: 0.5125
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5532 - accuracy: 0.5074
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5874 - accuracy: 0.5052
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6272 - accuracy: 0.5026
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6149 - accuracy: 0.5034
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6564 - accuracy: 0.5007
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
11000/25000 [============>.................] - ETA: 3s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 3s - loss: 7.6257 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6683 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6688 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6645 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 7s 272us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f185e203898> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1905451160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2431 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.2055 - val_crf_viterbi_accuracy: 0.6800

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
